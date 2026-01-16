"""Likelihood-based evaluator for multiple-choice tasks.

This evaluator measures accuracy on multiple-choice tasks (like MMLU) by
computing log-likelihood of each answer choice and selecting the highest.

Unlike generative evaluation which samples completions and checks correctness,
likelihood evaluation:
- Computes log P(choice | context) for each choice
- Selects argmax as the prediction
- Compares to gold answer index

Scoring Approach: Input Token Logprobs (NOT Generation)
-------------------------------------------------------
This evaluator uses **input token scoring**, not generation. The key insight
is that we don't need the model to generate anything - we just need the
probability it assigns to tokens we provide as input.

For each choice (e.g., "A", "B", "C", "D"), we:

1. **Send the answer as INPUT** (not as a generation target):
   ```
   messages = [
       {role: "user", content: "Question...\\nA. ...\\nB. ...\\nAnswer:"},
       {role: "assistant", content: "B"}  # <-- Answer is INPUT
   ]
   ```

2. **Get logprobs for input tokens** (backend-specific):

   **vLLM backend**: Request `prompt_logprobs` which returns logprobs for all
   INPUT tokens. vLLM requires `max_tokens >= 1`, so we set `max_tokens=1`,
   but we never read the generated token - only `prompt_logprobs`.

   **Transformers backend**: Tokenize the full sequence (context + answer),
   run a single forward pass, compute log_softmax over logits. No generation
   loop - just one forward pass on the complete input.

   Both backends produce logprobs for the full input sequence:
   ```
   [..., <|im_start|>, assistant, \\n, B, <|im_end|>, ...]
                                       ^
                                       └── We extract this logprob
   ```

3. **Use diff-based scoring**: Compare token sequences across all choices to
   find where they differ (the answer token position). This is chat-template
   agnostic - works regardless of how the template wraps messages.

This approach is more principled than generation-based scoring because:
- We measure exactly P(answer | context), not P(answer | context, model's prefix)
- No sampling variance - deterministic logprob extraction
- Faster - no autoregressive generation needed (just one forward pass)

Key design principles:
- Uses build_chat_messages for context construction (shared with generative)
- Uses compute_chat_log_probs with diff-based scoring (template-agnostic)
- Clean separation from generative evaluation (different paradigm)
- Supports both sync (Transformers) and async (vLLM) backends
- Supports length normalization (acc_norm) for fairer comparison
- Resumable via HydraConfig output directory

Usage
-----
    config = LikelihoodEvaluatorConfig(
        backend=TransformersBackendConfig(model_path="/path/to/model", ...),
        tasks=["mmlu_abstract_algebra"],
        num_fewshot=5,
        ...
    )
    evaluator = config.create_evaluator()
    score = await evaluator.evaluate(placement)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from pydantic import Field

from place_layers.backend.base import ChatMessage
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.lm_eval_utils import build_chat_messages
from place_layers.evaluator.progress import create_progress
from place_layers.evaluator.task_based_evaluator import (
    TaskBasedEvaluator,
    TaskBasedEvaluatorConfig,
)
from place_layers.evaluator.traces.types import LikelihoodRecord
from place_layers.evaluator.traces.writer import summarize_likelihood

if TYPE_CHECKING:
    from lm_eval.api.task import Task

    from place_layers.evaluator.lm_eval_utils import Docs

logger = logging.getLogger(__name__)


@dataclass
class _PreparedBatch:
    """Prepared batch data for async prefetching."""

    batch_indices: list[int]
    all_chats: list[tuple[list[ChatMessage], list[ChatMessage]]]
    gold_indices: list[int]
    choice_counts: list[int]
    choice_lengths: list[list[int]]
    choice_texts: list[tuple[str, ...]]
    context_messages_list: list[tuple[ChatMessage, ...]]
    choice_messages_list: list[tuple[ChatMessage, ...]]


class LikelihoodEvaluator(TaskBasedEvaluator):
    """Evaluator for multiple-choice tasks using log-likelihood scoring.

    For each problem:
    1. Build chat context via build_chat_messages (few-shot + question)
    2. Create (context, completion) chat tuples for each choice
    3. Compute log P(completion | context) via compute_chat_log_probs
    4. Select argmax as prediction
    5. Compare to gold answer

    Uses diff-based scoring that is chat template agnostic - compares token
    sequences across choices to find answer tokens, then extracts logprobs.
    Supports both sync (Transformers) and async (vLLM) backends.

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "LikelihoodEvaluatorConfig"

    def __init__(self, config: "LikelihoodEvaluatorConfig") -> None:
        super().__init__(config)
        logger.info(
            f"Saving likelihood results to: {self._output_dir}/validation.jsonl"
        )

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate a placement configuration using likelihood-based accuracy.

        Args:
            parameters: Layer placement to evaluate, or None for default config.
                       When None, no placement switching is performed (useful for
                       non-supernet models or backends that don't support placement).

        Returns:
            Accuracy (fraction of problems where argmax prediction is correct).
        """
        if parameters is not None:
            placement_start = time.perf_counter()
            await self._set_placement(parameters)
            placement_time = time.perf_counter() - placement_start
            logger.debug(f"Set placement took {placement_time:.2f}s")
        else:
            logger.debug("No placement specified, using default configuration")

        # Run evaluation (streams records to disk)
        await self._run_evaluation()

        # Compute summary from disk (single source of truth)
        summary = summarize_likelihood(self._output_dir)
        # Use print() for summary - logger.info goes to file only (console is WARNING+)
        print(f"\n{summary}")

        # Store summary for access to additional metrics (e.g., brier_score)
        self._last_summary = summary

        return summary.accuracy

    @property
    def brier_score(self) -> float | None:
        """Brier score from the last evaluation (None if not yet evaluated)."""
        if hasattr(self, "_last_summary") and self._last_summary is not None:
            return self._last_summary.brier_score
        return None

    async def _run_evaluation(self) -> None:
        """Run likelihood evaluation across all tasks, streaming records to disk."""
        self._last_inference_duration = 0.0

        for task_spec in self.config.tasks:
            prepared = self._prepare_task(task_spec, "multiple_choice")

            if prepared is None:
                # Task already complete
                continue

            task, task_name, docs, remaining_indices = prepared

            logger.info(
                f"Task '{task_name}': evaluating {len(remaining_indices)} problems"
            )

            # Evaluate task (streams records to disk) - this is the inference time
            start = time.perf_counter()
            await self._evaluate_task(task, task_name, docs, remaining_indices)
            self._last_inference_duration += time.perf_counter() - start

    def _prepare_batch(
        self,
        task: "Task",
        docs: "Docs",
        batch_indices: list[int],
    ) -> _PreparedBatch:
        """Prepare a batch by building chat tuples (CPU work).

        This is extracted to enable async prefetching - preparing the next batch
        while GPU inference runs on the current batch.
        """
        batch_docs = [docs[idx] for idx in batch_indices]

        all_chats: list[tuple[list[ChatMessage], list[ChatMessage]]] = []
        gold_indices: list[int] = []
        choice_counts: list[int] = []
        choice_lengths: list[list[int]] = []
        choice_texts: list[tuple[str, ...]] = []
        context_messages_list: list[tuple[ChatMessage, ...]] = []
        choice_messages_list: list[tuple[ChatMessage, ...]] = []

        for doc in batch_docs:
            chats, gold_idx, lengths, raw_choices = self._build_choice_chats(task, doc)
            all_chats.extend(chats)
            gold_indices.append(gold_idx)
            choice_counts.append(len(chats))
            choice_lengths.append(lengths)
            choice_texts.append(raw_choices)
            context_messages_list.append(tuple(chats[0][0]))
            choice_messages_list.append(tuple(c[1][0] for c in chats))

        return _PreparedBatch(
            batch_indices=batch_indices,
            all_chats=all_chats,
            gold_indices=gold_indices,
            choice_counts=choice_counts,
            choice_lengths=choice_lengths,
            choice_texts=choice_texts,
            context_messages_list=context_messages_list,
            choice_messages_list=choice_messages_list,
        )

    async def _evaluate_task(
        self,
        task: "Task",
        task_name: str,
        docs: "Docs",
        remaining_indices: list[int],
    ) -> None:
        """Evaluate a single task using chat-based log-prob scoring.

        Uses async prefetching to prepare the next batch while GPU inference
        runs on the current batch. This overlaps CPU work (chat building) with
        GPU work (inference) for better throughput.
        """
        n_problems = len(remaining_indices)
        batch_size = self.config.batch_size

        # Compute batch boundaries
        batch_starts = list(range(0, n_problems, batch_size))
        num_batches = len(batch_starts)

        if num_batches == 0:
            return

        with create_progress() as progress:
            progress_task = progress.add_task(f"[cyan]{task_name}", total=n_problems)

            # Prepare first batch synchronously
            first_indices = remaining_indices[batch_starts[0] : batch_starts[0] + batch_size]
            current_batch = self._prepare_batch(task, docs, first_indices)

            for batch_idx in range(num_batches):
                # Start prefetching next batch in background thread (if not last batch)
                next_batch_task = None
                if batch_idx + 1 < num_batches:
                    next_start = batch_starts[batch_idx + 1]
                    next_indices = remaining_indices[next_start : next_start + batch_size]
                    next_batch_task = asyncio.create_task(
                        asyncio.to_thread(self._prepare_batch, task, docs, next_indices)
                    )

                # Run inference on current batch (GPU work)
                results = await self._compute_chat_log_probs(
                    current_batch.all_chats, use_chat_template=self.config.use_chat_template
                )

                # Process results and create records
                batch_records = self._process_batch_results(
                    task_name, current_batch, results
                )

                # Stream records to disk
                self._trace_writer.append(batch_records)  # type: ignore[arg-type]
                progress.advance(progress_task, len(current_batch.batch_indices))

                # Wait for next batch to be ready (if prefetched)
                if next_batch_task is not None:
                    current_batch = await next_batch_task

    def _process_batch_results(
        self,
        task_name: str,
        batch: _PreparedBatch,
        results: list,
    ) -> list[LikelihoodRecord]:
        """Process inference results and create records."""
        offset = 0
        batch_records: list[LikelihoodRecord] = []

        for (
            gold_idx,
            n_choices,
            lengths,
            raw_choices,
            context_msgs,
            choice_msgs,
            problem_idx,
        ) in zip(
            batch.gold_indices,
            batch.choice_counts,
            batch.choice_lengths,
            batch.choice_texts,
            batch.context_messages_list,
            batch.choice_messages_list,
            batch.batch_indices,
        ):
            doc_results = results[offset : offset + n_choices]
            offset += n_choices

            doc_log_probs = [r.total_log_prob for r in doc_results]

            pred = int(np.argmax(doc_log_probs))
            correct = pred == gold_idx

            normalized = np.array(doc_log_probs) / np.array(lengths)
            pred_norm = int(np.argmax(normalized))
            correct_norm = pred_norm == gold_idx

            record = LikelihoodRecord(
                problem_id=f"{task_name}:{problem_idx}",
                task_name=task_name,
                context_messages=context_msgs,
                raw_choices=raw_choices,
                choices=choice_msgs,
                choice_token_ids=tuple(r.token_ids for r in doc_results),
                choice_logprobs=tuple(r.logprobs for r in doc_results),
                choice_answer_ranges=tuple(r.answer_range for r in doc_results),
                choice_log_probs=tuple(doc_log_probs),
                choice_lengths=tuple(lengths),
                gold_idx=gold_idx,
                pred_idx=pred,
                pred_idx_norm=pred_norm,
                correct=correct,
                correct_norm=correct_norm,
            )
            batch_records.append(record)

        return batch_records

    def _build_choice_chats(
        self, task: "Task", doc: dict
    ) -> tuple[
        list[tuple[list[ChatMessage], list[ChatMessage]]],
        int,
        list[int],
        tuple[str, ...],
    ]:
        """Build chat message tuples for each choice.

        Uses build_chat_messages for context (few-shot + question), then creates
        a completion message for each choice. The compute_chat_log_probs method
        uses diff-based scoring that is template-agnostic.

        Args:
            task: The lm-eval task.
            doc: The problem document.

        Returns:
            Tuple of:
            - List of (context_messages, completion_messages) tuples, one per choice
            - Gold answer index
            - List of choice lengths (in characters) for normalization
            - Tuple of choice texts (for recording)
        """
        # Get context messages using the shared builder
        context_msgs = build_chat_messages(
            task,
            doc,
            task.config.num_fewshot or 0,
            system_message=self.config.system_message,
        )

        # Get choices and gold answer
        # doc_to_choice is on ConfigurableTask, but all YAML-defined tasks use it
        choices: list[str] = task.doc_to_choice(doc)  # type: ignore[attr-defined]
        gold_target = task.doc_to_target(doc)

        # Determine gold index from target
        # Target can be: int (index), str (choice text), or list (multiple correct)
        if isinstance(gold_target, int):
            gold_idx = gold_target
        elif isinstance(gold_target, str):
            gold_idx = choices.index(gold_target)
        else:
            # For multiple correct answers, use first one
            # This is rare but handle it gracefully
            gold_idx = gold_target[0] if gold_target else 0

        # Build chat tuples for each choice
        chats: list[tuple[list[ChatMessage], list[ChatMessage]]] = []
        lengths: list[int] = []

        for choice in choices:
            # Completion is the assistant message with this choice
            completion_msgs: list[ChatMessage] = [
                {"role": "assistant", "content": choice}
            ]
            chats.append((context_msgs, completion_msgs))
            lengths.append(len(choice))  # Character length for normalization

        return chats, gold_idx, lengths, tuple(choices)


class LikelihoodEvaluatorConfig(TaskBasedEvaluatorConfig):
    """Config for multiple-choice tasks (MMLU, ARC, HellaSwag, etc.)."""

    _impl_class: ClassVar[type[BaseEvaluator]] = LikelihoodEvaluator

    evaluator_type: Literal["likelihood"] = "likelihood"

    # Batching
    batch_size: int = Field(
        default=32,
        description="Batch size for log-prob computation",
        gt=0,
    )

    # Chat template control
    use_chat_template: bool = Field(
        default=True,
        description="Whether to apply the tokenizer's chat template. "
        "Set to False to use raw text without special tokens.",
    )
    system_message: str | None = Field(
        default=None,
        description="Custom system message to prepend. If None and use_chat_template=True, "
        "the tokenizer's default system message is used. Set to empty string to disable "
        "system message entirely.",
    )
