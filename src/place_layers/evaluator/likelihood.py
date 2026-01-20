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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from pydantic import Field

from place_layers.backend.base import ChatMessage, LogProbResult
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


@dataclass(frozen=True)
class PredictionResult:
    """Result of computing predictions from log-probabilities."""

    log_probs: list[float]
    """Log-probability for each choice."""
    pred_idx: int
    """Predicted choice index (argmax of log_probs)."""
    correct: bool
    """Whether prediction matches gold answer."""
    pred_idx_norm: int
    """Predicted choice index with length normalization."""
    correct_norm: bool
    """Whether length-normalized prediction matches gold answer."""


@dataclass(frozen=True)
class ChoiceMetadata:
    """Metadata about choices for a multiple-choice problem."""

    gold_idx: int
    """Index of the correct answer."""
    lengths: list[int]
    """Character length of each choice (for normalization)."""
    raw_choices: tuple[str, ...]
    """Original choice text strings."""


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
            await self._set_placement(parameters)
        else:
            logger.debug("No placement specified, using default configuration")

        # Run evaluation (streams records to disk)
        await self._run_evaluation()

        # Compute summary from disk (single source of truth)
        summary = summarize_likelihood(self._output_dir)
        logger.info(f"\n{summary}")

        return summary.accuracy

    async def _run_evaluation(self) -> None:
        """Run likelihood evaluation across all tasks, streaming records to disk."""
        for task_spec in self.config.tasks:
            # Use _prepare_task for loading and resumption
            prepared = self._prepare_task(task_spec, "multiple_choice")

            if prepared is None:
                # Task already complete
                continue

            task, task_name, docs, remaining_indices = prepared

            logger.info(
                f"Task '{task_name}': evaluating {len(remaining_indices)} problems"
            )

            # Evaluate task (streams records to disk)
            await self._evaluate_task(task, task_name, docs, remaining_indices)

    async def _evaluate_task(
        self,
        task: "Task",
        task_name: str,
        docs: "Docs",
        remaining_indices: list[int],
    ) -> None:
        """Evaluate a single task using log-prob scoring.

        Dispatches to chat-based or text-based scoring based on config.use_chat_template.
        Streams LikelihoodRecord entries to disk.
        """
        if self.config.use_chat_template:
            await self._evaluate_task_chat(task, task_name, docs, remaining_indices)
        else:
            await self._evaluate_task_text(task, task_name, docs, remaining_indices)

    async def _evaluate_task_chat(
        self,
        task: "Task",
        task_name: str,
        docs: "Docs",
        remaining_indices: list[int],
    ) -> None:
        """Evaluate using chat-based log-prob scoring (with chat template)."""
        n_problems = len(remaining_indices)

        with create_progress() as progress:
            progress_task = progress.add_task(f"[cyan]{task_name}", total=n_problems)

            batch_size = self.config.batch_size
            for i in range(0, n_problems, batch_size):
                batch_indices = remaining_indices[i : i + batch_size]
                batch_docs = [docs[idx] for idx in batch_indices]

                # Build chat tuples for all choices in this batch
                all_chats: list[tuple[list[ChatMessage], list[ChatMessage]]] = []
                metadata_list: list[ChoiceMetadata] = []
                choice_counts: list[int] = []
                context_messages_list: list[tuple[ChatMessage, ...]] = []
                choice_messages_list: list[tuple[ChatMessage, ...]] = []

                for doc in batch_docs:
                    chats, metadata = self._build_choice_chats(task, doc)
                    all_chats.extend(chats)
                    metadata_list.append(metadata)
                    choice_counts.append(len(chats))
                    context_messages_list.append(tuple(chats[0][0]))
                    choice_messages_list.append(tuple(c[1][0] for c in chats))

                # Compute log-probs using chat-based API
                results = await self._compute_chat_log_probs(all_chats)

                # Unpack results and create records
                offset = 0
                batch_records: list[LikelihoodRecord] = []
                for metadata, n_choices, context_msgs, choice_msgs, problem_idx in zip(
                    metadata_list,
                    choice_counts,
                    context_messages_list,
                    choice_messages_list,
                    batch_indices,
                ):
                    doc_results = results[offset : offset + n_choices]
                    offset += n_choices

                    pred = self._compute_prediction(
                        doc_results, metadata.gold_idx, metadata.lengths
                    )

                    record = LikelihoodRecord(
                        problem_id=f"{task_name}:{problem_idx}",
                        task_name=task_name,
                        context_messages=context_msgs,
                        raw_choices=metadata.raw_choices,
                        choices=choice_msgs,
                        choice_token_ids=tuple(r.token_ids for r in doc_results),
                        choice_logprobs=tuple(r.logprobs for r in doc_results),
                        choice_answer_ranges=tuple(r.answer_range for r in doc_results),
                        choice_log_probs=tuple(pred.log_probs),
                        choice_lengths=tuple(metadata.lengths),
                        gold_idx=metadata.gold_idx,
                        pred_idx=pred.pred_idx,
                        pred_idx_norm=pred.pred_idx_norm,
                        correct=pred.correct,
                        correct_norm=pred.correct_norm,
                    )
                    batch_records.append(record)

                self._trace_writer.append(batch_records)  # type: ignore[arg-type]
                progress.advance(progress_task, len(batch_docs))

    async def _evaluate_task_text(
        self,
        task: "Task",
        task_name: str,
        docs: "Docs",
        remaining_indices: list[int],
    ) -> None:
        """Evaluate using text-based log-prob scoring (no chat template).

        Uses the same diff-based scoring approach as chat-based evaluation,
        but without chat template formatting. The backend handles diff-based
        scoring internally, so we just need to build the text groups.
        """
        n_problems = len(remaining_indices)

        with create_progress() as progress:
            progress_task = progress.add_task(f"[cyan]{task_name}", total=n_problems)

            batch_size = self.config.batch_size
            for i in range(0, n_problems, batch_size):
                batch_indices = remaining_indices[i : i + batch_size]
                batch_docs = [docs[idx] for idx in batch_indices]

                # Build text groups for all problems in batch
                text_groups: list[tuple[str, list[str]]] = []
                metadata_list: list[ChoiceMetadata] = []
                choice_counts: list[int] = []
                context_strings: list[str] = []

                for doc in batch_docs:
                    context, choices, metadata = self._build_choice_texts(task, doc)
                    text_groups.append((context, choices))
                    metadata_list.append(metadata)
                    choice_counts.append(len(choices))
                    context_strings.append(context)

                # Compute log-probs using text-based API (diff-based scoring in backend)
                results = await self._compute_text_log_probs(text_groups)

                # Unpack results and create records
                offset = 0
                batch_records: list[LikelihoodRecord] = []
                for metadata, n_choices, context_str, problem_idx in zip(
                    metadata_list,
                    choice_counts,
                    context_strings,
                    batch_indices,
                ):
                    doc_results = results[offset : offset + n_choices]
                    offset += n_choices

                    pred = self._compute_prediction(
                        doc_results, metadata.gold_idx, metadata.lengths
                    )

                    # For text-based, store context as a single user message
                    context_msgs: tuple[ChatMessage, ...] = (
                        {"role": "user", "content": context_str},
                    )
                    # Choices as assistant messages
                    choice_msgs: tuple[ChatMessage, ...] = tuple(
                        {"role": "assistant", "content": choice}
                        for choice in metadata.raw_choices
                    )

                    record = LikelihoodRecord(
                        problem_id=f"{task_name}:{problem_idx}",
                        task_name=task_name,
                        context_messages=context_msgs,
                        raw_choices=metadata.raw_choices,
                        choices=choice_msgs,
                        choice_token_ids=tuple(r.token_ids for r in doc_results),
                        choice_logprobs=tuple(r.logprobs for r in doc_results),
                        choice_answer_ranges=tuple(r.answer_range for r in doc_results),
                        choice_log_probs=tuple(pred.log_probs),
                        choice_lengths=tuple(metadata.lengths),
                        gold_idx=metadata.gold_idx,
                        pred_idx=pred.pred_idx,
                        pred_idx_norm=pred.pred_idx_norm,
                        correct=pred.correct,
                        correct_norm=pred.correct_norm,
                    )
                    batch_records.append(record)

                self._trace_writer.append(batch_records)  # type: ignore[arg-type]
                progress.advance(progress_task, len(batch_docs))

    def _get_gold_idx(self, task: "Task", doc: dict, choices: list[str]) -> int:
        """Extract gold answer index from task and document.

        Handles the various formats that lm-eval tasks use to specify the correct answer:
        - int: Direct index
        - str: Choice text (looked up in choices list)
        - list: Multiple correct answers (uses first one)

        Args:
            task: The lm-eval task.
            doc: The problem document.
            choices: List of choice strings.

        Returns:
            Index of the gold answer in the choices list.
        """
        gold_target = task.doc_to_target(doc)
        if isinstance(gold_target, int):
            return gold_target
        elif isinstance(gold_target, str):
            return choices.index(gold_target)
        else:
            # For multiple correct answers, use first one
            return gold_target[0] if gold_target else 0

    def _compute_prediction(
        self,
        results: list[LogProbResult],
        gold_idx: int,
        lengths: list[int],
    ) -> PredictionResult:
        """Compute predictions from log-prob results.

        Takes the log-probabilities from the backend and computes:
        - Raw prediction (argmax of log-probs)
        - Length-normalized prediction (argmax of log-probs / lengths)

        Args:
            results: Log-prob results from backend (one per choice).
            gold_idx: Index of the correct answer.
            lengths: Character lengths of each choice (for normalization).

        Returns:
            PredictionResult with log_probs, predictions, and correctness flags.
        """
        log_probs = [r.total_log_prob for r in results]
        pred_idx = int(np.argmax(log_probs))
        correct = pred_idx == gold_idx

        normalized = np.array(log_probs) / np.array(lengths)
        pred_idx_norm = int(np.argmax(normalized))
        correct_norm = pred_idx_norm == gold_idx

        return PredictionResult(
            log_probs=log_probs,
            pred_idx=pred_idx,
            correct=correct,
            pred_idx_norm=pred_idx_norm,
            correct_norm=correct_norm,
        )

    def _build_choice_chats(
        self, task: "Task", doc: dict
    ) -> tuple[list[tuple[list[ChatMessage], list[ChatMessage]]], ChoiceMetadata]:
        """Build chat message tuples for each choice.

        Uses build_chat_messages for context (few-shot + question), then creates
        a completion message for each choice. The compute_chat_log_probs method
        uses diff-based scoring that is template-agnostic.

        Args:
            task: The lm-eval task.
            doc: The problem document.

        Returns:
            Tuple of (chats, metadata) where:
            - chats: List of (context_messages, completion_messages) tuples
            - metadata: ChoiceMetadata with gold_idx, lengths, and raw_choices
        """
        # Get context messages using the shared builder
        context_msgs = build_chat_messages(task, doc, task.config.num_fewshot or 0)

        # Get choices and gold answer
        # doc_to_choice is on ConfigurableTask, but all YAML-defined tasks use it
        choices: list[str] = task.doc_to_choice(doc)  # type: ignore[attr-defined]
        gold_idx = self._get_gold_idx(task, doc, choices)

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

        metadata = ChoiceMetadata(
            gold_idx=gold_idx,
            lengths=lengths,
            raw_choices=tuple(choices),
        )
        return chats, metadata

    def _build_choice_texts(
        self, task: "Task", doc: dict
    ) -> tuple[str, list[str], ChoiceMetadata]:
        """Build raw text context and choices for text-based scoring.

        This method constructs prompts WITHOUT chat template formatting.
        Used when use_chat_template=False.

        Args:
            task: The lm-eval task.
            doc: The problem document.

        Returns:
            Tuple of (context, choices, metadata) where:
            - context: Few-shot examples + question as raw text
            - choices: List of choice strings (continuations)
            - metadata: ChoiceMetadata with gold_idx, lengths, and raw_choices
        """
        # Build context: few-shot examples + question as raw text
        num_fewshot = task.config.num_fewshot or 0
        context: str
        if num_fewshot > 0:
            # fewshot_context returns few-shot examples AND the current question
            fewshot_ctx = task.fewshot_context(doc, num_fewshot)
            context = str(fewshot_ctx) if fewshot_ctx else ""
        else:
            # Zero-shot: just the question
            question_raw = task.doc_to_text(doc)
            context = str(question_raw) if question_raw is not None else ""

        # Get choices and gold answer
        raw_choices: list[str] = task.doc_to_choice(doc)  # type: ignore[attr-defined]
        gold_idx = self._get_gold_idx(task, doc, raw_choices)

        # Match lm-eval's _encode_pair approach:
        # 1. Move trailing whitespace from context to continuation
        # 2. Prepend target_delimiter to choices
        # This ensures proper tokenization at word boundaries
        n_spaces = len(context) - len(context.rstrip())
        trailing_ws = context[-n_spaces:] if n_spaces > 0 else ""
        context = context.rstrip()

        delimiter = task.config.target_delimiter
        choices = [trailing_ws + delimiter + c for c in raw_choices]

        # Build lengths for normalization (use raw choices for char-level normalization)
        lengths = [len(choice) for choice in raw_choices]

        metadata = ChoiceMetadata(
            gold_idx=gold_idx,
            lengths=lengths,
            raw_choices=tuple(raw_choices),  # Original choices without space prefix
        )
        return context, choices, metadata


class LikelihoodEvaluatorConfig(TaskBasedEvaluatorConfig):
    """Config for multiple-choice tasks (MMLU, ARC, HellaSwag, etc.)."""

    _impl_class: ClassVar[type[BaseEvaluator]] = LikelihoodEvaluator

    evaluator_type: Literal["likelihood"] = "likelihood"

    # Chat template control
    use_chat_template: bool = Field(
        default=True,
        description="Whether to use chat template formatting. When False, prompts are "
        "formatted as raw text (context + choice) without any chat structure. "
        "Use False for evaluating base models or when chat template is undesired.",
    )

    # Batching
    batch_size: int = Field(
        default=32,
        description="Batch size for log-prob computation",
        gt=0,
    )
