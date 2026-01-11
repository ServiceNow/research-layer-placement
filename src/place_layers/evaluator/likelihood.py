"""Likelihood-based evaluator for multiple-choice tasks.

This evaluator measures accuracy on multiple-choice tasks (like MMLU) by
computing log-likelihood of each answer choice and selecting the highest.

Unlike generative evaluation which samples completions and checks correctness,
likelihood evaluation:
- Computes log P(choice | context) for each choice
- Selects argmax as the prediction
- Compares to gold answer index

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


# =============================================================================
# Evaluator Implementation
# =============================================================================


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

    def __init__(self, config: "LikelihoodEvaluatorConfig") -> None:
        super().__init__(config)
        self.config: LikelihoodEvaluatorConfig

        logger.info(
            f"Saving likelihood results to: {self._output_dir}/validation.jsonl"
        )

    # =========================================================================
    # Main Evaluation Entry Point
    # =========================================================================

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

    # =========================================================================
    # Evaluation Logic
    # =========================================================================

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
        """Evaluate a single task using chat-based log-prob scoring.

        Uses build_chat_messages for context and compute_chat_log_probs for
        scoring, which implements a diff-based approach that is chat template
        agnostic. Streams LikelihoodRecord entries to disk.
        """
        n_problems = len(remaining_indices)

        with create_progress() as progress:
            progress_task = progress.add_task(f"[cyan]{task_name}", total=n_problems)

            # Process in batches for efficiency
            batch_size = self.config.batch_size
            for i in range(0, n_problems, batch_size):
                batch_indices = remaining_indices[i : i + batch_size]
                batch_docs = [docs[idx] for idx in batch_indices]

                # Build chat tuples for all choices in this batch
                # Each problem contributes N chats where N = number of choices
                all_chats: list[tuple[list[ChatMessage], list[ChatMessage]]] = []
                gold_indices: list[int] = []
                choice_counts: list[int] = []
                choice_lengths: list[list[int]] = []  # For length normalization
                choice_texts: list[tuple[str, ...]] = []  # Raw choice strings
                context_messages_list: list[
                    tuple[ChatMessage, ...]
                ] = []  # Context per problem
                choice_messages_list: list[
                    tuple[ChatMessage, ...]
                ] = []  # Choices as ChatMessages

                for doc in batch_docs:
                    chats, gold_idx, lengths, raw_choices = self._build_choice_chats(
                        task, doc
                    )
                    all_chats.extend(chats)
                    gold_indices.append(gold_idx)
                    choice_counts.append(len(chats))
                    choice_lengths.append(lengths)
                    choice_texts.append(raw_choices)
                    # Extract context messages (same for all choices in this problem)
                    context_messages_list.append(tuple(chats[0][0]))
                    # Extract choice messages (the assistant message for each choice)
                    choice_messages_list.append(tuple(c[1][0] for c in chats))

                # Compute log-probs using chat-based API (diff-based scoring)
                # Returns ChatLogProbResult with full token-level detail
                results = await self._compute_chat_log_probs(all_chats)

                # Unpack results per document and create records
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
                    gold_indices,
                    choice_counts,
                    choice_lengths,
                    choice_texts,
                    context_messages_list,
                    choice_messages_list,
                    batch_indices,
                ):
                    # Extract results for this problem
                    doc_results = results[offset : offset + n_choices]
                    offset += n_choices

                    # Extract log probs for prediction
                    doc_log_probs = [r.total_log_prob for r in doc_results]

                    # Raw accuracy
                    pred = int(np.argmax(doc_log_probs))
                    correct = pred == gold_idx

                    # Length-normalized accuracy
                    normalized = np.array(doc_log_probs) / np.array(lengths)
                    pred_norm = int(np.argmax(normalized))
                    correct_norm = pred_norm == gold_idx

                    # Create record with full observability
                    record = LikelihoodRecord(
                        problem_id=f"{task_name}:{problem_idx}",
                        task_name=task_name,
                        # Chat messages
                        context_messages=context_msgs,
                        raw_choices=raw_choices,
                        choices=choice_msgs,
                        # Token-level data per choice
                        choice_token_ids=tuple(r.token_ids for r in doc_results),
                        choice_logprobs=tuple(r.logprobs for r in doc_results),
                        choice_answer_ranges=tuple(r.answer_range for r in doc_results),
                        # Scores
                        choice_log_probs=tuple(doc_log_probs),
                        choice_lengths=tuple(lengths),
                        # Predictions
                        gold_idx=gold_idx,
                        pred_idx=pred,
                        pred_idx_norm=pred_norm,
                        correct=correct,
                        correct_norm=correct_norm,
                    )
                    batch_records.append(record)

                # Stream records to disk
                self._trace_writer.append(batch_records)  # type: ignore[arg-type]

                progress.advance(progress_task, len(batch_docs))

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
        context_msgs = build_chat_messages(task, doc, task.config.num_fewshot or 0)

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


# =============================================================================
# Configuration
# =============================================================================


class LikelihoodEvaluatorConfig(TaskBasedEvaluatorConfig):
    """Configuration for likelihood-based evaluator.

    Used for multiple-choice tasks like MMLU, ARC, HellaSwag, etc.
    """

    _impl_class: ClassVar[type[BaseEvaluator]] = LikelihoodEvaluator

    evaluator_type: Literal["likelihood"] = "likelihood"

    # Batching
    batch_size: int = Field(
        default=32,
        description="Batch size for log-prob computation",
        gt=0,
    )
