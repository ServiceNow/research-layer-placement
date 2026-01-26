"""Trace precomputation evaluator for importance-weighted estimation.

This evaluator generates reasoning traces from a model and saves them
with their log-probabilities. These traces are used by the
ImportanceWeightedEvaluator for fast accuracy estimation during search.

Unlike GenerativeEvaluator which only checks correctness, this evaluator
also records the model's log-probability for each trace (needed for
importance sampling).

Usage
-----
    config = TracePrecomputeEvaluatorConfig(
        backend=VLLMBackendConfig(...),
        tasks=["gsm8k"],
        samples_per_problem=8,
        ...
    )
    async with config.create_evaluator() as evaluator:
        accuracy = await evaluator.evaluate(None)  # Returns teacher accuracy
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field

from place_layers.clients.types import GenerationResult
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.base_generative_evaluator import (
    BaseGenerativeEvaluator,
    BaseGenerativeEvaluatorConfig,
)
from place_layers.evaluator.lm_eval_utils import check_correctness
from place_layers.evaluator.traces.types import Trace

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


class TracePrecomputeEvaluator(BaseGenerativeEvaluator):
    """Evaluator that generates and saves traces with log-probabilities.

    This is used to pre-compute teacher traces for importance-weighted
    evaluation. The traces include the model's log-probability which
    enables importance sampling during search.

    Returns the teacher model's accuracy as the evaluation score.

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            accuracy = await evaluator.evaluate(placement)
    """

    config: "TracePrecomputeEvaluatorConfig"

    def __init__(self, config: "TracePrecomputeEvaluatorConfig") -> None:
        super().__init__(config)

    def _get_fewshot_seed(self) -> int | None:
        """Return seed for few-shot sampling."""
        return self.config.seed

    def _make_records(
        self, task: "Task", doc: dict, idx: int, result: GenerationResult
    ) -> list[Trace]:
        """Create Trace records from a generation result.

        Unlike GenerativeEvaluator._make_records, this includes teacher_log_prob
        which is needed for importance sampling.

        Truncated sample handling depends on evaluate_truncated_samples config:
        - When False (default): truncated samples marked incorrect
        - When True: truncated samples evaluated for correctness
        """
        task_name = task.config.task or ""
        traces = []
        prompt_tokens = tuple(result.prompt_token_ids)
        problem_id = f"{task_name}:{idx}"

        for sample_idx, sample in enumerate(result.samples):
            if sample.truncated and not self.config.evaluate_truncated_samples:
                # Truncated and we don't evaluate truncated -> mark incorrect
                logger.debug(f"Truncated sample {problem_id}:{sample_idx}")
                correct = False
            else:
                correct = check_correctness(task, doc, sample.text)

            trace = Trace(
                prompt_tokens=prompt_tokens,
                completion_tokens=tuple(sample.token_ids),
                correct=correct,
                truncated=sample.truncated,
                teacher_log_prob=sample.total_logprob,
                problem_id=problem_id,
                task_name=task_name,
                sample_idx=sample_idx,
                completion_text=sample.text,
            )
            traces.append(trace)

        return traces


class TracePrecomputeEvaluatorConfig(BaseGenerativeEvaluatorConfig):
    """Generates teacher traces with log-probabilities for importance-weighted evaluation."""

    _impl_class: ClassVar[type[BaseEvaluator]] = TracePrecomputeEvaluator

    evaluator_type: Literal["trace_precompute"] = "trace_precompute"

    # Random seed for reproducibility
    seed: int = Field(
        default=42,
        description="Random seed for few-shot sampling",
    )
