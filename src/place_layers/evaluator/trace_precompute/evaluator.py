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

from place_layers.core.types import LayerPlacement
from place_layers.evaluator.backend_evaluator import BackendEvaluator
from place_layers.evaluator.backend_evaluator_config import TaskBasedEvaluatorConfig
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.backend.base import (
    AsyncBackend,
    SyncBackend,
)
from place_layers.evaluator.traces.types import Trace
from place_layers.evaluator.traces.writer import summarize_generations
from place_layers.backend.vllm import GenerationResult
from place_layers.evaluator.lm_eval_utils import check_correctness
from place_layers.evaluator.task_runner import (
    prepare_task,
    run_generation_async,
    run_generation_sync,
)

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


# =============================================================================
# Record Factory
# =============================================================================


def _make_trace_records(
    task: "Task", doc: dict, idx: int, result: GenerationResult
) -> list[Trace]:
    """Create Trace records from a generation result.

    Unlike _make_generation_records, this includes teacher_log_prob
    which is needed for importance sampling.

    Skips truncated samples (they don't have valid log-probs).
    Returns empty list if all samples are truncated.
    """
    task_name = task.config.task
    traces = []
    prompt_tokens = tuple(result.prompt_token_ids)
    problem_id = f"{task_name}:{idx}"

    for sample_idx, sample in enumerate(result.samples):
        if sample.truncated:
            logger.debug(f"Skipping truncated sample {problem_id}:{sample_idx}")
            continue

        correct = check_correctness(task, doc, sample.text)
        trace = Trace(
            prompt_tokens=prompt_tokens,
            completion_tokens=tuple(sample.token_ids),
            correct=correct,
            teacher_log_prob=sample.total_logprob,
            problem_id=problem_id,
            task_name=task_name,
            sample_idx=sample_idx,
            completion_text=sample.text,
        )
        traces.append(trace)

    return traces


# =============================================================================
# Evaluator Implementation
# =============================================================================


class TracePrecomputeEvaluator(BackendEvaluator):
    """Evaluator that generates and saves traces with log-probabilities.

    This is used to pre-compute teacher traces for importance-weighted
    evaluation. The traces include the model's log-probability which
    enables importance sampling during search.

    Returns the teacher model's accuracy as the evaluation score.

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            accuracy = await evaluator.evaluate(placement)
    """

    def __init__(self, config: "TracePrecomputeEvaluatorConfig") -> None:
        super().__init__(config)
        self.config: TracePrecomputeEvaluatorConfig

        logger.info(f"Saving traces to: {self._output_dir}/validation.jsonl")

    # =========================================================================
    # Main Evaluation Entry Point
    # =========================================================================

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Generate traces for a placement configuration.

        Args:
            parameters: Layer placement to use, or None for default config.
                       When None, no placement switching is performed.

        Returns:
            Teacher accuracy (fraction of problems with any correct sample).
        """
        if parameters is not None:
            await self._set_placement(parameters)
        else:
            logger.debug("No placement specified, using default configuration")

        # Dispatch to async or sync generation
        if isinstance(self._backend, AsyncBackend):
            await self._run_tasks_async()
        elif isinstance(self._backend, SyncBackend):
            self._run_tasks_sync()
        else:
            raise TypeError(f"Unknown backend type: {type(self._backend)}")

        # Compute summary from disk (single source of truth)
        summary = summarize_generations(self._output_dir)
        logger.info(f"\n{summary}")

        return summary.problem_accuracy

    # =========================================================================
    # Task Running
    # =========================================================================

    def _run_tasks_sync(self) -> None:
        """Run all tasks synchronously."""
        assert isinstance(self._backend, SyncBackend)

        for task_spec in self.config.tasks:
            prepared = prepare_task(
                task_spec=task_spec,
                output_dir=self._output_dir,
                num_fewshot=self.config.num_fewshot,
                max_problems=self.config.max_problems_per_task,
                split=self.config.split,
                seed=self.config.seed,
            )

            if prepared is None:
                continue

            task, task_name, docs, remaining = prepared

            run_generation_sync(
                task=task,
                task_name=task_name,
                docs=docs,
                remaining_indices=remaining,
                backend=self._backend,
                writer=self._trace_writer,
                record_factory=_make_trace_records,
                samples_per_problem=self.config.samples_per_problem,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

    async def _run_tasks_async(self) -> None:
        """Run all tasks asynchronously."""
        assert isinstance(self._backend, AsyncBackend)

        for task_spec in self.config.tasks:
            prepared = prepare_task(
                task_spec=task_spec,
                output_dir=self._output_dir,
                num_fewshot=self.config.num_fewshot,
                max_problems=self.config.max_problems_per_task,
                split=self.config.split,
                seed=self.config.seed,
            )

            if prepared is None:
                continue

            task, task_name, docs, remaining = prepared

            await run_generation_async(
                task=task,
                task_name=task_name,
                docs=docs,
                remaining_indices=remaining,
                backend=self._backend,
                writer=self._trace_writer,
                record_factory=_make_trace_records,
                samples_per_problem=self.config.samples_per_problem,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )


# =============================================================================
# Configuration
# =============================================================================


class TracePrecomputeEvaluatorConfig(TaskBasedEvaluatorConfig):
    """Configuration for trace precomputation evaluator.

    Used to generate teacher traces with log-probabilities for
    importance-weighted accuracy estimation.
    """

    _impl_class: ClassVar[type[BaseEvaluator]] = TracePrecomputeEvaluator

    evaluator_type: Literal["trace_precompute"] = "trace_precompute"

    # Generation parameters
    samples_per_problem: int = Field(
        description="Number of traces to sample per problem",
        gt=0,
    )
    max_new_tokens: int = Field(
        description="Maximum tokens to generate per trace",
        gt=0,
    )
    temperature: float = Field(
        description="Sampling temperature (should be > 0 for diversity)",
        gt=0.0,
    )
    top_p: float = Field(
        description="Nucleus sampling parameter",
        gt=0.0,
        le=1.0,
    )

    # Random seed for reproducibility
    seed: int = Field(
        default=42,
        description="Random seed for few-shot sampling",
    )
