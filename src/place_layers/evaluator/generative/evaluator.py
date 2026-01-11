"""Generative evaluator for direct accuracy measurement.

This evaluator measures task accuracy by actually generating completions
from the model (as opposed to importance-weighted estimation from traces).

Key design principles:
- validation.jsonl is the single source of truth
- Resumable: scans completed problems from disk, skips them
- Lean: only stores Generation records, computes metrics on demand
- Summary computed via summarize_generations() at end

It supports both async backends (vLLM) and sync backends (Transformers),
dispatching to the appropriate execution path based on backend type.

Usage
-----
    config = GenerativeEvaluatorConfig(
        backend=TransformersBackendConfig(model_path="/path/to/model", ...),
        tasks=["gsm8k"],
        ...
    )
    evaluator = config.create_evaluator()
    score = await evaluator.evaluate(placement)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field

from place_layers.evaluator.backend_evaluator import BackendEvaluator
from place_layers.evaluator.backend_evaluator_config import TaskBasedEvaluatorConfig
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.backend.base import (
    AsyncBackend,
    SyncBackend,
)
from place_layers.evaluator.traces.types import Generation
from place_layers.evaluator.traces.writer import summarize_generations
from place_layers.evaluator.backend.vllm import GenerationResult
from place_layers.evaluator.lm_eval_utils import check_correctness
from place_layers.evaluator.task_runner import (
    prepare_task,
    run_generation_async,
    run_generation_sync,
)
from place_layers.core.types import LayerPlacement

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


# =============================================================================
# Record Factory
# =============================================================================


def _make_generation_records(
    task: "Task", doc: dict, idx: int, result: GenerationResult
) -> list[Generation]:
    """Create Generation records from a generation result.

    Skips truncated samples (incomplete text makes correctness checking unreliable).
    Returns empty list if all samples are truncated.
    """
    task_name = task.config.task
    generations = []
    problem_id = f"{task_name}:{idx}"

    for sample_idx, sample in enumerate(result.samples):
        if sample.truncated:
            logger.debug(f"Skipping truncated sample {problem_id}:{sample_idx}")
            continue

        correct = check_correctness(task, doc, sample.text)
        gen = Generation(
            prompt_tokens=tuple(result.prompt_token_ids),
            completion_tokens=tuple(sample.token_ids),
            correct=correct,
            problem_id=problem_id,
            task_name=task_name,
            sample_idx=sample_idx,
            completion_text=sample.text,
        )
        generations.append(gen)

    return generations


# =============================================================================
# Evaluator Implementation
# =============================================================================


class GenerativeEvaluator(BackendEvaluator):
    """Evaluator that measures accuracy through direct generation.

    Supports both async (vLLM) and sync (Transformers) backends, dispatching
    to the appropriate execution path based on backend type.

    Key features:
    - Resumable: checks validation.jsonl for completed problems
    - Lean: only stores Generation records, metrics computed at end
    - Progress logging without in-memory accuracy tracking

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    def __init__(self, config: "GenerativeEvaluatorConfig") -> None:
        super().__init__(config)
        self.config: GenerativeEvaluatorConfig

        logger.info(f"Saving generations to: {self._output_dir}/validation.jsonl")

    # =========================================================================
    # Main Evaluation Entry Point
    # =========================================================================

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate a placement configuration by generating and checking accuracy.

        Args:
            parameters: Layer placement to evaluate, or None for default config.
                       When None, no placement switching is performed (useful for
                       non-supernet models or backends that don't support placement).

        Returns:
            Pass@k accuracy (fraction of problems with any correct sample).
        """
        if parameters is not None:
            await self._set_placement(parameters)
        else:
            logger.debug("No placement specified, using default configuration")

        # Dispatch to async or sync evaluation
        if isinstance(self._backend, AsyncBackend):
            await self._run_tasks_async()
        else:
            self._run_tasks_sync()

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
                record_factory=_make_generation_records,
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
                record_factory=_make_generation_records,
                samples_per_problem=self.config.samples_per_problem,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )


# =============================================================================
# Configuration
# =============================================================================


class GenerativeEvaluatorConfig(TaskBasedEvaluatorConfig):
    """Configuration for generative evaluator."""

    _impl_class: ClassVar[type[BaseEvaluator]] = GenerativeEvaluator

    evaluator_type: Literal["generative"] = "generative"

    # Generation parameters
    samples_per_problem: int = Field(
        description="Number of completions per problem (1 for greedy, >1 for pass@k)",
        gt=0,
    )
    max_new_tokens: int = Field(
        description="Maximum tokens to generate per completion", gt=0
    )
    temperature: float = Field(
        description="Sampling temperature (0.0 for greedy)", ge=0.0
    )
    top_p: float = Field(description="Nucleus sampling parameter", gt=0.0, le=1.0)
