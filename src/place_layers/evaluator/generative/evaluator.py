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
        num_layers=24,
        mixer_names=["attention", "mamba", "gdn"],
        tasks=["gsm8k"],
        ...
    )
    evaluator = config.create_evaluator()
    score = await evaluator.evaluate(placement)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from hydra.core.hydra_config import HydraConfig
from pydantic import Field, model_validator

from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.base_evaluator_config import BaseEvaluatorConfig
from place_layers.evaluator.importance_weighted.generation_backend import (
    AsyncBackend,
    BackendConfig,
    SyncBackend,
)
from place_layers.evaluator.importance_weighted.traces import Generation
from place_layers.evaluator.importance_weighted.trace_writer import (
    TraceWriter,
    summarize_generations,
)
from place_layers.evaluator.importance_weighted.vllm_client import GenerationResult
from place_layers.evaluator.lm_eval_utils import (
    TaskSpecInput,
    check_correctness,
    get_task_name,
)
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


class GenerativeEvaluator(BaseEvaluator[LayerPlacement]):
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

        # Create backend (but don't enter context yet)
        self._backend: AsyncBackend | SyncBackend = config.backend.create_backend()

        # Track current placement to avoid redundant set_placement calls
        self._current_placement: tuple[str, ...] | None = None

        # Set up trace writer for saving generations
        hydra_cfg = HydraConfig.get()
        self._output_dir = Path(hydra_cfg.runtime.output_dir)
        self._writer = TraceWriter(self._output_dir)

        logger.info(
            f"GenerativeEvaluator initialized: {config.num_layers} layers, "
            f"{len(config.mixer_names)} mixers, backend={config.backend.backend_type}"
        )
        logger.info(f"Saving generations to: {self._output_dir}/validation.jsonl")

    async def __aenter__(self) -> Self:
        """Enter context: initialize backend resources."""
        if isinstance(self._backend, AsyncBackend):
            await self._backend.__aenter__()
        else:
            self._backend.__enter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: cleanup backend resources."""
        if isinstance(self._backend, AsyncBackend):
            await self._backend.__aexit__(exc_type, exc_val, exc_tb)
        else:
            self._backend.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def num_mixers(self) -> int:
        return len(self.config.mixer_names)

    # =========================================================================
    # Main Evaluation Entry Point
    # =========================================================================

    async def evaluate(self, parameters: LayerPlacement) -> float:
        """Evaluate a placement configuration by generating and checking accuracy.

        Returns pass@k accuracy (fraction of problems with any correct sample).
        """
        # Validate placement
        if len(parameters.assignments) != self.num_layers:
            raise ValueError(
                f"Placement has {len(parameters.assignments)} assignments, "
                f"expected {self.num_layers}"
            )

        # Convert assignment indices to mixer names
        mixer_names = tuple(
            self.config.mixer_names[idx] for idx in parameters.assignments
        )

        # Set placement via backend (skip if unchanged or single-mixer model)
        if mixer_names != self._current_placement:
            # Skip placement for single-mixer models (no dynamic switching needed)
            if len(set(self.config.mixer_names)) > 1:
                if isinstance(self._backend, AsyncBackend):
                    await self._backend.set_placement(mixer_names)
                else:
                    self._backend.set_placement(mixer_names)
                logger.debug(f"Set placement: {mixer_names[:4]}...")
            self._current_placement = mixer_names

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
                writer=self._writer,
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
                writer=self._writer,
                record_factory=_make_generation_records,
                samples_per_problem=self.config.samples_per_problem,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )


# =============================================================================
# Configuration
# =============================================================================


class GenerativeEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for generative evaluator."""

    _impl_class: ClassVar[type[BaseEvaluator]] = GenerativeEvaluator

    evaluator_type: Literal["generative"] = "generative"

    # Backend (polymorphic - vLLM or Transformers)
    backend: BackendConfig = Field(
        description="Generation backend configuration (vllm or transformers)"
    )

    # Model structure (required for BaseEvaluator interface)
    num_layers: int = Field(description="Number of layers in the model", gt=0)
    mixer_names: list[str] = Field(
        description="Available mixer types (e.g., ['attention', 'mamba', 'gdn'])"
    )

    # Tasks
    tasks: list[TaskSpecInput] = Field(
        description="List of lm-eval tasks to evaluate (e.g., ['gsm8k', 'math'])"
    )
    num_fewshot: int = Field(description="Default number of few-shot examples", ge=0)
    max_problems_per_task: int | None = Field(
        default=None,
        description="Max problems per task (None = all). Use for faster evaluation.",
    )
    split: str = Field(
        default="test",
        description="Dataset split to use ('test' or 'validation')",
    )

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

    @model_validator(mode="after")
    def validate_config(self) -> "GenerativeEvaluatorConfig":
        """Validate configuration consistency."""
        if not self.tasks:
            raise ValueError("At least one task must be specified")

        if not self.mixer_names:
            raise ValueError("At least one mixer name must be specified")

        return self
