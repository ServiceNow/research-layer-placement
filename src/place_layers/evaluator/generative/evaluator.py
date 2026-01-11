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

        # Model structure (populated in __aenter__ from backend)
        self._num_layers: int | None = None
        self._mixer_names: tuple[str, ...] | None = None

        # Track current placement to avoid redundant set_placement calls
        self._current_placement: tuple[str, ...] | None = None

        # Set up trace writer for saving generations
        hydra_cfg = HydraConfig.get()
        self._output_dir = Path(hydra_cfg.runtime.output_dir)
        self._trace_writer = TraceWriter(self._output_dir)

        logger.info(f"Saving generations to: {self._output_dir}/validation.jsonl")

    async def __aenter__(self) -> Self:
        """Enter context: initialize backend and optionally get model structure."""
        if isinstance(self._backend, AsyncBackend):
            await self._backend.__aenter__()
        else:
            self._backend.__enter__()

        # Try to get model structure from backend (optional - not all backends support it)
        try:
            structure = self._backend.model_structure
            self._num_layers = structure.num_layers
            self._mixer_names = structure.mixer_names
            logger.info(
                f"GenerativeEvaluator initialized: {self._num_layers} layers, "
                f"{len(self._mixer_names)} mixers, backend={self.config.backend.backend_type}"
            )
        except NotImplementedError:
            # Model structure not available (e.g., vLLM backend)
            # This is fine if we're not doing placement switching
            logger.info(
                f"GenerativeEvaluator initialized: model structure not available "
                f"(backend={self.config.backend.backend_type}). "
                f"Placement switching will not be supported."
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: cleanup backend resources."""
        if isinstance(self._backend, AsyncBackend):
            await self._backend.__aexit__(exc_type, exc_val, exc_tb)
        else:
            self._backend.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_layers(self) -> int:
        if self._num_layers is None:
            raise RuntimeError("num_layers not available - evaluator must be entered first")
        return self._num_layers

    @property
    def num_mixers(self) -> int:
        if self._mixer_names is None:
            raise RuntimeError("num_mixers not available - evaluator must be entered first")
        return len(self._mixer_names)

    @property
    def mixer_names(self) -> tuple[str, ...]:
        """Available mixer types (e.g., ('attention', 'gdn', 'kda'))."""
        if self._mixer_names is None:
            raise RuntimeError("mixer_names not available - evaluator must be entered first")
        return self._mixer_names

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
            # Validate placement
            if len(parameters.assignments) != self.num_layers:
                raise ValueError(
                    f"Placement has {len(parameters.assignments)} assignments, "
                    f"expected {self.num_layers}"
                )

            # Convert assignment indices to mixer names
            mixer_names = tuple(
                self.mixer_names[idx] for idx in parameters.assignments
            )

            # Set placement via backend (skip if unchanged or single-mixer model)
            if mixer_names != self._current_placement:
                # Skip placement for single-mixer models (no dynamic switching needed)
                if len(set(self.mixer_names)) > 1:
                    if isinstance(self._backend, AsyncBackend):
                        await self._backend.set_placement(mixer_names)
                    else:
                        self._backend.set_placement(mixer_names)
                    logger.debug(f"Set placement: {mixer_names[:4]}...")
                self._current_placement = mixer_names
        else:
            # No placement specified - run with default model configuration
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


class GenerativeEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for generative evaluator."""

    _impl_class: ClassVar[type[BaseEvaluator]] = GenerativeEvaluator

    evaluator_type: Literal["generative"] = "generative"

    # Backend (polymorphic - vLLM, Transformers, or REST)
    # The backend provides model structure (num_layers, mixer_names) after initialization
    backend: BackendConfig = Field(
        description="Generation backend configuration (vllm, transformers, or rest)"
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
        return self
