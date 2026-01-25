"""Base class for evaluators that iterate over lm-eval tasks.

This module provides the TaskBasedEvaluator which adds task iteration
infrastructure on top of BackendEvaluator:

- Task loading via lm-eval
- Task type validation (generate_until vs multiple_choice)
- Document retrieval and max_problems limiting
- Resumability via TraceWriter.get_completed_problems()

Concrete evaluators (GenerativeEvaluator, LikelihoodEvaluator, etc.)
inherit from this and implement their specific evaluation logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Self, cast

from pydantic import Field, model_validator

from place_layers.evaluator.backend_evaluator import (
    BackendEvaluator,
    BackendEvaluatorConfig,
)
from place_layers.evaluator.lm_eval_utils import (
    Docs,
    TaskSpecInput,
    get_docs,
    get_task_name,
    load_tasks,
)

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


class TaskBasedEvaluator(BackendEvaluator):
    """Base class for evaluators that iterate over lm-eval tasks.

    Extends BackendEvaluator with task iteration infrastructure:
    - _prepare_task(): Get task from cache, validate type, check resumability
    - Uses self._trace_writer.get_completed_problems() for resumability
    - Caches tasks and docs for efficiency across multiple placements

    Subclasses must implement:
    - evaluate(): The actual evaluation logic
    - _get_fewshot_seed(): Return seed for few-shot sampling (or None)

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "TaskBasedEvaluatorConfig"

    def __init__(self, config: "TaskBasedEvaluatorConfig") -> None:
        """Initialize task-based evaluator.

        Args:
            config: Evaluator configuration with task settings.
        """
        super().__init__(config)
        # Cache for batch-loaded tasks (populated in __aenter__)
        self._task_cache: dict[str, "Task"] | None = None
        # Cache for docs - persists across placements (lazy populated in _prepare_task)
        self._docs_cache: dict[str, Docs] = {}

    def _get_fewshot_seed(self) -> int | None:
        """Return seed for few-shot sampling, or None for default behavior.

        Subclasses can override this to provide a specific seed.
        """
        return None

    async def __aenter__(self) -> Self:
        """Enter context: initialize backend and load tasks.

        Tasks are loaded here (not in __init__) because:
        1. Backend must be initialized first
        2. Seed may depend on evaluator-specific logic (_get_fewshot_seed)
        3. Loading all tasks together is ~6x faster than individual loading
        """
        await super().__aenter__()

        # Batch load all tasks for efficiency
        seed = self._get_fewshot_seed()
        logger.info(f"Batch-loading {len(self.config.tasks)} tasks...")
        self._task_cache = load_tasks(
            self.config.tasks, self.config.num_fewshot, seed
        )
        logger.info(f"Loaded {len(self._task_cache)} tasks")

        return self

    def _prepare_task(
        self,
        task_spec: TaskSpecInput,
        expected_output_type: Literal["generate_until", "multiple_choice"],
    ) -> tuple["Task", str, Docs, list[int]] | None:
        """Get task from cache and prepare for evaluation.

        This handles:
        - Retrieving the pre-loaded task from cache
        - Validating task type (generate_until or multiple_choice)
        - Getting documents (cached across placements)
        - Applying max_problems limit
        - Checking completed problems for resumability
        - Computing remaining indices

        Args:
            task_spec: Task name or TaskSpec with overrides.
            expected_output_type: Expected task type ('generate_until' or 'multiple_choice').

        Returns:
            None if task is already complete.
            (task, task_name, docs, remaining_indices) otherwise.
        """
        task_name = get_task_name(task_spec)
        logger.info(f"Task '{task_name}': preparing...")

        # Tasks must be pre-loaded in __aenter__
        if self._task_cache is None:
            raise RuntimeError(
                "Tasks not loaded. TaskBasedEvaluator must be used as async context manager."
            )

        task = self._task_cache[task_name]

        # Validate task type
        if task.config.output_type != expected_output_type:
            raise ValueError(
                f"Task '{task_name}' has output_type='{task.config.output_type}', "
                f"expected '{expected_output_type}'"
            )

        # Get documents from cache (or load and cache on first access)
        if task_name not in self._docs_cache:
            task.download()
            self._docs_cache[task_name] = get_docs(task, self.config.split)

        docs = self._docs_cache[task_name]

        # Apply max_problems limit
        total_problems = len(docs)
        if self.config.max_problems_per_task is not None:
            total_problems = min(total_problems, self.config.max_problems_per_task)
            # Convert to list for slicing (docs might be HF Dataset)
            docs = cast(Docs, [docs[i] for i in range(total_problems)])

        # Check completed problems for resumability
        completed = self._trace_writer.get_completed_problems(task_name)
        remaining = [i for i in range(total_problems) if i not in completed]

        if not remaining:
            logger.info(f"Task '{task_name}': already complete, skipping")
            return None

        logger.info(
            f"Task '{task_name}': {len(remaining)} remaining "
            f"({len(completed)} already done, {total_problems} total)"
        )

        return task, task_name, docs, remaining


class TaskBasedEvaluatorConfig(BackendEvaluatorConfig):
    """Config for evaluators that process lm-eval benchmark tasks."""

    tasks: list[TaskSpecInput] = Field(
        description="List of lm-eval tasks to evaluate (e.g., ['gsm8k', 'math'])"
    )
    num_fewshot: int = Field(
        description="Number of few-shot examples",
        ge=0,
    )
    max_problems_per_task: int | None = Field(
        default=None,
        description="Max problems per task (None = all). Use for faster evaluation.",
    )
    split: str = Field(
        default="test",
        description="Dataset split to use ('test' or 'validation')",
    )

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        """Validate that at least one task is specified."""
        if not self.tasks:
            raise ValueError("At least one task must be specified")
        return self
