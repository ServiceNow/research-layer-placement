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
from typing import TYPE_CHECKING, Literal, Self

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
    load_task,
)

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


class TaskBasedEvaluator(BackendEvaluator):
    """Base class for evaluators that iterate over lm-eval tasks.

    Extends BackendEvaluator with task iteration infrastructure:
    - _prepare_task(): Load task, validate type, check resumability
    - Uses self._trace_writer.get_completed_problems() for resumability

    Subclasses must implement:
    - evaluate(): The actual evaluation logic

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    def __init__(self, config: "TaskBasedEvaluatorConfig") -> None:
        """Initialize task-based evaluator.

        Args:
            config: Evaluator configuration with task settings.
        """
        super().__init__(config)
        self.config: "TaskBasedEvaluatorConfig"

    def _prepare_task(
        self,
        task_spec: TaskSpecInput,
        expected_output_type: Literal["generate_until", "multiple_choice"],
        seed: int | None = None,
    ) -> tuple["Task", str, Docs, list[int]] | None:
        """Load task and get remaining problem indices.

        This handles:
        - Loading the lm-eval task
        - Validating task type (generate_until or multiple_choice)
        - Getting documents for the configured split
        - Applying max_problems limit
        - Checking completed problems for resumability
        - Computing remaining indices

        Args:
            task_spec: Task name or TaskSpec with overrides.
            expected_output_type: Expected task type ('generate_until' or 'multiple_choice').
            seed: Random seed for few-shot sampling (optional).

        Returns:
            None if task is already complete.
            (task, task_name, docs, remaining_indices) otherwise.
        """
        task_name = get_task_name(task_spec)
        logger.info(f"Task '{task_name}': loading...")

        # Load task
        task = load_task(task_spec, self.config.num_fewshot, seed)

        # Validate task type
        if task.config.output_type != expected_output_type:
            raise ValueError(
                f"Task '{task_name}' has output_type='{task.config.output_type}', "
                f"expected '{expected_output_type}'"
            )

        # Download dataset if needed
        task.download()

        # Get documents
        docs = get_docs(task, self.config.split)

        # Apply max_problems limit
        total_problems = len(docs)
        if self.config.max_problems_per_task is not None:
            total_problems = min(total_problems, self.config.max_problems_per_task)
            # Convert to list for slicing (docs might be HF Dataset)
            docs = [docs[i] for i in range(total_problems)]

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


# =============================================================================
# Configuration
# =============================================================================


class TaskBasedEvaluatorConfig(BackendEvaluatorConfig):
    """Configuration for evaluators that iterate over lm-eval tasks.

    This is an intermediate base class for evaluators that process
    lm-eval benchmark tasks (generative, likelihood). Concrete configs
    should inherit from this and define evaluator_type and _impl_class.

    Provides:
    - tasks: List of lm-eval tasks to evaluate
    - num_fewshot: Number of few-shot examples
    - max_problems_per_task: Optional limit on problems per task
    - split: Dataset split to use

    Note: ImportanceWeightedEvaluatorConfig inherits from BackendEvaluatorConfig
    directly since it uses trace datasets rather than lm-eval tasks.
    """

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
