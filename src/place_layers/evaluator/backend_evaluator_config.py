"""Configuration base classes for backend-based evaluators.

This module provides intermediate configuration base classes:
- BackendEvaluatorConfig: Adds backend field for all backend evaluators
- TaskBasedEvaluatorConfig: Adds task iteration fields for lm-eval tasks

These are NOT concrete configs - they don't have evaluator_type or _impl_class.
Concrete evaluator configs inherit from these and add the discriminator.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from place_layers.evaluator.base_evaluator_config import BaseEvaluatorConfig
from place_layers.evaluator.backend.base import BackendConfig
from place_layers.evaluator.lm_eval_utils import TaskSpecInput


class BackendEvaluatorConfig(BaseEvaluatorConfig):
    """Base configuration for evaluators using generation backends.

    This is an intermediate base class - concrete evaluator configs
    should inherit from this and define evaluator_type and _impl_class.

    Provides:
    - backend: Polymorphic backend configuration (vLLM, Transformers, REST)
    """

    backend: BackendConfig = Field(
        description="Generation backend configuration (vllm, transformers, or rest)"
    )


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
