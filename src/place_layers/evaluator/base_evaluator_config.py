"""Base configuration for evaluators."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from place_layers.evaluator.base_evaluator import BaseEvaluator


class BaseEvaluatorConfig(BaseModel):
    """Base configuration for all evaluator implementations."""

    model_config = {"arbitrary_types_allowed": True}

    evaluator_type: str = Field(description="Type identifier for evaluator")

    # Each subclass must define this
    _impl_class: ClassVar[type[BaseEvaluator]]

    def create_evaluator(self) -> BaseEvaluator:
        """Smart constructor: instantiate evaluator from config."""
        return self._impl_class(config=self)
