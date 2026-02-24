"""Base configuration for optimization problems."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from place_layers.evaluator.evaluators import Evaluators
    from place_layers.problem.base_problem import BaseProblem


class BaseProblemConfig(BaseModel):
    """Base configuration for all problem implementations."""

    model_config = {"arbitrary_types_allowed": True}

    problem_type: str = Field(description="Type identifier for problem")

    # Each subclass must define this
    _impl_class: ClassVar[type[BaseProblem]]

    def create_problem(self, evaluators: "Evaluators", objective_name: str) -> "BaseProblem":
        """Smart constructor: instantiate problem from config and evaluators."""
        return self._impl_class(evaluators=evaluators, config=self, objective_name=objective_name)
