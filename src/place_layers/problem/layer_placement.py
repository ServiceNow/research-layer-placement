"""Layer placement optimization problem."""

from __future__ import annotations

from typing import ClassVar, Literal, Self

from pydantic import BaseModel, Field, model_validator

from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.problem.base_problem import BaseProblem
from place_layers.problem.base_problem_config import BaseProblemConfig


class LayerPlacement(BaseModel):
    """
    Immutable assignment of layers to mixer types.

    This is the primary parameter type for layer placement optimization.
    Each layer is assigned to exactly one mixer type.
    """

    model_config = {"frozen": True}

    assignments: tuple[int, ...] = Field(
        description="Mapping from layer index to mixer index"
    )


class LayerPlacementProblem(BaseProblem[LayerPlacement]):
    """Layer placement optimization problem."""

    def __init__(
        self,
        evaluator: BaseEvaluator[LayerPlacement],
        config: LayerPlacementProblemConfig,
    ) -> None:
        super().__init__(evaluator, config)
        self.config: LayerPlacementProblemConfig  # Type narrowing

        # Validate budgets against evaluator dimensions
        num_layers = evaluator.config.num_layers
        num_mixers = evaluator.config.num_mixers

        if len(config.budgets) >= num_mixers:
            raise ValueError(
                f"Length of budgets ({len(config.budgets)}) must be less than "
                f"number of mixers ({num_mixers})"
            )

        if sum(config.budgets) > num_layers:
            raise ValueError(
                f"Sum of budgets ({sum(config.budgets)}) cannot exceed "
                f"number of layers ({num_layers})"
            )


class LayerPlacementProblemConfig(BaseProblemConfig):
    """Configuration for layer placement optimization problem.

    Note: num_layers and num_mixers are determined by the evaluator.
    This config only specifies the budget constraints.
    """

    _impl_class: ClassVar[type[BaseProblem]] = LayerPlacementProblem

    problem_type: Literal["layer_placement"] = "layer_placement"

    budgets: list[int] = Field(
        description="Budget for each mixer level. "
        "budgets[i] specifies how many layers use mixers[i]. "
        "Length must be less than number of mixers (last mixer used for remaining). "
        "Validated against evaluator dimensions at runtime.",
    )

    @model_validator(mode="after")
    def validate_budgets(self) -> Self:
        """Validate budget constraints (runtime checks against evaluator in __init__)."""
        if any(b <= 0 for b in self.budgets):
            raise ValueError("All budgets must be positive")

        return self
