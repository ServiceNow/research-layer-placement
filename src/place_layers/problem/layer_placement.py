"""Layer placement optimization problem."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.linear_cost_model import LinearCostModelConfig
from place_layers.problem.base_problem import BaseProblem
from place_layers.problem.base_problem_config import BaseProblemConfig


if TYPE_CHECKING:
    from place_layers.evaluator.base_evaluator import BaseEvaluator


class LayerPlacementProblem(BaseProblem[LayerPlacement]):
    """Layer placement optimization problem with count-based budgets."""

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


class CostConstrainedProblem(BaseProblem[LayerPlacement]):
    """
    Layer placement optimization problem with cost-based constraints.

    Supports two optimization modes via discriminated union:
    1. cost_max_performance: Maximize performance subject to cost <= cost_budget
    2. cost_min_cost: Minimize cost subject to performance >= performance_target
    """

    def __init__(
        self,
        evaluator: BaseEvaluator[LayerPlacement],
        config: "CostConstrainedProblemConfig",
    ) -> None:
        super().__init__(evaluator, config)
        self.config: CostConstrainedProblemConfig  # Type narrowing

        # Create cost model
        self.cost_model = config.cost_model.create_cost_model()

        # Validate dimensions
        num_mixers = evaluator.config.num_mixers
        if len(config.cost_model.mixer_costs) != num_mixers:
            raise ValueError(
                f"Number of mixer costs ({len(config.cost_model.mixer_costs)}) "
                f"must match number of mixers ({num_mixers})"
            )

        # Validate constraint based on problem type
        num_layers = evaluator.config.num_layers
        min_cost = self.cost_model.min_cost(num_layers)
        max_cost = self.cost_model.max_cost(num_layers)

        if isinstance(config, MaxPerformanceProblemConfig):
            if config.cost_budget < min_cost:
                raise ValueError(
                    f"cost_budget ({config.cost_budget}) is less than minimum "
                    f"possible cost ({min_cost})"
                )
            if config.cost_budget > max_cost:
                raise ValueError(
                    f"cost_budget ({config.cost_budget}) exceeds maximum "
                    f"possible cost ({max_cost}). This constraint is trivially satisfied."
                )

    def compute_cost(self, placement: LayerPlacement) -> float:
        """Compute the cost of a placement configuration."""
        return self.cost_model.cost(placement)


class BaseCostConstrainedProblemConfig(BaseProblemConfig):
    """
    Base configuration for cost-constrained layer placement optimization.

    This problem type uses a cost function instead of hard count-based budgets.
    The cost function maps layer-to-mixer assignments to a scalar cost value,
    typically representing inference latency/throughput.

    Two optimization modes are available as separate config types:

    1. MaxPerformanceProblemConfig (problem_type="cost_max_performance"):
       - Objective: Maximize performance (evaluation score)
       - Constraint: cost <= cost_budget
       - Use case: "Get best accuracy within latency budget"

    2. MinCostProblemConfig (problem_type="cost_min_cost"):
       - Objective: Minimize cost
       - Constraint: performance >= performance_target
       - Use case: "Get fastest model that meets accuracy threshold"
    """

    _impl_class: ClassVar[type[BaseProblem]] = CostConstrainedProblem

    cost_model: LinearCostModelConfig = Field(
        description="Cost model configuration defining per-mixer costs"
    )


class MaxPerformanceProblemConfig(BaseCostConstrainedProblemConfig):
    """
    Cost-constrained problem: Maximize performance subject to cost <= budget.

    Use case: "Get best accuracy within latency budget"
    """

    problem_type: Literal["cost_max_performance"] = "cost_max_performance"

    cost_budget: float = Field(
        description="Maximum allowed cost. Must be between min and max possible costs.",
        gt=0,
    )


class MinCostProblemConfig(BaseCostConstrainedProblemConfig):
    """
    Cost-constrained problem: Minimize cost subject to performance >= target.

    Use case: "Get fastest model that meets accuracy threshold"
    """

    problem_type: Literal["cost_min_cost"] = "cost_min_cost"

    performance_target: float = Field(
        description="Minimum required performance. "
        "Search will find cheapest configuration meeting this threshold.",
    )


# Discriminated union of cost-constrained problem configs
CostConstrainedProblemConfig = Annotated[
    MaxPerformanceProblemConfig | MinCostProblemConfig,
    Field(discriminator="problem_type"),
]
