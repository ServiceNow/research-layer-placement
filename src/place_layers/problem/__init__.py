"""Problem definitions for optimization."""

from typing import Annotated

from pydantic import Field

from place_layers.problem.layer_placement import (
    BaseCostConstrainedProblemConfig,
    CostConstrainedProblem,
    CostConstrainedProblemConfig,
    LayerPlacementProblem,
    LayerPlacementProblemConfig,
    MaxPerformanceProblemConfig,
    MinCostProblemConfig,
)


# Discriminated union of all problems
ProblemConfig = Annotated[
    LayerPlacementProblemConfig | MaxPerformanceProblemConfig | MinCostProblemConfig,
    Field(discriminator="problem_type"),
]
