"""Problem definitions for optimization."""

from typing import Annotated

from pydantic import Field

from place_layers.problem.layer_placement import LayerPlacementProblemConfig


# Discriminated union of all problems
ProblemConfig = Annotated[
    LayerPlacementProblemConfig,
    Field(discriminator="problem_type"),
]
