"""Cost model implementations for layer placement optimization."""

from typing import Annotated

from pydantic import Field

from place_layers.cost_model.base_cost_model import BaseCostModel
from place_layers.cost_model.base_cost_model_config import BaseCostModelConfig
from place_layers.cost_model.linear_cost_model import (
    LinearCostModel,
    LinearCostModelConfig,
)


# Discriminated union of all cost models
CostModelConfig = Annotated[
    LinearCostModelConfig,
    Field(discriminator="cost_model_type"),
]
