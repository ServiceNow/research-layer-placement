"""Base configuration for cost-constrained search strategies."""

from pydantic import Field

from place_layers.search_strategy.base_search_strategy_config import (
    BaseSearchStrategyConfig,
)


class BaseCostConstrainedConfig(BaseSearchStrategyConfig):
    """
    Base configuration shared by all cost-constrained search variants.

    This defines the outer loop parameters that control allocation navigation.
    Subclasses add their own inner loop parameters.
    """

    max_allocations: int = Field(
        description="Maximum number of allocations to explore",
        gt=0,
    )
