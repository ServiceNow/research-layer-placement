"""Base configuration for cost-constrained search strategies."""

from pydantic import Field

from place_layers.search_strategy.base_search_strategy_config import (
    BaseSearchStrategyConfig,
)


class BaseCostConstrainedConfig(BaseSearchStrategyConfig):
    """Marker base for cost-constrained search strategies.

    This is the shared base for all strategies that operate on
    CostConstrainedProblem instances. It carries no extra fields.
    """


class AllocationIterationConfig(BaseCostConstrainedConfig):
    """Configuration for strategies that iterate through allocations in cost order.

    These strategies enumerate allocations, call _search_within_allocation()
    for each, and track the global best.
    """

    max_allocations: int = Field(
        description="Maximum number of allocations to explore",
        gt=0,
    )
