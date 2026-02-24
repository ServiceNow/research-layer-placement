"""Base configuration for cost-constrained search strategies."""

from pydantic import Field

from place_layers.search_strategy.base_search_strategy_config import (
    BaseSearchStrategyConfig,
)


class BaseCostConstrainedConfig(BaseSearchStrategyConfig):
    """Base for cost-constrained search strategies.

    All strategies that operate on CostConstrainedProblem instances
    share the optional evaluation log path for JSONL recording.
    """

    evaluation_log_path: str | None = Field(
        description="Path to write evaluation JSONL log. None to skip.",
    )


class AllocationIterationConfig(BaseCostConstrainedConfig):
    """Configuration for strategies that iterate through allocations in cost order.

    These strategies enumerate allocations, call _search_within_allocation()
    for each, and track the global best.
    """

    max_allocations: int = Field(
        description="Maximum number of allocations to explore",
        gt=0,
    )
