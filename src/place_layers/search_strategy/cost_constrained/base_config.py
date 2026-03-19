"""Base configuration for cost-constrained search strategies."""

from pydantic import Field

from place_layers.search_strategy.base_search_strategy_config import (
    BaseSearchStrategyConfig,
)


class BaseCostConstrainedConfig(BaseSearchStrategyConfig):
    """Base for cost-constrained search strategies.

    All strategies that operate on CostConstrainedProblem instances
    share the optional evaluation log path for JSONL recording and
    seed data paths for pre-loading surrogate training data.
    """

    evaluation_log_path: str | None = Field(
        description="Path to write evaluation JSONL log. None to skip.",
    )
    seed_data_paths: list[str] = Field(
        default_factory=list,
        description="Paths to seed data files (.json or .jsonl) "
        "loaded at startup to train the surrogate.",
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
