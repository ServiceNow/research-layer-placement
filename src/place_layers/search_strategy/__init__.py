"""Search strategy implementations."""

from typing import Annotated

from pydantic import Field

from place_layers.search_strategy.cost_constrained import (
    AnnealingConfig,
    AnnealingSearch,
    BeamEntry,
    CostConstrainedConfig,
    CostConstrainedSearchResult,
    HierarchicalConfig,
    HierarchicalSearch,
    SurrogateSearchConfig,
    ViterbiBayesianSurrogateSearchConfig,
    ViterbiSurrogateSearchConfig,
)


# Discriminated union of all search strategies
SearchStrategyConfig = Annotated[
    HierarchicalConfig
    | AnnealingConfig
    | SurrogateSearchConfig
    | ViterbiSurrogateSearchConfig
    | ViterbiBayesianSurrogateSearchConfig,
    Field(discriminator="strategy_type"),
]
