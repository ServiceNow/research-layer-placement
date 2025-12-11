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
    SwapBeamConfig,
    SwapBeamSearch,
)
from place_layers.search_strategy.cost_constrained_beam_search import (
    CostConstrainedBeamSearch,
    CostConstrainedBeamSearchConfig,
)
from place_layers.search_strategy.exhaustive_search import ExhaustiveSearchConfig
from place_layers.search_strategy.hierarchical_beam_search import (
    HierarchicalBeamSearchConfig,
)


# Discriminated union of all search strategies
SearchStrategyConfig = Annotated[
    HierarchicalBeamSearchConfig
    | ExhaustiveSearchConfig
    | CostConstrainedBeamSearchConfig
    | SwapBeamConfig
    | HierarchicalConfig
    | AnnealingConfig,
    Field(discriminator="strategy_type"),
]
