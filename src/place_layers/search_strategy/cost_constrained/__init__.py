"""Cost-constrained search strategy implementations.

This module provides four variants of cost-constrained search:

1. SwapBeamSearch: Original swap-based beam search (thorough but expensive)
2. HierarchicalSearch: Phase-by-phase hierarchical beam search (~10x faster)
3. AnnealingSearch: Simulated annealing (controllable evaluation budget)
4. SurrogateSearch: Surrogate-assisted search (RF + Expected Improvement)

All variants support two optimization modes:
- MAX_PERFORMANCE: Maximize performance subject to cost <= budget
- MIN_COST: Minimize cost subject to performance >= target
"""

from typing import Annotated

from pydantic import Field

from place_layers.search_strategy.cost_constrained.allocation_iteration import (
    AllocationIterationSearch,
)
from place_layers.search_strategy.cost_constrained.annealing import (
    AnnealingConfig,
    AnnealingSearch,
)
from place_layers.search_strategy.cost_constrained.base_config import (
    AllocationIterationConfig,
)
from place_layers.search_strategy.cost_constrained.hierarchical import (
    HierarchicalConfig,
    HierarchicalSearch,
)
from place_layers.search_strategy.cost_constrained.surrogate import (
    SurrogateSearchConfig,
    SurrogateSearch,
)
from place_layers.search_strategy.cost_constrained.swap_beam import (
    SwapBeamConfig,
    SwapBeamSearch,
)
from place_layers.search_strategy.cost_constrained.viterbi_surrogate import (
    ViterbiSurrogateSearchConfig,
    ViterbiSurrogateSearch,
)
from place_layers.search_strategy.cost_constrained.types import (
    BeamEntry,
    CostConstrainedSearchResult,
)

# Discriminated union of all cost-constrained search strategy configs
CostConstrainedConfig = Annotated[
    SwapBeamConfig
    | HierarchicalConfig
    | AnnealingConfig
    | SurrogateSearchConfig
    | ViterbiSurrogateSearchConfig,
    Field(discriminator="strategy_type"),
]
