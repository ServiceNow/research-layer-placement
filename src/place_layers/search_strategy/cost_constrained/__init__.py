"""Cost-constrained search strategy implementations.

This module provides three variants of cost-constrained search:

1. HierarchicalSearch: Phase-by-phase hierarchical beam search (~10x faster)
2. AnnealingSearch: Simulated annealing (controllable evaluation budget)
3. SurrogateSearch: Surrogate-assisted search (RF + Expected Improvement)

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
    HierarchicalConfig
    | AnnealingConfig
    | SurrogateSearchConfig
    | ViterbiSurrogateSearchConfig,
    Field(discriminator="strategy_type"),
]
