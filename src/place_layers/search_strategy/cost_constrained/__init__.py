"""Cost-constrained search strategy implementations.

This module provides several variants of cost-constrained search:

1. HierarchicalSearch: Phase-by-phase hierarchical beam search (~10x faster)
2. AnnealingSearch: Simulated annealing (controllable evaluation budget)
3. SurrogateSearch: Surrogate-assisted SA search (Ridge/Bayesian Ridge + acquisition)
4. ViterbiSurrogateSearch: Ridge Bootstrap Viterbi DP + EI selection
5. ViterbiBayesianSurrogateSearch: Bayesian Viterbi with Pareto LCB

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
    ViterbiBayesianSurrogateSearch,
    ViterbiBayesianSurrogateSearchConfig,
    ViterbiSurrogateSearch,
    ViterbiSurrogateSearchConfig,
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
    | ViterbiSurrogateSearchConfig
    | ViterbiBayesianSurrogateSearchConfig,
    Field(discriminator="strategy_type"),
]
