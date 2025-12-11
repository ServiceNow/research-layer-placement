"""Cost-constrained search strategy implementations.

This module provides three variants of cost-constrained search that share
the same outer loop (allocation navigation) but differ in their inner loop
(assignment search within each allocation):

1. SwapBeamSearch: Original swap-based beam search (thorough but expensive)
2. HierarchicalSearch: Phase-by-phase hierarchical beam search (~10x faster)
3. AnnealingSearch: Simulated annealing (controllable evaluation budget)

All variants support two optimization modes:
- MAX_PERFORMANCE: Maximize performance subject to cost <= budget
- MIN_COST: Minimize cost subject to performance >= target
"""

from typing import Annotated

from pydantic import Field

from place_layers.search_strategy.cost_constrained.annealing import (
    AnnealingConfig,
    AnnealingSearch,
)
from place_layers.search_strategy.cost_constrained.hierarchical import (
    HierarchicalConfig,
    HierarchicalSearch,
)
from place_layers.search_strategy.cost_constrained.swap_beam import (
    SwapBeamConfig,
    SwapBeamSearch,
)
from place_layers.search_strategy.cost_constrained.types import (
    BeamEntry,
    CostConstrainedSearchResult,
)

# Discriminated union of all cost-constrained search strategy configs
CostConstrainedConfig = Annotated[
    SwapBeamConfig | HierarchicalConfig | AnnealingConfig,
    Field(discriminator="strategy_type"),
]
