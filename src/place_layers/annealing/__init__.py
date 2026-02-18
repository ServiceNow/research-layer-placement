"""Simulated annealing building blocks.

Provides move operators and core SA primitives used by multiple search
strategies (AnnealingSearch, SurrogateSearch).
"""

from place_layers.annealing.events import (
    AnnealingCompletedEvent,
    AnnealingReheatEvent,
    AnnealingRestartEvent,
    AnnealingStepEvent,
)
from place_layers.annealing.moves import (
    contiguous_ruin,
    generate_move,
    random_assignment,
    ruin_and_recreate,
    sample_perturbation_size,
    type_mutation,
)
from place_layers.annealing.sa_core import (
    MultiStartSAResult,
    ReheatConfig,
    SAConfig,
    SAObserver,
    TopKTracker,
    metropolis_accept,
    pick_restart_allocation,
    run_multi_start_sa,
)
