"""Events emitted during cost-constrained search.

These events form the contract between search strategies and display observers.
Each event captures a specific point in the search process with all relevant data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from place_layers.search_strategy.progress.estimator import AllocationProgressInfo


@dataclass(frozen=True)
class SearchStartedEvent:
    """Emitted when search begins."""

    mode: Literal["max_performance", "min_cost"]
    constraint_value: float  # Budget (max_perf) or target (min_cost)
    min_cost: float
    max_cost: float
    num_layers: int
    mixer_names: list[str]
    max_allocations: int
    total_allocations: int
    eval_count_lower_bound: int | None = None
    eval_count_upper_bound: int | None = None
    eval_count_expected: int | None = None


@dataclass(frozen=True)
class AllocationStartedEvent:
    """Emitted when starting search within a new allocation.

    Includes per-allocation evaluation bounds for progress tracking.
    """

    allocation_index: int
    total_allocations: int  # max_allocations limit
    allocation: tuple[int, ...]
    allocation_cost: float
    mixer_names: list[str]
    # Per-allocation evaluation bounds (from progress estimator)
    eval_lower_bound: int | None = None
    eval_upper_bound: int | None = None
    eval_expected: int | None = None


@dataclass(frozen=True)
class PhaseStartedEvent:
    """Emitted when starting a new phase within hierarchical search."""

    phase_index: int
    total_phases: int
    mixer_name: str
    budget: int  # Number of layers to assign in this phase


@dataclass(frozen=True)
class BeamGrowthStepEvent:
    """Emitted after each beam growth step.

    Contains detailed information about what was tried and the resulting beam state.
    This enables visualization of the search evolution at each step.
    """

    step: int
    total_steps: int
    phase_index: int
    mixer_name: str
    base_layers: list[int]  # Layers already fixed for this mixer before this step
    tried: list[dict]  # All candidates tried: [{"layer": int, "score": float}, ...]
    beam: list[dict]  # Top-k after selection: [{"layers": [...], "score": float}, ...]


@dataclass(frozen=True)
class BestUpdatedEvent:
    """Emitted when a new global best is found."""

    score: float
    cost: float
    allocation: tuple[int, ...]
    assignments: list[str]


@dataclass(frozen=True)
class PhaseCompletedEvent:
    """Emitted when a phase within hierarchical search completes."""

    phase_index: int
    total_phases: int
    mixer_name: str
    budget: int  # Number of layers assigned in this phase
    final_layers: list[int]  # Layer indices assigned to this mixer
    final_score: float


@dataclass(frozen=True)
class BoundsUpdatedEvent:
    """Emitted when evaluation bounds are refined during search.

    This event is emitted after each allocation completes, allowing
    observers to track how bounds narrow as actual evaluations
    replace estimates.

    Attributes:
        previous_lower: Lower bound before update.
        previous_upper: Upper bound before update.
        new_lower: Lower bound after update.
        new_upper: Upper bound after update.
        completed: Total evaluations done so far.
        remaining_allocations: Number of allocations left to explore.
        reason: Why bounds were updated (e.g., "allocation_complete").
    """

    previous_lower: int
    previous_upper: int
    new_lower: int
    new_upper: int
    completed: int
    remaining_allocations: int
    reason: str


@dataclass(frozen=True)
class AllocationPlanEvent:
    """Emitted after search starts to show planned allocations.

    Contains the list of allocations that will be explored, with estimated
    evaluation counts for each. Useful for dry_run preview and live search.
    """

    allocation_info: tuple["AllocationProgressInfo", ...]
    mixer_names: list[str]


@dataclass(frozen=True)
class SearchCompletedEvent:
    """Emitted when search finishes."""

    best_score: float
    best_cost: float
    total_evaluations: int
    allocations_explored: int
