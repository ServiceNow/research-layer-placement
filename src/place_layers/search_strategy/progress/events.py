"""Events emitted during cost-constrained search.

These events form the contract between search strategies and display observers.
Each event captures a specific point in the search process with all relevant data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from place_layers.search_strategy.progress.estimator import AllocationProgressInfo


class SearchStartedEvent(BaseModel):
    """Emitted when search begins."""

    model_config = {"frozen": True}

    mode: Literal["max_performance", "min_cost"]
    constraint_value: float  # Budget (max_perf) or target (min_cost)
    min_cost: float
    max_cost: float
    num_layers: int
    mixer_names: tuple[str, ...]
    max_allocations: int
    total_allocations: int
    eval_count_lower_bound: int | None = None
    eval_count_upper_bound: int | None = None
    eval_count_expected: int | None = None


class AllocationStartedEvent(BaseModel):
    """Emitted when starting search within a new allocation.

    Includes per-allocation evaluation bounds for progress tracking.
    """

    model_config = {"frozen": True}

    allocation_index: int
    total_allocations: int  # max_allocations limit
    allocation: tuple[int, ...]
    allocation_cost: float
    mixer_names: tuple[str, ...]
    # Per-allocation evaluation bounds (from progress estimator)
    eval_lower_bound: int | None = None
    eval_upper_bound: int | None = None
    eval_expected: int | None = None


class PhaseStartedEvent(BaseModel):
    """Emitted when starting a new phase within hierarchical search."""

    model_config = {"frozen": True}

    phase_index: int
    total_phases: int
    mixer_name: str
    budget: int  # Number of layers to assign in this phase


class BeamGrowthStepEvent(BaseModel):
    """Emitted after each beam growth step.

    Contains detailed information about what was tried and the resulting beam state.
    This enables visualization of the search evolution at each step.
    """

    model_config = {"frozen": True}

    step: int
    total_steps: int
    phase_index: int
    mixer_name: str
    base_layers: tuple[int, ...]  # Layers already fixed for this mixer before this step
    tried: tuple[dict, ...]  # All candidates tried: [{"layer": int, "score": float}, ...]
    beam: tuple[dict, ...]  # Top-k after selection: [{"layers": [...], "score": float}, ...]


class BestUpdatedEvent(BaseModel):
    """Emitted when a new global best is found."""

    model_config = {"frozen": True}

    score: float
    cost: float
    allocation: tuple[int, ...]
    assignments: tuple[str, ...]


class PhaseCompletedEvent(BaseModel):
    """Emitted when a phase within hierarchical search completes."""

    model_config = {"frozen": True}

    phase_index: int
    total_phases: int
    mixer_name: str
    budget: int  # Number of layers assigned in this phase
    final_layers: tuple[int, ...]  # Layer indices assigned to this mixer
    final_score: float


class BoundsUpdatedEvent(BaseModel):
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

    model_config = {"frozen": True}

    previous_lower: int
    previous_upper: int
    new_lower: int
    new_upper: int
    completed: int
    remaining_allocations: int
    reason: str


class AllocationPlanEvent(BaseModel):
    """Emitted after search starts to show planned allocations.

    Contains the list of allocations that will be explored, with estimated
    evaluation counts for each. Useful for dry_run preview and live search.
    """

    model_config = {"frozen": True}

    allocation_info: tuple["AllocationProgressInfo", ...]
    mixer_names: tuple[str, ...]


class SearchCompletedEvent(BaseModel):
    """Emitted when search finishes."""

    model_config = {"frozen": True}

    best_score: float
    best_cost: float
    total_evaluations: int
    allocations_explored: int
