"""Events emitted during cost-constrained search.

These events form the contract between search strategies and display observers.
Each event captures a specific point in the search process with all relevant data.

Placement Representation
------------------------

All placements use the canonical tuple representation where index=layer:

- **Complete placement**: ``tuple[str, ...]`` - all layers assigned to a mixer
- **Partial placement**: ``tuple[str | None, ...]`` - None for unassigned layers

Helper types for beam search events:

- ``BeamEntry``: Partial placement being expanded (parents in a step)
- ``EvaluatedCandidate``: Complete placement that was scored
- ``SelectedEntry``: Partial placement selected for next step

Event Emission Contract
-----------------------

The following table describes when each event is emitted and what triggers it.
This is the semantic contract that search strategies must follow.

**Fixed events (emitted once per search):**

- ``SearchStartedEvent``: At search start, before any allocations.
- ``AllocationPlanEvent``: After search_started, if progress estimator present.
- ``SearchCompletedEvent``: At search end, after all allocations complete.

**Per-allocation events (emitted once per allocation explored):**

- ``AllocationStartedEvent``: When beginning search within a new allocation.
- ``AllocationCompletedEvent``: When search within an allocation finishes.
- ``BestUpdatedEvent``: When allocation result improves global best (0-1 per alloc).
- ``BoundsUpdatedEvent``: When progress bounds are refined (0-1 per alloc).

**Per-phase events (emitted once per phase per allocation):**

Phases are the hierarchical search stages (one per mixer except the last).
Phases with budget=0 are skipped and emit no events.

- ``PhaseStartedEvent``: When beginning a phase within an allocation.
- ``PhaseCompletedEvent``: When a phase finishes.

**Per-step events (emitted once per beam growth step):**

Each phase has ``budget`` steps (step 1 is prescoring, steps 2..budget are growth).

- ``BeamGrowthStepStartedEvent``: Before evaluating candidates in a step.
- ``BeamGrowthStepEvent``: After step completes, with tried/pruned candidates.

**Per-evaluation events (emitted once per model evaluation):**

- ``CandidateEvaluatedEvent``: After each candidate placement is scored.
  This is by far the most frequent event (~93-99% of all events).

Scaling Formula
---------------

Total events ≈ 3 + 4A + 2P + 2S + E

Where:
- A = allocations explored (max_allocations)
- P = total phases across all allocations (sum of non-zero budgets count)
- S = total steps across all allocations (sum of all budget values)
- E = total evaluations (dominates the count)

Example: 10 allocations, 8000 evaluations → ~8,200 events (97% candidate_evaluated)
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from place_layers.search_strategy.progress.estimator import AllocationProgressInfo


# =============================================================================
# Beam entry types for strongly-typed events
# =============================================================================


class BeamEntry(BaseModel):
    """A partial placement state in the beam.

    Represents an entry being expanded during beam search. The partial_placement
    uses None for unassigned layers.
    """

    model_config = {"frozen": True}

    partial_placement: tuple[str | None, ...]  # None for unassigned layers
    score: float | None


class EvaluatedCandidate(BaseModel):
    """A candidate that was evaluated.

    Contains the complete placement (all layers assigned) that was scored,
    along with provenance information.
    """

    model_config = {"frozen": True}

    layer: int  # Layer added in this step
    score: float
    parent_idx: int
    placement: tuple[str, ...]  # Complete placement (no None values)


class SelectedEntry(BaseModel):
    """Selected for next step after pruning.

    Represents a beam entry that survived pruning, with provenance
    showing which parent it came from and which layer was added.
    """

    model_config = {"frozen": True}

    partial_placement: tuple[str | None, ...]  # None for unassigned
    score: float
    parent_idx: int
    added_layer: int


class SearchStartedEvent(BaseModel):
    """Emitted when search begins."""

    model_config = {"frozen": True}

    event_type: Literal["search_started"] = "search_started"
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

    event_type: Literal["allocation_started"] = "allocation_started"
    allocation_index: int
    total_allocations: int  # max_allocations limit
    allocation: tuple[int, ...]
    allocation_cost: float
    mixer_names: tuple[str, ...]
    # Per-allocation evaluation bounds (from progress estimator)
    eval_lower_bound: int | None = None
    eval_upper_bound: int | None = None
    eval_expected: int | None = None


class AllocationCompletedEvent(BaseModel):
    """Emitted when search within an allocation completes.

    Contains the final placement found for this allocation, plus all top
    placements from the beam for global leaderboard tracking.
    """

    model_config = {"frozen": True}

    event_type: Literal["allocation_completed"] = "allocation_completed"
    allocation_index: int
    allocation: tuple[int, ...]
    allocation_cost: float
    mixer_names: tuple[str, ...]
    # Final assignments - tuple of mixer names, one per layer
    assignments: tuple[str, ...] | None  # None if no valid assignment found
    score: float | None  # None if no valid assignment found
    # All top placements from this allocation (sorted by score descending)
    # Each entry is (assignments, score)
    top_placements: tuple[tuple[tuple[str, ...], float], ...] = ()


class PhaseStartedEvent(BaseModel):
    """Emitted when starting a new phase within hierarchical search."""

    model_config = {"frozen": True}

    event_type: Literal["phase_started"] = "phase_started"
    phase_index: int
    total_phases: int
    mixer_name: str
    budget: int  # Number of layers to assign in this phase


class BeamGrowthStepEvent(BaseModel):
    """Emitted after each beam growth step.

    Contains detailed information about what was tried and the resulting beam state.
    This enables visualization of the search evolution at each step.

    When beam_width > 1, multiple parents are expanded simultaneously. The `parents`
    field contains all beam entries being expanded, and `tried`/`beam` entries include
    provenance information (parent_idx, added_layer) to track which parent produced
    each candidate.

    All placements use the canonical representation:
    - Complete: tuple[str, ...] where index=layer, value=mixer
    - Partial: tuple[str | None, ...] where None means unassigned
    """

    model_config = {"frozen": True}

    event_type: Literal["beam_growth_step"] = "beam_growth_step"
    step: int
    total_steps: int
    phase_index: int
    mixer_name: str

    # All parents being expanded
    parents: tuple[BeamEntry, ...]

    # All candidates tried with complete placement
    tried: tuple[EvaluatedCandidate, ...]

    # Top-k after selection with provenance
    beam: tuple[SelectedEntry, ...]

    # Global best candidate (for highlighting) - None if tried is empty
    best_candidate: EvaluatedCandidate | None

    # Candidates that were pruned (not selected for beam), sorted by score descending
    cut_candidates: tuple[EvaluatedCandidate, ...]


class BestUpdatedEvent(BaseModel):
    """Emitted when a new global best is found."""

    model_config = {"frozen": True}

    event_type: Literal["best_updated"] = "best_updated"
    score: float
    cost: float
    allocation: tuple[int, ...]
    assignments: tuple[str, ...]


class PhaseCompletedEvent(BaseModel):
    """Emitted when a phase within hierarchical search completes."""

    model_config = {"frozen": True}

    event_type: Literal["phase_completed"] = "phase_completed"
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

    event_type: Literal["bounds_updated"] = "bounds_updated"
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

    event_type: Literal["allocation_plan"] = "allocation_plan"
    allocation_info: tuple["AllocationProgressInfo", ...]
    mixer_names: tuple[str, ...]


class BeamGrowthStepStartedEvent(BaseModel):
    """Emitted when a beam growth step begins (before evaluations).

    This event fires at the START of each step, allowing displays to show
    a working node that updates as candidates are evaluated.
    """

    model_config = {"frozen": True}

    event_type: Literal["beam_growth_step_started"] = "beam_growth_step_started"
    step: int
    total_steps: int
    phase_index: int
    mixer_name: str
    num_candidates: int  # How many candidates will be evaluated
    # Parents being expanded in this step
    parents: tuple[BeamEntry, ...]


class CandidateEvaluatedEvent(BaseModel):
    """Emitted after each candidate evaluation during a step.

    Allows displays to stream results as they come in, showing progress
    within a step and the current best candidate.

    The `placement` field contains the complete placement (all layers assigned)
    that was evaluated, using the canonical representation where index=layer
    and value=mixer name.
    """

    model_config = {"frozen": True}

    event_type: Literal["candidate_evaluated"] = "candidate_evaluated"
    step: int
    total_steps: int
    phase_index: int
    mixer_name: str
    # The candidate that was just evaluated
    layer: int
    score: float
    parent_idx: int
    # Complete placement that was scored
    placement: tuple[str, ...]
    # Progress within step
    eval_index: int  # 1-based index of this eval within step
    total_evals: int  # Total evals expected in this step


class SearchCompletedEvent(BaseModel):
    """Emitted when search finishes."""

    model_config = {"frozen": True}

    event_type: Literal["search_completed"] = "search_completed"
    best_score: float
    best_cost: float
    total_evaluations: int
    allocations_explored: int


# Discriminated union of all search events
AnySearchEvent = Annotated[
    SearchStartedEvent
    | AllocationStartedEvent
    | AllocationCompletedEvent
    | AllocationPlanEvent
    | PhaseStartedEvent
    | PhaseCompletedEvent
    | BeamGrowthStepStartedEvent
    | CandidateEvaluatedEvent
    | BeamGrowthStepEvent
    | BestUpdatedEvent
    | BoundsUpdatedEvent
    | SearchCompletedEvent,
    Field(discriminator="event_type"),
]

# Re-export beam entry types for use by search strategies
__all__ = [
    "BeamEntry",
    "EvaluatedCandidate",
    "SelectedEntry",
    "AnySearchEvent",
    # Event types
    "SearchStartedEvent",
    "AllocationStartedEvent",
    "AllocationCompletedEvent",
    "AllocationPlanEvent",
    "PhaseStartedEvent",
    "PhaseCompletedEvent",
    "BeamGrowthStepStartedEvent",
    "CandidateEvaluatedEvent",
    "BeamGrowthStepEvent",
    "BestUpdatedEvent",
    "BoundsUpdatedEvent",
    "SearchCompletedEvent",
]
