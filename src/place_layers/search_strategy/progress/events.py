"""Events emitted during cost-constrained search.

These events form the contract between search strategies and display observers.
Each event captures a specific point in the search process with all relevant data.

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

    Contains the final placement found for this allocation, with all mixer
    assignments grouped by type for display.
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
    # Assignments grouped by mixer for compact display
    # Format: {"FA": (0, 3, 7), "SWA": (1, 2, 5), ...}
    assignments_by_mixer: dict[str, tuple[int, ...]]


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

    The `fixed` field in parents/beam entries contains assignments from prior phases,
    enabling visualization of different paths through the search tree. In phase 2+,
    different beam entries may have different fixed assignments from phase 1.
    """

    model_config = {"frozen": True}

    event_type: Literal["beam_growth_step"] = "beam_growth_step"
    step: int
    total_steps: int
    phase_index: int
    mixer_name: str

    # All parents being expanded (replaces base_layers)
    # Format: [{"layers": [...], "score": float | None, "fixed": dict}, ...]
    # The "fixed" dict maps layer_idx -> mixer_name for prior phase assignments
    parents: tuple[dict, ...]

    # All candidates tried with provenance
    # Format: [{"layer": int, "score": float, "parent_idx": int}, ...]
    tried: tuple[dict, ...]

    # Top-k after selection with provenance
    # Format: [{"layers": [...], "score": float, "parent_idx": int, "added_layer": int, "fixed": dict}, ...]
    beam: tuple[dict, ...]


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
    # Format: [{"layers": [...], "score": float | None, "fixed": dict}, ...]
    parents: tuple[dict, ...]


class CandidateEvaluatedEvent(BaseModel):
    """Emitted after each candidate evaluation during a step.

    Allows displays to stream results as they come in, showing progress
    within a step and the current best candidate.
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
