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

**Per-phase events (emitted once per phase per allocation, hierarchical only):**

Phases are the hierarchical search stages (one per mixer except the last).
Phases with budget=0 are skipped and emit no events.

- ``BeamPhaseStartedEvent``: When beginning a phase within an allocation.
- ``BeamPhaseCompletedEvent``: When a phase finishes.

**Per-step events (emitted once per beam growth step, hierarchical only):**

Each phase has ``budget`` steps (step 1 is prescoring, steps 2..budget are growth).

- ``BeamGrowthStepStartedEvent``: Before evaluating candidates in a step.
- ``BeamGrowthStepEvent``: After step completes, with tried/pruned candidates.

**Per-evaluation events (emitted once per model evaluation, hierarchical only):**

- ``BeamCandidateEvaluatedEvent``: After each candidate placement is scored.
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
# Beam node ID for graph reconstruction
# =============================================================================


class BeamNodeId(BaseModel):
    """Stable identifier for beam search nodes.

    Uses actual values (not indices) so IDs are self-describing without
    needing external context to interpret. Enables full graph reconstruction
    across steps and phases.

    Fields:
        allocation: Counts per mixer as tuple in phase_order,
            e.g., (("FA", 4), ("SWA", 2), ("GDN", 2)).
            IMPORTANT: The allocation dict MUST be in phase_order before
            conversion. This is guaranteed when the dict comes from
            cost_model methods (allocation_at_cost_bound, allocations_by_cost).
            Convert with tuple(allocation.items()) - do NOT sort.
        mixer: The mixer being placed in this phase, e.g., "SWA".
            None for root node (before any phase starts).
        step: Step number within the phase (0-based, 0=prescoring).
            None for root node.
        seq: Sequence number within the step (0-based, disambiguator).
            None for root node.
    """

    model_config = {"frozen": True}

    allocation: tuple[tuple[str, int], ...]
    mixer: str | None
    step: int | None
    seq: int | None


# =============================================================================
# Beam entry types for strongly-typed events
# =============================================================================


class BeamEntry(BaseModel):
    """A partial placement state in the beam.

    Represents an entry being expanded during beam search. The partial_placement
    uses None for unassigned layers.

    The `id` field is a structured BeamNodeId that enables graph reconstruction
    across steps and phases.
    """

    model_config = {"frozen": True}

    id: BeamNodeId
    partial_placement: tuple[str | None, ...]  # None for unassigned layers
    score: float | None


class EvaluatedCandidate(BaseModel):
    """A candidate that was evaluated.

    Contains the complete placement (all layers assigned) that was scored,
    along with provenance information.

    The `id` field is a structured BeamNodeId and `parent_id` links to the parent
    BeamEntry that was expanded to produce this candidate.
    """

    model_config = {"frozen": True}

    id: BeamNodeId
    parent_id: BeamNodeId | None  # ID of parent BeamEntry (None only for root)
    layer: int  # Layer added in this step
    score: float
    placement: tuple[str, ...]  # Complete placement (no None values)


class SelectedEntry(BaseModel):
    """Selected for next step after pruning.

    Represents a beam entry that survived pruning, with provenance
    showing which parent it came from and which layer was added.

    The `id` field is a structured BeamNodeId and `parent_id` links to the parent
    BeamEntry that was expanded to produce this entry.
    """

    model_config = {"frozen": True}

    id: BeamNodeId
    parent_id: BeamNodeId | None  # ID of parent BeamEntry (None only for root)
    partial_placement: tuple[str | None, ...]  # None for unassigned
    score: float
    added_layer: int


# =============================================================================
# Common/outer-loop events (all strategies)
# =============================================================================


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
    allocation: dict[str, int]
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
    allocation: dict[str, int]
    allocation_cost: float
    mixer_names: tuple[str, ...]
    # Final assignments - tuple of mixer names, one per layer
    assignments: tuple[str, ...] | None  # None if no valid assignment found
    score: float | None  # None if no valid assignment found
    # All top placements from this allocation (sorted by score descending)
    # Each entry is (assignments, score)
    top_placements: tuple[tuple[tuple[str, ...], float], ...] = ()


class BestUpdatedEvent(BaseModel):
    """Emitted when a new global best is found."""

    model_config = {"frozen": True}

    event_type: Literal["best_updated"] = "best_updated"
    score: float
    cost: float
    allocation: dict[str, int]
    assignments: tuple[str, ...]


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


class SearchCompletedEvent(BaseModel):
    """Emitted when search finishes."""

    model_config = {"frozen": True}

    event_type: Literal["search_completed"] = "search_completed"
    best_score: float
    best_cost: float
    total_evaluations: int
    allocations_explored: int


# =============================================================================
# Hierarchical beam search events
# =============================================================================


class BeamPhaseStartedEvent(BaseModel):
    """Emitted when starting a new phase within hierarchical search."""

    model_config = {"frozen": True}

    event_type: Literal["beam_phase_started"] = "beam_phase_started"
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
    provenance information (parent_id, added_layer) to track which parent produced
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


class BeamPhaseCompletedEvent(BaseModel):
    """Emitted when a phase within hierarchical search completes."""

    model_config = {"frozen": True}

    event_type: Literal["beam_phase_completed"] = "beam_phase_completed"
    phase_index: int
    total_phases: int
    mixer_name: str
    budget: int  # Number of layers assigned in this phase
    final_layers: tuple[int, ...]  # Layer indices assigned to this mixer
    final_score: float


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


class BeamCandidateEvaluatedEvent(BaseModel):
    """Emitted after each candidate evaluation during a step.

    Allows displays to stream results as they come in, showing progress
    within a step and the current best candidate.

    The `placement` field contains the complete placement (all layers assigned)
    that was evaluated, using the canonical representation where index=layer
    and value=mixer name.

    The `id` and `parent_id` fields are structured BeamNodeIds for graph
    reconstruction. Step can be accessed via `id.step`, mixer via `id.mixer`.
    """

    model_config = {"frozen": True}

    event_type: Literal["beam_candidate_evaluated"] = "beam_candidate_evaluated"
    total_steps: int
    mixer_name: str
    # The candidate that was just evaluated
    id: BeamNodeId
    parent_id: BeamNodeId | None  # None only for root
    layer: int
    score: float
    # Complete placement that was scored
    placement: tuple[str, ...]
    # Progress within step
    eval_index: int  # 1-based index of this eval within step
    total_evals: int  # Total evals expected in this step


# =============================================================================
# Swap-beam search events
# =============================================================================


class SwapBeamNodeId(BaseModel):
    """Stable identifier for swap-beam search nodes.

    Unlike hierarchical search (which builds partial placements), swap-beam
    always works with complete placements. IDs track iteration and sequence.

    Fields:
        allocation: Counts per mixer as tuple in phase_order,
            e.g., (("FA", 4), ("SWA", 2), ("GDN", 2)).
            IMPORTANT: The allocation dict MUST be in phase_order before
            conversion. This is guaranteed when the dict comes from
            cost_model methods. Convert with tuple(allocation.items()).
        iteration: Iteration number (0 = initial evaluation).
        seq: Sequence number within the iteration (0-based).
    """

    model_config = {"frozen": True}

    allocation: tuple[tuple[str, int], ...]
    iteration: int
    seq: int


class SwapCandidate(BaseModel):
    """A swap candidate in swap-beam search.

    Represents either an initial state or a state reached via a swap.
    All placements are complete (no None values).
    """

    model_config = {"frozen": True}

    id: SwapBeamNodeId
    parent_id: SwapBeamNodeId | None  # None for initial state
    swap: tuple[int, int]  # (layer_a, layer_b), or (-1, -1) for initial
    assignments: tuple[str, ...]  # Complete placement
    score: float


class SwapBeamIterationStartedEvent(BaseModel):
    """Emitted before evaluating candidates in a swap-beam iteration.

    For iteration 0, this fires after the initial evaluation is complete
    and shows the beam before any swaps. For subsequent iterations,
    it shows the beam at the start of that iteration.
    """

    model_config = {"frozen": True}

    event_type: Literal["swap_beam_iteration_started"] = "swap_beam_iteration_started"
    iteration: int  # 0 = initial, 1+ = swap iterations
    max_iterations: int
    num_candidates: int  # How many candidates will be evaluated
    beam: tuple[SwapCandidate, ...]  # Current beam state


class SwapEvaluatedEvent(BaseModel):
    """Emitted after each swap is evaluated.

    Allows displays to stream results as they come in.
    """

    model_config = {"frozen": True}

    event_type: Literal["swap_evaluated"] = "swap_evaluated"
    iteration: int
    id: SwapBeamNodeId
    parent_id: SwapBeamNodeId
    swap: tuple[int, int]  # (layer_a, layer_b)
    assignments: tuple[str, ...]  # Complete placement after swap
    score: float
    eval_index: int  # 1-based index within iteration
    total_evals: int  # Total evals expected in this iteration


class SwapBeamIterationEvent(BaseModel):
    """Emitted after a swap-beam iteration completes.

    Contains the candidates tried and the resulting beam after selection.
    """

    model_config = {"frozen": True}

    event_type: Literal["swap_beam_iteration"] = "swap_beam_iteration"
    iteration: int
    max_iterations: int
    tried: tuple[SwapCandidate, ...]  # All candidates evaluated
    beam: tuple[SwapCandidate, ...]  # Top-k after selection
    best_candidate: SwapCandidate | None  # Best from this iteration
    improved: bool  # Whether beam improved over previous iteration
    converged: bool  # True if no improvement (early termination)


# =============================================================================
# Annealing search events
# =============================================================================


class AnnealingStepEvent(BaseModel):
    """Emitted after each annealing step.

    Contains full state information for analysis and replay.
    """

    model_config = {"frozen": True}

    event_type: Literal["annealing_step"] = "annealing_step"
    step: int  # 0-based step number
    max_steps: int
    swap: tuple[int, int]  # (layer_a, layer_b) that was tried
    proposed_assignments: tuple[str, ...]  # Placement after swap
    proposed_score: float
    temperature: float  # Temperature before this step
    delta: float  # proposed_score - current_score
    acceptance_probability: float  # exp(delta/T) if delta < 0, else 1.0
    accepted: bool  # Whether the swap was accepted
    current_assignments: tuple[str, ...]  # Current state after decision
    current_score: float  # Current score after decision
    best_assignments: tuple[str, ...]  # Best seen so far
    best_score: float  # Best score seen so far


class AnnealingCompletedEvent(BaseModel):
    """Emitted when annealing within an allocation completes."""

    model_config = {"frozen": True}

    event_type: Literal["annealing_completed"] = "annealing_completed"
    steps_taken: int
    final_temperature: float
    termination_reason: Literal["max_steps", "min_temperature", "no_swaps"]
    best_assignments: tuple[str, ...]
    best_score: float


# Discriminated union of all search events
AnySearchEvent = Annotated[
    # Common/outer-loop events
    SearchStartedEvent
    | AllocationStartedEvent
    | AllocationCompletedEvent
    | AllocationPlanEvent
    | BestUpdatedEvent
    | BoundsUpdatedEvent
    | SearchCompletedEvent
    # Hierarchical beam search events
    | BeamPhaseStartedEvent
    | BeamPhaseCompletedEvent
    | BeamGrowthStepStartedEvent
    | BeamGrowthStepEvent
    | BeamCandidateEvaluatedEvent
    # Swap-beam events
    | SwapBeamIterationStartedEvent
    | SwapEvaluatedEvent
    | SwapBeamIterationEvent
    # Annealing events
    | AnnealingStepEvent
    | AnnealingCompletedEvent,
    Field(discriminator="event_type"),
]

# Re-export beam entry types for use by search strategies
__all__ = [
    "BeamNodeId",
    "BeamEntry",
    "EvaluatedCandidate",
    "SelectedEntry",
    "AnySearchEvent",
    # Common/outer-loop events
    "SearchStartedEvent",
    "AllocationStartedEvent",
    "AllocationCompletedEvent",
    "AllocationPlanEvent",
    "BestUpdatedEvent",
    "BoundsUpdatedEvent",
    "SearchCompletedEvent",
    # Hierarchical beam search events
    "BeamPhaseStartedEvent",
    "BeamPhaseCompletedEvent",
    "BeamGrowthStepStartedEvent",
    "BeamGrowthStepEvent",
    "BeamCandidateEvaluatedEvent",
    # Swap-beam types
    "SwapBeamNodeId",
    "SwapCandidate",
    "SwapBeamIterationStartedEvent",
    "SwapEvaluatedEvent",
    "SwapBeamIterationEvent",
    # Annealing types
    "AnnealingStepEvent",
    "AnnealingCompletedEvent",
]
