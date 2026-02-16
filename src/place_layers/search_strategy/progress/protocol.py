"""Observer protocol for search progress events.

The SearchObserver protocol defines the interface that display implementations
must satisfy. Using Protocol allows for structural subtyping - any class with
the right methods can be used as an observer without explicit inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from place_layers.search_strategy.progress.events import (
    AllocationCompletedEvent,
    AllocationPlanEvent,
    AllocationStartedEvent,
    AnnealingCompletedEvent,
    AnnealingReheatEvent,
    AnnealingRestartEvent,
    AnnealingStepEvent,
    BeamCandidateEvaluatedEvent,
    BeamGrowthStepEvent,
    BeamGrowthStepStartedEvent,
    BeamPhaseCompletedEvent,
    BeamPhaseStartedEvent,
    BestUpdatedEvent,
    BoundsUpdatedEvent,
    SearchCompletedEvent,
    SearchStartedEvent,
    SwapBeamIterationEvent,
    SwapBeamIterationStartedEvent,
    SwapEvaluatedEvent,
)


@runtime_checkable
class SearchObserver(Protocol):
    """Protocol for observing search progress.

    Implementations receive events at key points during the search process.
    All methods have default no-op behavior, so observers can choose which
    events to handle.
    """

    # =========================================================================
    # Common/outer-loop events (all strategies)
    # =========================================================================

    def on_search_started(self, event: SearchStartedEvent) -> None:
        """Called when search begins."""
        ...

    def on_allocation_plan(self, event: AllocationPlanEvent) -> None:
        """Called after search starts to show planned allocations."""
        ...

    def on_allocation_started(self, event: AllocationStartedEvent) -> None:
        """Called when starting search within a new allocation."""
        ...

    def on_allocation_completed(self, event: AllocationCompletedEvent) -> None:
        """Called when search within an allocation completes."""
        ...

    def on_best_updated(self, event: BestUpdatedEvent) -> None:
        """Called when a new global best is found."""
        ...

    def on_bounds_updated(self, event: BoundsUpdatedEvent) -> None:
        """Called when evaluation bounds are refined during search."""
        ...

    def on_search_completed(self, event: SearchCompletedEvent) -> None:
        """Called when search finishes."""
        ...

    def on_eval_completed(self, eval_count: int) -> None:
        """Called after each evaluation completes (throttled counter)."""
        ...

    # =========================================================================
    # Hierarchical beam search events
    # =========================================================================

    def on_beam_phase_started(self, event: BeamPhaseStartedEvent) -> None:
        """Called when starting a new phase (hierarchical search only)."""
        ...

    def on_beam_phase_completed(self, event: BeamPhaseCompletedEvent) -> None:
        """Called when a phase completes (hierarchical search only)."""
        ...

    def on_beam_growth_step_started(self, event: BeamGrowthStepStartedEvent) -> None:
        """Called when a beam growth step begins (before evaluations)."""
        ...

    def on_beam_growth_step(self, event: BeamGrowthStepEvent) -> None:
        """Called after each beam growth step."""
        ...

    def on_beam_candidate_evaluated(self, event: BeamCandidateEvaluatedEvent) -> None:
        """Called after each candidate evaluation during a beam growth step."""
        ...

    # =========================================================================
    # Swap-beam search events
    # =========================================================================

    def on_swap_beam_iteration_started(
        self, event: SwapBeamIterationStartedEvent
    ) -> None:
        """Called when a swap-beam iteration begins (before evaluations)."""
        ...

    def on_swap_evaluated(self, event: SwapEvaluatedEvent) -> None:
        """Called after each swap is evaluated during swap-beam search."""
        ...

    def on_swap_beam_iteration(self, event: SwapBeamIterationEvent) -> None:
        """Called after a swap-beam iteration completes."""
        ...

    # =========================================================================
    # Annealing search events
    # =========================================================================

    def on_annealing_step(self, event: AnnealingStepEvent) -> None:
        """Called after each annealing step."""
        ...

    def on_annealing_completed(self, event: AnnealingCompletedEvent) -> None:
        """Called when annealing within an allocation completes."""
        ...

    def on_annealing_reheat(self, event: AnnealingReheatEvent) -> None:
        """Called when adaptive reheating triggers during annealing."""
        ...

    def on_annealing_restart(self, event: AnnealingRestartEvent) -> None:
        """Called when a new SA restart begins."""
        ...
