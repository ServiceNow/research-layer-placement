"""Observer protocol for search progress events.

The SearchObserver protocol defines the interface that display implementations
must satisfy. Using Protocol allows for structural subtyping - any class with
the right methods can be used as an observer without explicit inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from place_layers.search_strategy.progress.events import (
    AllocationStartedEvent,
    BeamGrowthStepEvent,
    BestUpdatedEvent,
    BoundsUpdatedEvent,
    PhaseCompletedEvent,
    PhaseStartedEvent,
    SearchCompletedEvent,
    SearchStartedEvent,
)


@runtime_checkable
class SearchObserver(Protocol):
    """Protocol for observing search progress.

    Implementations receive events at key points during the search process.
    All methods have default no-op behavior, so observers can choose which
    events to handle.
    """

    def on_search_started(self, event: SearchStartedEvent) -> None:
        """Called when search begins."""
        ...

    def on_allocation_started(self, event: AllocationStartedEvent) -> None:
        """Called when starting search within a new allocation."""
        ...

    def on_phase_started(self, event: PhaseStartedEvent) -> None:
        """Called when starting a new phase (hierarchical search only)."""
        ...

    def on_phase_completed(self, event: PhaseCompletedEvent) -> None:
        """Called when a phase completes (hierarchical search only)."""
        ...

    def on_beam_growth_step(self, event: BeamGrowthStepEvent) -> None:
        """Called after each beam growth step."""
        ...

    def on_best_updated(self, event: BestUpdatedEvent) -> None:
        """Called when a new global best is found."""
        ...

    def on_search_completed(self, event: SearchCompletedEvent) -> None:
        """Called when search finishes."""
        ...

    def on_eval_completed(self, eval_count: int) -> None:
        """Called after each evaluation completes."""
        ...

    def on_bounds_updated(self, event: BoundsUpdatedEvent) -> None:
        """Called when evaluation bounds are refined during search."""
        ...
