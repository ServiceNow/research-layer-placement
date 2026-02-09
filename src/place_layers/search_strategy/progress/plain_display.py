"""Plain text display for headless search jobs.

This module provides a simple display that prints timestamped log lines
for search progress. Unlike RichDisplay, it uses no cursor manipulation,
spinners, or progress bars - just plain print statements suitable for
CI/CD pipelines, batch jobs, and log files.

Output format (with phases/steps/leaderboard enabled):
    [14:30:22] SEARCH_STARTED: mode=max_performance, layers=48, allocations=10
    [14:30:22]   Budget <= 8.0000, cost bounds [4.0000, 20.0000]
    [14:30:22]   Mixers: FA, SWA, GDN | Expected evals: ~1,797
    [14:30:22] ALLOCATION #1/10: FA:12 SWA:8 GDN:28 (cost=8.0000)
    [14:30:23]   PHASE 1/2: +FA (budget=12)
    [14:30:23]     Starting beam:
    [14:30:23]       B0: (no assignments)
    [14:30:23]     Step 1/12: Evaluating 48 candidates...
    [14:30:23]     Step 1/12: [10/48] B0 + FA@1 + GDN@[0,2,...,47] (5.1), ...
    [14:30:23]     Step 1/12: [48/48] B0 + FA@45 + GDN@[0,1,...,44] (4.9), ...
    [14:30:23]     Step 1/12: Prune
    [14:30:23]       B0 <- B0 + FA@6 (5.2)*
    [14:30:23]       B1 <- B0 + FA@16 (4.8)
    [14:30:45]   PHASE 1/2 COMPLETE: FA@[0,2,4,6,...] -> 0.8234
    [14:30:45] ALLOCATION #1 COMPLETE: FA@[...], SWA@[...], GDN@[...] -> 0.8234
    [14:31:02] BEST_UPDATED: score=0.8567, cost=0.4500
    [14:35:00] SEARCH_COMPLETE: best=0.8567, evals=1234, allocations=10
    [14:35:00]   Leaderboard:
    [14:35:00]     1. 0.8567 (cost=0.45) FA:12 SWA:8 GDN:28

Notation:
- Phase header shows mixer being placed
- Candidate placements show parent context + diff: `B0 + FA@1 + GDN@[0,2,...]`
- Starting beam shows partial placement from prior phases
- Prune outputs show lineage: `B0 <- B1 + FA@6`

Usage:
    from place_layers.search_strategy.progress import PlainDisplay, PlainDisplayConfig

    config = PlainDisplayConfig(
        show_allocation_start=True,
        show_allocation_complete=True,
        show_best_updated=True,
        show_phases=True,
        show_steps=True,
        show_leaderboard=True,
        eval_progress_interval=10,
    )
    with config.create_observer() as observer:
        search_strategy = create_search_strategy(observers=[observer])
        result = await search_strategy.search(problem)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from place_layers.search_strategy.progress.formatter import PlacementFormatterMixin

if TYPE_CHECKING:
    from types import TracebackType

    from place_layers.search_strategy.progress.config import PlainDisplayConfig
    from place_layers.search_strategy.progress.events import (
        AllocationCompletedEvent,
        AllocationPlanEvent,
        AllocationStartedEvent,
        AnnealingCompletedEvent,
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


class PlainDisplay(PlacementFormatterMixin):
    """Plain text display for headless search jobs.

    Produces simple timestamped log lines without terminal manipulation.
    Implements the SearchObserver protocol and context manager protocol.

    Shows hierarchical output with configurable detail levels:
    - Allocation-level: no indent
    - Phase-level: 2 spaces
    - Step-level: 4 spaces

    The show_* config options control which events are printed.
    """

    def __init__(self, config: "PlainDisplayConfig") -> None:
        """Initialize the plain display.

        Args:
            config: Configuration controlling which events to show.
        """
        self._config = config
        # Mixin attributes (set in on_search_started)
        self._mixer_names: tuple[str, ...] = ()
        self._num_layers: int = 0
        # State tracking for enhanced output
        self._constraint_value: float = 0.0
        self._min_cost: float = 0.0
        self._max_cost: float = 0.0
        self._eval_expected: int | None = None
        self._current_allocation_index: int = 0
        self._max_allocations: int = 0
        # Leaderboard: list of (score, cost, allocation, placement_str)
        self._leaderboard: list[tuple[float, float, dict[str, int], str]] = []
        # State tracking for live progress
        self._step_eval_count: int = 0
        self._step_total_evals: int = 0
        # Buffer for evaluated candidates since last print: (parent_id, placement, score)
        self._step_candidates: list[tuple["BeamNodeId | None", tuple[str, ...], float]] = []
        # Parent entries for current step, keyed by ID (for diff formatting and lookup)
        from place_layers.search_strategy.progress.events import BeamEntry, BeamNodeId
        self._step_parents_by_id: dict[BeamNodeId, BeamEntry] = {}
        self._step_parent_id_to_idx: dict[BeamNodeId, int] = {}

    def __enter__(self) -> "PlainDisplay":
        """Enter context manager (no-op)."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Exit context manager (no-op)."""
        del exc_type, exc_val, exc_tb  # Unused

    def _timestamp(self) -> str:
        """Get formatted timestamp string."""
        return datetime.now().strftime(self._config.timestamp_format)

    def _print(self, event_name: str, message: str, indent: int = 0) -> None:
        """Print a timestamped event message with optional indentation."""
        indent_str = "  " * indent
        print(f"[{self._timestamp()}] {indent_str}{event_name}: {message}")

    def _print_line(self, message: str, indent: int = 0) -> None:
        """Print a timestamped line without event name."""
        indent_str = "  " * indent
        print(f"[{self._timestamp()}] {indent_str}{message}")

    # =========================================================================
    # Common/outer-loop event handlers (all strategies)
    # =========================================================================

    def on_search_started(self, event: "SearchStartedEvent") -> None:
        """Handle search started event."""
        self._mixer_names = event.mixer_names
        self._num_layers = event.num_layers
        self._constraint_value = event.constraint_value
        self._min_cost = event.min_cost
        self._max_cost = event.max_cost
        self._eval_expected = event.eval_count_expected
        self._max_allocations = event.max_allocations
        self._leaderboard = []

        self._print(
            "SEARCH_STARTED",
            f"mode={event.mode}, layers={event.num_layers}, "
            f"allocations={event.max_allocations}",
        )
        # Additional context lines
        self._print_line(
            f"Budget <= {event.constraint_value:.4f}, "
            f"cost bounds [{event.min_cost:.4f}, {event.max_cost:.4f}]",
            indent=1,
        )
        mixer_str = ", ".join(event.mixer_names)
        if event.eval_count_expected is not None:
            self._print_line(
                f"Mixers: {mixer_str} | Expected evals: ~{event.eval_count_expected:,}",
                indent=1,
            )
        else:
            self._print_line(f"Mixers: {mixer_str}", indent=1)

    def on_allocation_plan(self, event: "AllocationPlanEvent") -> None:
        """Handle allocation plan event (no-op for plain display)."""
        del event  # Not displayed in plain mode

    def on_allocation_started(self, event: "AllocationStartedEvent") -> None:
        """Handle allocation started event."""
        self._current_allocation_index = event.allocation_index
        if self._config.show_allocation_start:
            alloc_str = self._format_allocation(event.allocation)
            self._print(
                f"ALLOCATION #{event.allocation_index}/{event.total_allocations}",
                f"{alloc_str} (cost={event.allocation_cost:.4f})",
            )

    def on_allocation_completed(self, event: "AllocationCompletedEvent") -> None:
        """Handle allocation completed event."""
        # Add ALL placements from this allocation to leaderboard
        for assignments, score in event.top_placements:
            by_mixer = self._group_layers_by_mixer(assignments)
            placement_str = self._format_placement(by_mixer, max_layers_per_mixer=8)
            self._leaderboard.append(
                (score, event.allocation_cost, event.allocation, placement_str)
            )

        # Sort globally by score descending
        self._leaderboard.sort(key=lambda x: x[0], reverse=True)
        # Trim to bound memory (keep top 50)
        self._leaderboard = self._leaderboard[:50]

        if self._config.show_allocation_complete:
            alloc_str = self._format_allocation(event.allocation)
            if event.score is not None and event.assignments is not None:
                by_mixer = self._group_layers_by_mixer(event.assignments)
                placement_str = self._format_placement(by_mixer, max_layers_per_mixer=8)
                self._print(
                    f"ALLOCATION #{event.allocation_index} COMPLETE",
                    f"{placement_str} -> {event.score:.4g}",
                )
            elif event.score is not None:
                self._print(
                    f"ALLOCATION #{event.allocation_index} COMPLETE",
                    f"{alloc_str} -> {event.score:.4g}",
                )
            else:
                self._print(
                    f"ALLOCATION #{event.allocation_index} COMPLETE",
                    f"{alloc_str} -> (no valid placement)",
                )

    # =========================================================================
    # Global best tracking event handlers (all strategies)
    # =========================================================================

    def on_best_updated(self, event: "BestUpdatedEvent") -> None:
        """Handle best updated event."""
        if self._config.show_best_updated:
            self._print(
                "BEST_UPDATED",
                f"score={event.score:.4g}, cost={event.cost:.4f}",
            )

    def on_bounds_updated(self, event: "BoundsUpdatedEvent") -> None:
        """Handle bounds updated event (no-op for plain display)."""
        del event  # Not displayed in plain mode

    def on_search_completed(self, event: "SearchCompletedEvent") -> None:
        """Handle search completed event."""
        self._print(
            "SEARCH_COMPLETE",
            f"best={event.best_score:.4g}, evals={event.total_evaluations}, "
            f"allocations={event.allocations_explored}",
        )

        # Print leaderboard if enabled and we have results
        if self._config.show_leaderboard and self._leaderboard:
            self._print_line("Leaderboard:", indent=1)
            for rank, (score, cost, allocation, _placement_str) in enumerate(
                self._leaderboard, start=1
            ):
                alloc_str = self._format_allocation(allocation)
                self._print_line(
                    f"{rank}. {score:.4g} (cost={cost:.2f}) {alloc_str}",
                    indent=2,
                )

    def on_eval_completed(self, eval_count: int) -> None:
        """Handle evaluation completed (no-op for plain display)."""
        del eval_count  # Not displayed in plain mode

    # =========================================================================
    # Hierarchical beam search event handlers
    # =========================================================================

    def on_beam_phase_started(self, event: "BeamPhaseStartedEvent") -> None:
        """Handle beam phase started event."""
        if self._config.show_phases:
            self._print(
                f"PHASE {event.phase_index + 1}/{event.total_phases}",
                f"+{event.mixer_name} (budget={event.budget})",
                indent=1,
            )

    def on_beam_phase_completed(self, event: "BeamPhaseCompletedEvent") -> None:
        """Handle beam phase completed event."""
        if self._config.show_phases:
            layers_str = self._format_layer_assignment(
                event.mixer_name, event.final_layers, max_show=10
            )
            self._print(
                f"PHASE {event.phase_index + 1}/{event.total_phases} COMPLETE",
                f"{layers_str} -> {event.final_score:.4g}",
                indent=1,
            )

    def on_beam_growth_step_started(self, event: "BeamGrowthStepStartedEvent") -> None:
        """Handle beam growth step started - show initial beam and progress."""
        # Always capture parent placements for diff formatting (used by on_beam_candidate_evaluated)
        # Build lookups from parent ID to index and parent entry
        self._step_parents_by_id = {p.id: p for p in event.parents}
        self._step_parent_id_to_idx = {p.id: i for i, p in enumerate(event.parents)}

        if not self._config.show_steps:
            return

        # Reset tracking state
        self._step_eval_count = 0
        self._step_total_evals = event.num_candidates
        self._step_candidates = []

        # For step 1, show starting beam state (one entry per line)
        if event.step == 1:
            self._print_line("Starting beam:", indent=2)
            for i, parent in enumerate(event.parents):
                # Format partial placement (None entries are skipped)
                by_mixer = self._group_layers_by_mixer(parent.partial_placement)
                placement_str = self._format_placement(by_mixer)
                if not placement_str:
                    placement_str = "(no assignments)"
                self._print_line(f"B{i}: {placement_str}", indent=3)

        self._print_line(
            f"Step {event.step}/{event.total_steps}: "
            f"Evaluating {event.num_candidates} candidates...",
            indent=2,
        )

    def on_beam_candidate_evaluated(self, event: "BeamCandidateEvaluatedEvent") -> None:
        """Handle beam candidate evaluated - show all candidates at intervals."""
        if not self._config.show_steps:
            return

        self._step_eval_count = event.eval_index

        # Buffer evaluated candidate with parent ID and complete placement
        self._step_candidates.append((event.parent_id, event.placement, event.score))

        # Print progress at intervals or at end
        interval = self._config.eval_progress_interval
        at_interval = interval > 0 and event.eval_index % interval == 0
        at_end = event.eval_index == event.total_evals

        if at_interval or at_end:
            # Format all buffered candidates with parent context and diff notation
            candidate_strs = []
            for parent_id, placement, score in self._step_candidates:
                parent_idx = self._step_parent_id_to_idx.get(parent_id, 0) if parent_id else 0
                parent_entry = self._step_parents_by_id.get(parent_id) if parent_id else None
                parent_placement = parent_entry.partial_placement if parent_entry else ()
                diff = self._format_placement_diff(parent_placement, placement)
                candidate_strs.append(f"B{parent_idx} + {diff} ({score:.3g})")
            candidates_str = ", ".join(candidate_strs)

            # step is always set for evaluated candidates (only root has step=None)
            step = event.id.step if event.id.step is not None else 0
            self._print_line(
                f"Step {step + 1}/{event.total_steps}: "
                f"[{event.eval_index}/{event.total_evals}] {candidates_str}",
                indent=2,
            )
            # Clear buffer after printing
            self._step_candidates = []

    def on_beam_growth_step(self, event: "BeamGrowthStepEvent") -> None:
        """Handle beam growth step event - show prune results with lineage."""
        if not self._config.show_steps:
            return

        # Print all beam entries, one per line (no limit)
        if event.beam:
            self._print_line(
                f"Step {event.step}/{event.total_steps}: Prune",
                indent=2,
            )
            for i, entry in enumerate(event.beam):
                marker = "*" if i == 0 else ""
                # Format: B0 <- B1 + FA@6 (5.2)*
                # Use diff formatter to show what changed from parent
                parent_idx = self._step_parent_id_to_idx.get(entry.parent_id, 0) if entry.parent_id else 0
                parent_entry = self._step_parents_by_id.get(entry.parent_id) if entry.parent_id else None
                parent_partial = parent_entry.partial_placement if parent_entry else ()
                diff = self._format_placement_diff(parent_partial, entry.partial_placement)
                self._print_line(
                    f"B{i} <- B{parent_idx} + {diff} "
                    f"({entry.score:.3g}){marker}",
                    indent=3,
                )

    # =========================================================================
    # Swap-beam search event handlers
    # =========================================================================

    def on_swap_beam_iteration_started(
        self, event: "SwapBeamIterationStartedEvent"
    ) -> None:
        """Handle swap-beam iteration started event."""
        if not self._config.show_steps:
            return

        if event.iteration == 0:
            # Initial state - show beam
            by_mixer = self._group_layers_by_mixer(event.beam[0].assignments)
            placement_str = self._format_placement(by_mixer, max_layers_per_mixer=8)
            self._print_line(
                f"Initial: {placement_str} (score={event.beam[0].score:.4g})",
                indent=1,
            )
        else:
            self._print_line(
                f"Iteration {event.iteration}/{event.max_iterations}: "
                f"Evaluating {event.num_candidates} swaps...",
                indent=1,
            )

    def on_swap_evaluated(self, event: "SwapEvaluatedEvent") -> None:
        """Handle swap evaluated event (no-op for plain display)."""
        # Could be made configurable to show per-swap progress
        del event

    def on_swap_beam_iteration(self, event: "SwapBeamIterationEvent") -> None:
        """Handle swap-beam iteration completed event."""
        if not self._config.show_steps:
            return

        if event.iteration == 0:
            return  # Already shown in iteration_started

        best = event.best_candidate
        status = "converged" if event.converged else ("improved" if event.improved else "no change")
        best_str = f"best={best.score:.4g}" if best else "no candidates"
        self._print_line(
            f"Iteration {event.iteration}/{event.max_iterations}: "
            f"{status}, {best_str}",
            indent=1,
        )

    # =========================================================================
    # Annealing search event handlers
    # =========================================================================

    def on_annealing_step(self, event: "AnnealingStepEvent") -> None:
        """Handle annealing step event (show at intervals)."""
        if not self._config.show_steps:
            return

        # Show every N steps (configurable via eval_progress_interval)
        interval = max(1, self._config.eval_progress_interval)
        at_interval = (event.step + 1) % interval == 0
        at_end = event.step + 1 == event.max_steps

        if at_interval or at_end:
            accept_str = "accepted" if event.accepted else "rejected"
            self._print_line(
                f"Step {event.step + 1}/{event.max_steps}: "
                f"T={event.temperature:.3g}, delta={event.delta:+.4g}, "
                f"{accept_str}, best={event.best_score:.4g}",
                indent=1,
            )

    def on_annealing_completed(self, event: "AnnealingCompletedEvent") -> None:
        """Handle annealing completed event."""
        if not self._config.show_steps:
            return

        by_mixer = self._group_layers_by_mixer(event.best_assignments)
        placement_str = self._format_placement(by_mixer, max_layers_per_mixer=8)
        self._print_line(
            f"Annealing complete ({event.termination_reason}): "
            f"{event.steps_taken} steps, best={event.best_score:.4g}",
            indent=1,
        )
        if self._config.show_phases:
            self._print_line(f"Best: {placement_str}", indent=2)
