"""Rich-based live display for search progress.

This module implements a visual progress display using Rich's Live component
to show real-time search status including:
- Search configuration (mode, bounds, budget)
- Current allocation being explored
- Phase and beam progress (for hierarchical search)
- Best candidates in beam
- Global best found
- Evaluation count with progress bar

The display uses a hybrid print+live approach:
- Completed steps/phases: Printed to terminal as permanent output (scrolls up)
- Current state: Rich Live display at bottom showing active step

Progress bounds are received via BoundsUpdatedEvent from the search strategy,
which owns the progress estimator and emits events as allocations complete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from place_layers.search_strategy.progress.events import (
    AllocationCompletedEvent,
    AllocationPlanEvent,
    AllocationStartedEvent,
    BeamGrowthStepEvent,
    BeamGrowthStepStartedEvent,
    BoundsUpdatedEvent,
    CandidateEvaluatedEvent,
    PhaseCompletedEvent,
    PhaseStartedEvent,
    SearchCompletedEvent,
    SearchStartedEvent,
)

if TYPE_CHECKING:
    from types import TracebackType

    from place_layers.search_strategy.progress.events import BestUpdatedEvent


class LeaderboardEntry(BaseModel):
    """An entry in the results leaderboard."""

    score: float
    cost: float
    allocation: tuple[int, ...]  # e.g., (2, 1, 7) for m0:2, m1:1, m2:7
    assignments: tuple[str, ...]  # Per-layer mixer assignments


class StepData(BaseModel):
    """Data for a completed beam growth step."""

    step: int
    total_steps: int
    expand_content: str  # Formatted expand section
    prune_content: str  # Formatted prune section


class PhaseData(BaseModel):
    """Data for a phase (completed or in-progress)."""

    phase_index: int
    total_phases: int
    mixer_name: str
    budget: int
    steps: list[StepData] = []
    completion_text: str | None = None  # Set when phase completes


class RichSearchDisplay:
    """Rich Live display for search progress.

    Uses the observer pattern to receive events from the search strategy
    and updates the display accordingly. Thread-safe through Rich's Live.

    The display uses a hybrid approach:
    - Completed steps are printed permanently to the terminal
    - Current state is shown in a live display at the bottom

    Usage:
        with RichSearchDisplay() as display:
            search_strategy = create_search_strategy(observer=display)
            await search_strategy.search(problem)
    """

    # Throttle eval updates to avoid display flicker
    _EVAL_UPDATE_INTERVAL = 10

    def __init__(
        self,
        console: Console | None = None,
    ) -> None:
        self._console = console or Console()
        self._live: Live | None = None

        # State from events
        self._mode: str = ""
        self._constraint_value: float = 0.0
        self._min_cost: float = 0.0
        self._max_cost: float = 0.0
        self._num_layers: int = 0
        self._mixer_names: tuple[str, ...] = ()
        self._max_allocations: int = 0

        self._allocation_index: int = 0
        self._current_allocation: tuple[int, ...] = ()
        self._current_allocation_cost: float = 0.0

        # Leaderboard of completed allocation results
        self._leaderboard: list[LeaderboardEntry] = []

        self._eval_count: int = 0
        self._last_eval_update: int = 0

        # Overall eval count bounds (from BoundsUpdatedEvent)
        self._eval_lower_bound: int | None = None
        self._eval_upper_bound: int | None = None
        self._eval_expected: int | None = None

        # Per-allocation progress tracking
        self._alloc_eval_start: int = 0  # eval_count when allocation started
        self._alloc_expected: int | None = None

        # Accumulated state for nested panel rendering
        self._completed_phases: list[PhaseData] = []
        self._current_phase: PhaseData | None = None
        self._current_step_content: str = ""  # In-progress step text

        # Track current step for streaming candidate results
        self._step_results: list[dict] = []  # Accumulated results during step
        self._current_step: int = 0
        self._current_total_steps: int = 0

    def __enter__(self) -> "RichSearchDisplay":
        """Start the Live display."""
        self._live = Live(
            self._build_display(),
            console=self._console,
            refresh_per_second=4,
            transient=True,
            vertical_overflow="visible",
        )
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the Live display."""
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)

    def _update_live(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_display())

    def _build_display(self) -> Group:
        """Build the complete nested Panel structure from current state."""
        components = []

        # Build allocation panel with nested phases
        if self._allocation_index > 0 and self._current_allocation:
            allocation_content = self._build_allocation_content()
            allocation_panel = Panel(
                allocation_content,
                title=f"[bold yellow]Allocation {self._allocation_index}/{self._max_allocations}[/bold yellow]",
                border_style="yellow",
            )
            components.append(allocation_panel)

        # Leaderboard panel
        if self._leaderboard:
            components.append(self._render_leaderboard())

        # Progress bar
        components.append(self._render_progress())

        return Group(*components)

    def _build_allocation_content(self) -> Group:
        """Build content for allocation panel: phases with nested steps."""
        components: list[Text | Panel] = []

        # Allocation summary line
        alloc_str = self._format_allocation(self._current_allocation)
        components.append(
            Text.from_markup(f"{alloc_str}  [dim]cost={self._current_allocation_cost:.4f}[/dim]")
        )

        # Completed phases
        for phase in self._completed_phases:
            phase_panel = self._build_phase_panel(phase, is_current=False)
            components.append(phase_panel)

        # Current phase (in progress)
        if self._current_phase is not None:
            phase_panel = self._build_phase_panel(self._current_phase, is_current=True)
            components.append(phase_panel)

        return Group(*components)

    def _build_phase_panel(self, phase: PhaseData, is_current: bool) -> Panel:
        """Build a phase panel with nested step panels."""
        step_components: list[Panel | Text] = []

        for step in phase.steps:
            step_panel = Panel(
                Group(
                    Text.from_markup(step.expand_content),
                    Text.from_markup(step.prune_content),
                ),
                title=f"Step {step.step}/{step.total_steps}",
                border_style="dim",
            )
            step_components.append(step_panel)

        # In-progress step (if this is current phase and we have content)
        if is_current and self._current_step_content:
            step_panel = Panel(
                Text.from_markup(self._current_step_content),
                title=f"Step {self._current_step}/{self._current_total_steps}",
                border_style="dim blue",
            )
            step_components.append(step_panel)

        # Completion text
        if phase.completion_text:
            step_components.append(Text.from_markup(phase.completion_text))

        return Panel(
            Group(*step_components) if step_components else Text(""),
            title=f"[bold green]Phase {phase.phase_index}/{phase.total_phases}[/bold green]: "
            f"Place [cyan]{phase.mixer_name}[/cyan] (budget={phase.budget})",
            border_style="green" if phase.completion_text else "blue",
        )

    def _render_leaderboard(self) -> Panel:
        """Render the leaderboard table."""
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", justify="right", style="dim")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Cost", justify="right")
        table.add_column("Allocation")
        table.add_column("Placement")

        for i, entry in enumerate(self._leaderboard[:5], 1):  # Top 5
            alloc_str = self._format_allocation(entry.allocation)
            placement_str = self._format_placement(entry.assignments)
            table.add_row(
                str(i),
                f"{entry.score:.4f}",
                f"{entry.cost:.4f}",
                alloc_str,
                placement_str,
            )

        return Panel(table, title="[bold]Leaderboard[/bold]", border_style="green")

    def _render_progress(self) -> Group:
        """Render the dual progress bars (overall and per-allocation)."""
        components = []

        # Overall progress bar using expected value
        overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]Overall:[/bold]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self._console,
            transient=False,
            expand=False,
        )

        # Use expected value for progress calculation (more accurate than bounds midpoint)
        if self._eval_expected is not None:
            overall_progress.add_task(
                "Overall",
                total=self._eval_expected,
                completed=self._eval_count,
            )
        elif self._eval_upper_bound is not None:
            overall_progress.add_task(
                "Overall",
                total=self._eval_upper_bound,
                completed=self._eval_count,
            )
        else:
            overall_progress.add_task("Overall", total=None)

        components.append(overall_progress)

        # Per-allocation progress bar (if we have allocation bounds)
        if self._alloc_expected is not None:
            alloc_evals = self._eval_count - self._alloc_eval_start
            alloc_progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]Allocation:[/bold]"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self._console,
                transient=False,
                expand=False,
            )
            alloc_progress.add_task(
                "Allocation",
                total=self._alloc_expected,
                completed=alloc_evals,
            )
            components.append(alloc_progress)

        return Group(*components)

    # =========================================================================
    # Print methods for permanent output using Rich widgets
    # =========================================================================

    def _print_search_header(self) -> None:
        """Print search header as a Rich panel."""
        mode_str = "MAX PERFORMANCE" if self._mode == "max_performance" else "MIN COST"
        if self._mode == "max_performance":
            constraint_str = f"Budget ≤ {self._constraint_value:.4f}"
        else:
            constraint_str = f"Target ≥ {self._constraint_value:.4f}"

        # Build header table
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column()

        table.add_row("Mode", mode_str)
        table.add_row("Constraint", constraint_str)
        table.add_row("Cost bounds", f"[{self._min_cost:.4f}, {self._max_cost:.4f}]")
        table.add_row("Layers", str(self._num_layers))
        table.add_row("Mixers", ", ".join(self._mixer_names))
        if self._eval_expected is not None:
            table.add_row("Expected evals", f"~{self._eval_expected:,}")
            if self._eval_lower_bound is not None and self._eval_upper_bound is not None:
                table.add_row(
                    "95% CI", f"[{self._eval_lower_bound:,}, {self._eval_upper_bound:,}]"
                )
        elif self._eval_lower_bound is not None and self._eval_upper_bound is not None:
            table.add_row(
                "Est. evals", f"{self._eval_lower_bound:,} - {self._eval_upper_bound:,}"
            )

        panel = Panel(
            table,
            title="[bold]Layer Placement Optimization[/bold]",
            border_style="blue",
        )
        self._console.print()
        self._console.print(panel)

    def _print_allocation_complete(self, event: AllocationCompletedEvent) -> None:
        """Print allocation completion summary with full placement."""
        if event.assignments is None or event.score is None:
            self._console.print("[dim]  No valid placement found for this allocation[/dim]")
            return

        # Format assignments by mixer in canonical order using _format_placement
        # which accepts dict[str, tuple[int, ...]] format
        placement_str = self._format_placement(
            event.assignments_by_mixer, max_layers_per_mixer=8, escape_brackets=True
        )

        self._console.print(
            f"[bold cyan]✓ Allocation {event.allocation_index} complete[/bold cyan]: "
            f"{placement_str} → {event.score:.4g}"
        )

    # =========================================================================
    # Formatting primitives and building blocks
    # =========================================================================

    def _format_layer_list(
        self, layers: tuple[int, ...] | list[int], max_show: int = 10
    ) -> str:
        """Format a list of layer indices compactly.

        Args:
            layers: Layer indices to format
            max_show: Maximum layers to show before truncating

        Returns:
            Comma-separated string like "1,2,3" or "1,2,...,9" if truncated
        """
        if len(layers) <= max_show:
            return ",".join(str(l) for l in layers)
        first_half = max_show // 2
        return ",".join(str(l) for l in layers[:first_half]) + f",...,{layers[-1]}"

    def _format_layer_assignment(
        self,
        mixer: str,
        layers: tuple[int, ...] | list[int],
        max_show: int = 10,
        escape_brackets: bool = False,
    ) -> str:
        """Format a single mixer's layer assignment.

        Args:
            mixer: Mixer name (e.g., "FA", "SWA")
            layers: Layer indices assigned to this mixer
            max_show: Maximum layers to show before truncating
            escape_brackets: Whether to escape brackets for Rich markup

        Returns:
            String like "FA@[1,2,3]" or "FA@\\[1,2,3]" if escaped
        """
        layers_str = self._format_layer_list(layers, max_show)
        bracket = "\\[" if escape_brackets else "["
        return f"{mixer}@{bracket}{layers_str}]"

    def _group_layers_by_mixer(
        self,
        assignments: tuple[str, ...] | dict[int, str],
    ) -> dict[str, list[int]]:
        """Group layer indices by their assigned mixer.

        Args:
            assignments: Either per-layer tuple or dict mapping layer→mixer

        Returns:
            Dict mapping mixer name to sorted list of layer indices
        """
        by_mixer: dict[str, list[int]] = {}

        if isinstance(assignments, dict):
            # dict[int, str] format (layer → mixer)
            for layer, mixer in assignments.items():
                by_mixer.setdefault(mixer, []).append(layer)
        else:
            # tuple[str, ...] format (index = layer, value = mixer)
            for layer_idx, mixer in enumerate(assignments):
                by_mixer.setdefault(mixer, []).append(layer_idx)

        # Sort layers within each mixer
        for layers in by_mixer.values():
            layers.sort()

        return by_mixer

    def _format_context_string(
        self,
        prior_assignments: dict[int, str],
        current_mixer: str | None,
        current_layers: tuple[int, ...] | list[int],
        max_show: int = 6,
    ) -> str:
        """Format combined prior and current phase context for display.

        Args:
            prior_assignments: Dict mapping layer→mixer for prior phase assignments
            current_mixer: Current phase mixer name (or None)
            current_layers: Current phase layer indices
            max_show: Maximum layers to show per mixer before truncating

        Returns:
            String like " ([dim]FA@[0,5][/dim], SWA@[1,2,3])" or "" if empty.
        """
        prior_ctx = ""
        if prior_assignments:
            by_mixer = self._group_layers_by_mixer(prior_assignments)
            parts = [
                self._format_layer_assignment(m, by_mixer[m], max_show, escape_brackets=True)
                for m in self._mixer_names
                if m in by_mixer
            ]
            prior_ctx = ", ".join(parts)

        current_ctx = ""
        if current_mixer and current_layers:
            current_ctx = self._format_layer_assignment(
                current_mixer, current_layers, max_show, escape_brackets=True
            )

        if prior_ctx and current_ctx:
            return f" ([dim]{prior_ctx}[/dim], {current_ctx})"
        elif prior_ctx:
            return f" ([dim]{prior_ctx}[/dim])"
        elif current_ctx:
            return f" ({current_ctx})"
        return ""

    def _format_allocation(self, allocation: tuple[int, ...]) -> str:
        """Format allocation as 'FA:2 SWA:1 GDN:7'."""
        parts = [f"{name}:{count}" for name, count in zip(self._mixer_names, allocation)]
        return " ".join(parts)

    def _format_placement(
        self,
        assignments: tuple[str, ...] | dict[str, tuple[int, ...]],
        max_layers_per_mixer: int = 10,
        escape_brackets: bool = False,
    ) -> str:
        """Format per-layer assignments as grouped placement string.

        Args:
            assignments: Either per-layer tuple (index=layer, value=mixer)
                        or dict mapping mixer→layers (for pre-grouped data)
            max_layers_per_mixer: Max layers to show per mixer before truncating
            escape_brackets: Whether to escape brackets for Rich markup

        Returns:
            String like "FA@[0,2], GDN@[1], SWA@[3,4,5]"
        """
        # Handle pre-grouped dict format
        if isinstance(assignments, dict):
            by_mixer = {mixer: list(layers) for mixer, layers in assignments.items()}
        else:
            by_mixer = self._group_layers_by_mixer(assignments)

        # Format in canonical mixer_names order
        parts = []
        for mixer in self._mixer_names:
            if mixer in by_mixer:
                parts.append(
                    self._format_layer_assignment(
                        mixer, by_mixer[mixer], max_layers_per_mixer, escape_brackets
                    )
                )
        return ", ".join(parts)

    def _format_expand_section(self, event: BeamGrowthStepEvent) -> str:
        """Format the expand section of a beam step as text."""
        lines = []
        # Use unified B naming and show step context
        if event.step == 1:
            lines.append("Expand initial beam:")
        else:
            lines.append(f"Expand beam from step {event.step - 1}:")

        # Group tried candidates by parent_idx
        tried_by_parent: dict[int, list[dict]] = {}
        for t in event.tried:
            parent_idx = t.get("parent_idx", 0)
            if parent_idx not in tried_by_parent:
                tried_by_parent[parent_idx] = []
            tried_by_parent[parent_idx].append(t)

        # Sort each parent's tried list by score descending
        for parent_idx in tried_by_parent:
            tried_by_parent[parent_idx].sort(key=lambda x: x["score"], reverse=True)

        # Find global best across all tried candidates
        global_best = max(event.tried, key=lambda x: x["score"]) if event.tried else None
        global_best_key = (
            (global_best["parent_idx"], global_best["layer"]) if global_best else None
        )

        # Show each parent and its tried candidates
        for parent_idx, parent in enumerate(event.parents):
            parent_layers = parent.get("layers", [])

            # Use helper to format context string
            context_str = self._format_context_string(
                parent.get("fixed", {}), event.mixer_name, parent_layers, max_show=6
            )

            # Format tried candidates for this parent
            tried_list = tried_by_parent.get(parent_idx, [])
            tried_parts = []
            for t in tried_list[:5]:
                is_global_best = (parent_idx, t["layer"]) == global_best_key
                if is_global_best:
                    tried_parts.append(f"[green]+{t['layer']}→{t['score']:.4g}★[/green]")
                else:
                    tried_parts.append(f"+{t['layer']}→{t['score']:.4g}")
            tried_str = "  ".join(tried_parts)
            if len(tried_list) > 5:
                tried_str += "  [dim]...[/dim]"

            lines.append(f"  B{parent_idx}{context_str}: {tried_str}")

        return "\n".join(lines)

    def _format_prune_section(self, event: BeamGrowthStepEvent) -> str:
        """Format the prune section of a beam step as text."""
        lines = []
        beam_size = len(event.beam)
        lines.append(f"Prune to {beam_size}:")

        for i, b in enumerate(event.beam):
            parent_idx = b.get("parent_idx", 0)
            added_layer = b.get("added_layer", "?")

            # Get parent's layers and append the added layer
            parent = event.parents[parent_idx] if parent_idx < len(event.parents) else {}
            parent_layers = list(parent.get("layers", []))
            if added_layer != "?":
                parent_layers.append(added_layer)

            # Use helper to format context string
            context_str = self._format_context_string(
                parent.get("fixed", {}), event.mixer_name, parent_layers, max_show=6
            )

            # Show lineage: B0 ← B{parent}+{layer}
            lineage = f"← B{parent_idx}+{added_layer}"

            if i == 0:
                lines.append(f"  [green]B{i} {lineage}{context_str} → {b['score']:.4g}★[/green]")
            else:
                lines.append(f"  B{i} {lineage}{context_str} → {b['score']:.4g}")

        # Show runners-up (candidates that just missed the cut)
        beam_keys = {(b.get("parent_idx"), b.get("added_layer")) for b in event.beam}
        cut_candidates = [
            t
            for t in event.tried
            if (t.get("parent_idx"), t["layer"]) not in beam_keys
        ]
        # Sort by score descending to show runners-up first
        cut_candidates.sort(key=lambda x: x["score"], reverse=True)

        if cut_candidates:
            # Show top 2-3 runners-up in compact format
            shown = cut_candidates[:3]
            cut_parts = [f"+{c['layer']}→{c['score']:.4g}" for c in shown]
            cut_str = ", ".join(cut_parts)

            remaining = len(cut_candidates) - len(shown)
            if remaining > 0:
                cut_str += f"  (+{remaining} more)"

            lines.append(f"  [dim]cut: {cut_str}[/dim]")

        return "\n".join(lines)

    def _print_search_complete(self, event: SearchCompletedEvent) -> None:
        """Print search completion as a Rich panel with leaderboard."""
        # Summary table
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column(style="green" if event.best_score else "")

        summary.add_row("Best score", f"{event.best_score:.4f}")
        summary.add_row("Best cost", f"{event.best_cost:.4f}")
        summary.add_row("Evaluations", f"{event.total_evaluations:,}")
        summary.add_row("Allocations explored", str(event.allocations_explored))

        # Build leaderboard table
        if self._leaderboard:
            leaderboard_table = Table(
                show_header=True,
                header_style="bold",
                box=None,
                padding=(0, 1),
            )
            leaderboard_table.add_column("#", justify="right", style="dim")
            leaderboard_table.add_column("Score", justify="right", style="green")
            leaderboard_table.add_column("Cost", justify="right")
            leaderboard_table.add_column("Allocation")
            leaderboard_table.add_column("Placement")

            for i, entry in enumerate(self._leaderboard, 1):
                alloc_str = self._format_allocation(entry.allocation)
                placement_str = self._format_placement(entry.assignments)
                leaderboard_table.add_row(
                    str(i),
                    f"{entry.score:.4f}",
                    f"{entry.cost:.4f}",
                    alloc_str,
                    placement_str,
                )

            # Combine into group
            content = Group(
                summary,
                Text(),  # Spacer
                Text("Final Leaderboard:", style="bold"),
                leaderboard_table,
            )
        else:
            content = summary

        panel = Panel(
            content,
            title="[bold green]Search Complete[/bold green]",
            border_style="green",
        )
        self._console.print()
        self._console.print(panel)

    # =========================================================================
    # Observer interface methods
    # =========================================================================

    def on_search_started(self, event: SearchStartedEvent) -> None:
        """Handle search started event."""
        self._mode = event.mode
        self._constraint_value = event.constraint_value
        self._min_cost = event.min_cost
        self._max_cost = event.max_cost
        self._num_layers = event.num_layers
        self._mixer_names = event.mixer_names
        self._max_allocations = event.max_allocations

        # Initialize bounds from event (may be None if not provided)
        self._eval_lower_bound = event.eval_count_lower_bound
        self._eval_upper_bound = event.eval_count_upper_bound
        self._eval_expected = event.eval_count_expected

        # Reset accumulated state for fresh search
        self._completed_phases = []
        self._current_phase = None
        self._current_step_content = ""
        self._leaderboard = []

        # Print search header using Rich panel
        self._print_search_header()

        self._update_live()

    def on_allocation_plan(self, event: AllocationPlanEvent) -> None:
        """Handle allocation plan event - show summary table."""
        if len(event.allocation_info) > 20:
            return  # Too many to display as table

        table = Table(show_header=True, header_style="bold", padding=(0, 1))
        table.add_column("#", justify="right", style="dim")
        table.add_column("Allocation", no_wrap=True)
        table.add_column("Cost", justify="right")
        table.add_column("Expected evals", justify="right")
        table.add_column("95% CI", justify="right", no_wrap=True)

        for i, info in enumerate(event.allocation_info, 1):
            alloc_str = self._format_allocation(info.allocation)
            table.add_row(
                str(i),
                alloc_str,
                f"{info.cost:.4f}",
                f"~{int(info.expected):,}",
                f"[{info.estimated_lower:,}, {info.estimated_upper:,}]",
            )

        self._console.print()
        self._console.print(
            Panel(table, title="[bold]Per-Allocation Breakdown[/bold]", border_style="yellow")
        )

    def on_allocation_started(self, event: AllocationStartedEvent) -> None:
        """Handle allocation started event."""
        self._allocation_index = event.allocation_index
        self._current_allocation = event.allocation
        self._current_allocation_cost = event.allocation_cost

        # Reset accumulated state for new allocation
        self._completed_phases = []
        self._current_phase = None
        self._current_step_content = ""

        # Capture per-allocation bounds for progress tracking
        self._alloc_eval_start = self._eval_count
        self._alloc_expected = event.eval_expected

        self._update_live()

    def on_allocation_completed(self, event: AllocationCompletedEvent) -> None:
        """Handle allocation completed event."""
        # Print allocation completion summary
        self._print_allocation_complete(event)

        # Add to leaderboard if valid result
        if event.score is not None and event.assignments is not None:
            entry = LeaderboardEntry(
                score=event.score,
                cost=event.allocation_cost,
                allocation=event.allocation,
                assignments=event.assignments,
            )
            self._leaderboard.append(entry)
            # Sort by score descending
            self._leaderboard.sort(key=lambda e: e.score, reverse=True)

        # Clear phases for next allocation
        self._completed_phases = []
        self._current_phase = None
        self._current_step_content = ""

        self._update_live()

    def on_phase_started(self, event: PhaseStartedEvent) -> None:
        """Handle phase started event."""
        # Create new current phase for nested panel display
        self._current_phase = PhaseData(
            phase_index=event.phase_index,
            total_phases=event.total_phases,
            mixer_name=event.mixer_name,
            budget=event.budget,
            steps=[],
        )

        self._update_live()

    def on_beam_growth_step_started(self, event: BeamGrowthStepStartedEvent) -> None:
        """Handle beam growth step started - set in-progress step content."""
        self._step_results = []
        self._current_step = event.step
        self._current_total_steps = event.total_steps

        # Set in-progress step content
        self._current_step_content = f"[dim]evaluating 0/{event.num_candidates}...[/dim]"
        self._update_live()

    def on_candidate_evaluated(self, event: CandidateEvaluatedEvent) -> None:
        """Handle individual candidate result - stream into display."""
        # Accumulate result
        self._step_results.append({
            "layer": event.layer,
            "score": event.score,
            "parent_idx": event.parent_idx,
        })

        # Sort by score to find current best
        sorted_results = sorted(self._step_results, key=lambda x: x["score"], reverse=True)
        best = sorted_results[0] if sorted_results else None

        # Update in-progress step content
        progress_text = f"evaluating {event.eval_index}/{event.total_evals}"
        if best:
            best_text = f" [green]best: +{best['layer']}→{best['score']:.4g}[/green]"
        else:
            best_text = ""

        self._current_step_content = f"[dim]{progress_text}[/dim]{best_text}"

        self._update_live()

    def on_beam_growth_step(self, event: BeamGrowthStepEvent) -> None:
        """Handle beam growth step event."""
        # Format step content and add to current phase
        expand_content = self._format_expand_section(event)
        prune_content = self._format_prune_section(event)

        step_data = StepData(
            step=event.step,
            total_steps=event.total_steps,
            expand_content=expand_content,
            prune_content=prune_content,
        )

        if self._current_phase is not None:
            self._current_phase.steps.append(step_data)

        # Clear in-progress step
        self._current_step_content = ""
        self._step_results = []

        self._update_live()

    def on_phase_completed(self, event: PhaseCompletedEvent) -> None:
        """Handle phase completed event."""
        if self._current_phase is not None:
            # Use building block for consistent formatting
            assignment_str = self._format_layer_assignment(
                event.mixer_name, event.final_layers, max_show=10, escape_brackets=True
            )
            # Set completion text
            self._current_phase.completion_text = (
                f"[bold green]✓ Complete[/bold green]: "
                f"{assignment_str} → {event.final_score:.4g}"
            )
            # Move to completed
            self._completed_phases.append(self._current_phase)
            self._current_phase = None

        self._update_live()

    def on_best_updated(self, event: "BestUpdatedEvent") -> None:
        """Handle best updated event (no-op, leaderboard tracks results)."""
        del event  # Unused - leaderboard tracks results via on_allocation_completed

    def on_search_completed(self, event: SearchCompletedEvent) -> None:
        """Handle search completed event."""
        self._eval_count = event.total_evaluations

        # Print search completion summary
        self._print_search_complete(event)

        self._update_live()

    def on_eval_completed(self, eval_count: int) -> None:
        """Handle evaluation completed - throttled to avoid flicker."""
        self._eval_count = eval_count
        # Only update display every N evals
        if eval_count - self._last_eval_update >= self._EVAL_UPDATE_INTERVAL:
            self._last_eval_update = eval_count
            self._update_live()

    def on_bounds_updated(self, event: BoundsUpdatedEvent) -> None:
        """Handle bounds refinement during search.

        Updates the progress bar to reflect more accurate bounds as
        search progresses and actual eval counts replace estimates.
        """
        self._eval_lower_bound = event.new_lower
        self._eval_upper_bound = event.new_upper
        self._update_live()
