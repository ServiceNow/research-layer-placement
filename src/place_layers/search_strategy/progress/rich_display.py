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

import time
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
    BestUpdatedEvent,
    BoundsUpdatedEvent,
    CandidateEvaluatedEvent,
    PhaseCompletedEvent,
    PhaseStartedEvent,
    SearchCompletedEvent,
    SearchStartedEvent,
)

if TYPE_CHECKING:
    from types import TracebackType


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
        self._start_time: float | None = None

        # State from events
        self._mode: str = ""
        self._constraint_value: float = 0.0
        self._min_cost: float = 0.0
        self._max_cost: float = 0.0
        self._num_layers: int = 0
        self._mixer_names: tuple[str, ...] = ()
        self._max_allocations: int = 0
        self._total_allocations: int = 0

        self._allocation_index: int = 0
        self._current_allocation: tuple[int, ...] = ()
        self._current_allocation_cost: float = 0.0

        self._phase_index: int = 0
        self._total_phases: int = 0
        self._phase_mixer: str = ""
        self._phase_budget: int = 0

        self._beam_step: int = 0
        self._beam_total_steps: int = 0
        self._current_beam: tuple[dict, ...] = ()
        self._current_tried: tuple[dict, ...] = ()
        self._current_parents: tuple[dict, ...] = ()

        self._best_score: float | None = None
        self._best_cost: float | None = None
        self._best_allocation: tuple[int, ...] | None = None
        self._best_assignments: tuple[str, ...] | None = None

        self._eval_count: int = 0
        self._last_eval_update: int = 0

        # Overall eval count bounds (from BoundsUpdatedEvent)
        self._eval_lower_bound: int | None = None
        self._eval_upper_bound: int | None = None
        self._eval_expected: int | None = None

        # Per-allocation progress tracking
        self._alloc_eval_start: int = 0  # eval_count when allocation started
        self._alloc_lower: int | None = None
        self._alloc_upper: int | None = None
        self._alloc_expected: int | None = None

        # Accumulated state for nested panel rendering
        self._completed_phases: list[PhaseData] = []
        self._current_phase: PhaseData | None = None
        self._current_step_content: str = ""  # In-progress step text

        # Track current step for streaming candidate results
        self._step_results: list[dict] = []  # Accumulated results during step
        self._step_num_candidates: int = 0
        self._current_step: int = 0
        self._current_total_steps: int = 0

    def __enter__(self) -> "RichSearchDisplay":
        """Start the Live display."""
        self._start_time = time.time()
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

        # Best found panel (compact)
        if self._best_score is not None:
            components.append(self._render_best_compact())

        # Progress bar
        components.append(self._render_progress())

        return Group(*components)

    def _build_allocation_content(self) -> Group:
        """Build content for allocation panel: phases with nested steps."""
        components: list[Text | Panel] = []

        # Allocation summary line
        alloc_parts = []
        for name, count in zip(self._mixer_names, self._current_allocation):
            alloc_parts.append(f"[cyan]{name}[/cyan]:{count}")
        alloc_str = "  ".join(alloc_parts)
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

    def _render_best_compact(self) -> Panel:
        """Render a compact best found panel."""
        content = Text()
        content.append(f"{self._best_score:.4f}", style="green bold")
        content.append(" | Cost: ")
        content.append(f"{self._best_cost:.4f}")
        if self._best_allocation:
            alloc_parts = []
            for name, count in zip(self._mixer_names, self._best_allocation):
                alloc_parts.append(f"{name}={count}")
            content.append(f" | {{{', '.join(alloc_parts)}}}")

        return Panel(content, title="[green]Best[/green]", border_style="green")

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

        # Format assignments by mixer in canonical order (mixer_names)
        # This ensures consistent display order: FA, SWA, GDN, KDA, etc.
        ordered_by_mixer = {
            name: event.assignments_by_mixer[name]
            for name in event.mixer_names
            if name in event.assignments_by_mixer
        }
        placement_str = self._format_fixed_assignments(ordered_by_mixer)

        self._console.print(
            f"[bold cyan]✓ Allocation {event.allocation_index} complete[/bold cyan]: "
            f"{placement_str} → {event.score:.4g}"
        )

    def _format_fixed_assignments(self, fixed_by_mixer: dict[str, tuple[int, ...]]) -> str:
        """Format fixed assignments compactly.

        Returns strings like:
        - Short list (≤8 layers): "FA@[3,7,12,0,15]"
        - Long list (>8 layers): "FA@[0,1,3,7,...,23]"
        - Multiple mixers: "FA@[3,7], SWA@[1,2,...,14]"
        """
        if not fixed_by_mixer:
            return ""

        parts = []
        for mixer, layers in fixed_by_mixer.items():
            if len(layers) <= 8:
                layers_str = ",".join(str(l) for l in layers)
            else:
                # First 4 + last 1
                layers_str = ",".join(str(l) for l in layers[:4]) + f",...,{layers[-1]}"
            parts.append(f"{mixer}@\\[{layers_str}]")

        return ", ".join(parts)

    def _format_prior_context(self, fixed: dict[int, str]) -> str:
        """Format fixed assignments from prior phases compactly.

        Args:
            fixed: Dict mapping layer_idx -> mixer_name for prior phase assignments.

        Returns:
            Compact string like "FA@[0,5,9]" or "FA@[0,5], SWA@[3,7]" for multiple mixers.
            Empty string if no prior assignments.
        """
        if not fixed:
            return ""

        # Group layers by mixer
        by_mixer: dict[str, list[int]] = {}
        for layer, mixer in fixed.items():
            by_mixer.setdefault(mixer, []).append(layer)

        # Sort layers within each mixer and format
        parts = []
        for mixer in sorted(by_mixer.keys()):
            layers = sorted(by_mixer[mixer])
            if len(layers) <= 6:
                layers_str = ",".join(str(l) for l in layers)
            else:
                # First 3 + last 1
                layers_str = ",".join(str(l) for l in layers[:3]) + f",...,{layers[-1]}"
            parts.append(f"{mixer}@\\[{layers_str}]")

        return ", ".join(parts)

    def _format_layers(self, layers: tuple[int, ...] | list[int], max_show: int = 10) -> str:
        """Format a list of layers compactly."""
        if len(layers) <= max_show:
            return ",".join(str(l) for l in layers)
        else:
            first_half = max_show // 2
            return ",".join(str(l) for l in layers[:first_half]) + f",...,{layers[-1]}"

    def _format_expand_section(self, event: BeamGrowthStepEvent) -> str:
        """Format the expand section of a beam step as text."""
        lines = []
        num_parents = len(event.parents)
        lines.append(f"Expand {num_parents} parent{'s' if num_parents != 1 else ''}:")

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
            if len(parent_layers) <= 6:
                layers_str = ",".join(str(l) for l in parent_layers)
            else:
                layers_str = (
                    ",".join(str(l) for l in parent_layers[:3])
                    + f",...,{parent_layers[-1]}"
                )

            # Format prior phase context (if any)
            prior_ctx = self._format_prior_context(parent.get("fixed", {}))

            # Format current phase assignment
            current_ctx = f"{event.mixer_name}@\\[{layers_str}]" if parent_layers else ""

            # Combine into single parenthesized string
            if prior_ctx and current_ctx:
                context_str = f" ([dim]{prior_ctx}[/dim], {current_ctx})"
            elif prior_ctx:
                context_str = f" ([dim]{prior_ctx}[/dim])"
            elif current_ctx:
                context_str = f" ({current_ctx})"
            else:
                context_str = ""

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

            lines.append(f"  P{parent_idx}{context_str}: {tried_str}")

        return "\n".join(lines)

    def _format_prune_section(self, event: BeamGrowthStepEvent) -> str:
        """Format the prune section of a beam step as text."""
        lines = []
        beam_size = len(event.beam)
        lines.append(f"Prune to {beam_size}:")

        for i, b in enumerate(event.beam[:3]):
            parent_idx = b.get("parent_idx", 0)
            added_layer = b.get("added_layer", "?")

            # Get parent's layers and append the added layer
            parent = event.parents[parent_idx] if parent_idx < len(event.parents) else {}
            parent_layers = list(parent.get("layers", []))
            if added_layer != "?":
                parent_layers.append(added_layer)

            # Format layers string
            if len(parent_layers) <= 6:
                layers_str = ",".join(str(l) for l in parent_layers)
            else:
                layers_str = (
                    ",".join(str(l) for l in parent_layers[:3])
                    + f",...,{parent_layers[-1]}"
                )

            # Format prior phase context (if any)
            prior_ctx = self._format_prior_context(parent.get("fixed", {}))

            # Format current phase assignment
            current_ctx = f"{event.mixer_name}@\\[{layers_str}]" if parent_layers else ""

            # Combine into single parenthesized string
            if prior_ctx and current_ctx:
                context_str = f" ([dim]{prior_ctx}[/dim], {current_ctx})"
            elif prior_ctx:
                context_str = f" ([dim]{prior_ctx}[/dim])"
            elif current_ctx:
                context_str = f" ({current_ctx})"
            else:
                context_str = ""

            if i == 0:
                lines.append(f"  [green]B{i}{context_str} → {b['score']:.4g}★[/green]")
            else:
                lines.append(f"  B{i}{context_str} → {b['score']:.4g}")

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

    def _print_best_update(self) -> None:
        """Print new best found as permanent output."""
        self._console.print(
            f"  [bold green]★ NEW BEST[/bold green]: {self._best_score:.4f} | "
            f"cost={self._best_cost:.4f}"
        )

    def _print_search_complete(self, event: SearchCompletedEvent) -> None:
        """Print search completion as a Rich panel with table."""
        # Summary table
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column(style="green" if event.best_score else "")

        summary.add_row("Best score", f"{event.best_score:.4f}")
        summary.add_row("Best cost", f"{event.best_cost:.4f}")
        summary.add_row("Evaluations", f"{event.total_evaluations:,}")
        summary.add_row("Allocations explored", str(event.allocations_explored))

        # Final placement table
        if self._best_assignments:
            # Group by mixer type
            mixer_layers: dict[str, list[int]] = {}
            for i, mixer in enumerate(self._best_assignments):
                if mixer not in mixer_layers:
                    mixer_layers[mixer] = []
                mixer_layers[mixer].append(i)

            placement_table = Table(
                show_header=True,
                header_style="bold",
                box=None,
                padding=(0, 1),
            )
            placement_table.add_column("Mixer", style="cyan")
            placement_table.add_column("Count", justify="right")
            placement_table.add_column("Layers")

            for mixer in self._mixer_names:
                if mixer in mixer_layers:
                    layers = mixer_layers[mixer]
                    if len(layers) <= 8:
                        layers_str = ",".join(str(l) for l in layers)
                    else:
                        layers_str = ",".join(str(l) for l in layers[:4])
                        layers_str += f",...,{layers[-1]}"
                    placement_table.add_row(mixer, str(len(layers)), f"[{layers_str}]")

            # Combine into group
            content = Group(
                summary,
                Text(),  # Spacer
                Text("Final Placement:", style="bold"),
                placement_table,
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
        self._total_allocations = event.total_allocations

        # Initialize bounds from event (may be None if not provided)
        self._eval_lower_bound = event.eval_count_lower_bound
        self._eval_upper_bound = event.eval_count_upper_bound
        self._eval_expected = event.eval_count_expected

        # Reset accumulated state for fresh search
        self._completed_phases = []
        self._current_phase = None
        self._current_step_content = ""

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
            alloc_dict = dict(zip(event.mixer_names, info.allocation, strict=True))
            alloc_str = "  ".join(f"{k}:{v}" for k, v in alloc_dict.items())
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

        # Reset phase state for new allocation
        self._phase_index = 0
        self._phase_mixer = ""
        self._phase_budget = 0
        self._beam_step = 0
        self._beam_total_steps = 0
        self._current_beam = ()
        self._current_tried = ()
        self._current_parents = ()

        # Reset accumulated state for new allocation
        self._completed_phases = []
        self._current_phase = None
        self._current_step_content = ""

        # Capture per-allocation bounds for progress tracking
        self._alloc_eval_start = self._eval_count
        self._alloc_lower = event.eval_lower_bound
        self._alloc_upper = event.eval_upper_bound
        self._alloc_expected = event.eval_expected

        self._update_live()

    def on_allocation_completed(self, event: AllocationCompletedEvent) -> None:
        """Handle allocation completed event."""
        # Print allocation completion summary
        self._print_allocation_complete(event)

        # Clear phases for next allocation
        self._completed_phases = []
        self._current_phase = None
        self._current_step_content = ""

        self._update_live()

    def on_phase_started(self, event: PhaseStartedEvent) -> None:
        """Handle phase started event."""
        self._phase_index = event.phase_index
        self._total_phases = event.total_phases
        self._phase_mixer = event.mixer_name
        self._phase_budget = event.budget
        self._beam_step = 0
        self._beam_total_steps = event.budget - 1 if event.budget > 1 else 0
        self._current_beam = ()
        self._current_tried = ()
        self._current_parents = ()

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
        self._step_num_candidates = event.num_candidates
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
        self._beam_step = event.step
        self._beam_total_steps = event.total_steps
        self._current_beam = event.beam
        self._current_tried = event.tried
        self._current_parents = event.parents

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
            # Format layers string
            layers_str = self._format_layers(event.final_layers)
            # Set completion text
            self._current_phase.completion_text = (
                f"[bold green]✓ Complete[/bold green]: "
                f"{event.mixer_name}@\\[{layers_str}] → {event.final_score:.4g}"
            )
            # Move to completed
            self._completed_phases.append(self._current_phase)
            self._current_phase = None

        self._update_live()

    def on_best_updated(self, event: BestUpdatedEvent) -> None:
        """Handle best updated event."""
        self._best_score = event.score
        self._best_cost = event.cost
        self._best_allocation = event.allocation
        self._best_assignments = event.assignments

        # Print new best
        self._print_best_update()

        self._update_live()

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
