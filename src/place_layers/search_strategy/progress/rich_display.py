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

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

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

if TYPE_CHECKING:
    from types import TracebackType


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
        self._mixer_names: list[str] = []
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
        self._current_beam: list[dict] = []
        self._current_tried: list[dict] = []
        self._current_base_layers: list[int] = []

        self._best_score: float | None = None
        self._best_cost: float | None = None
        self._best_allocation: tuple[int, ...] | None = None
        self._best_assignments: list[str] | None = None

        self._eval_count: int = 0
        self._last_eval_update: int = 0

        # Eval count bounds (from BoundsUpdatedEvent)
        self._eval_lower_bound: int | None = None
        self._eval_upper_bound: int | None = None

        # Track current phase tree for building up the visualization
        self._phase_tree: Tree | None = None

    def __enter__(self) -> "RichSearchDisplay":
        """Start the Live display."""
        self._start_time = time.time()
        self._live = Live(
            self._render_live(),
            console=self._console,
            refresh_per_second=4,
            transient=True,  # Live display disappears when updated
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
            self._live.update(self._render_live())

    def _render_live(self) -> Group:
        """Render the live display at the bottom of the terminal."""
        components = []

        # Current step being expanded
        if self._phase_mixer and self._beam_step > 0:
            components.append(self._render_current_step())

        # Best found panel (compact)
        if self._best_score is not None:
            components.append(self._render_best_compact())

        # Progress bar
        components.append(self._render_progress())

        return Group(*components)

    def _render_current_step(self) -> Panel:
        """Render the current step being expanded."""
        content = Text()

        # Show what we're expanding
        if self._current_base_layers:
            base_str = ",".join(str(l) for l in self._current_base_layers)
            content.append(f"Expanding [{base_str}]: ", style="bold")
        else:
            content.append("Initial scoring: ", style="bold")

        # Show top tried candidates
        if self._current_tried:
            tried_parts = []
            for i, t in enumerate(self._current_tried[:8]):
                marker = "[green]★[/green]" if i == 0 else ""
                tried_parts.append(f"@{t['layer']}→{t['score']:.1f}{marker}")
            content.append(" ".join(tried_parts))
            if len(self._current_tried) > 8:
                content.append(" ...")

        title = f"Step {self._beam_step}/{self._beam_total_steps}: {self._phase_mixer}"
        return Panel(content, title=title, border_style="cyan")

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

    def _render_progress(self) -> Progress:
        """Render the progress bar."""
        # Include percentage column only if we have a known upper bound
        if self._eval_upper_bound is not None:
            progress = Progress(
                SpinnerColumn(),
                TextColumn(f"Evals: {self._eval_count:,}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self._console,
                transient=False,
            )
            progress.add_task(
                "Searching...",
                total=self._eval_upper_bound,
                completed=self._eval_count,
            )
        else:
            progress = Progress(
                SpinnerColumn(),
                TextColumn(f"Evals: {self._eval_count:,}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self._console,
                transient=False,
            )
            progress.add_task("Searching...", total=None)
        return progress

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
        if self._eval_lower_bound is not None and self._eval_upper_bound is not None:
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

    def _print_allocation_header(self) -> None:
        """Print allocation header as a Rich panel."""
        # Format allocation with mixer names
        alloc_parts = []
        for name, count in zip(self._mixer_names, self._current_allocation):
            alloc_parts.append(f"[cyan]{name}[/cyan]:{count}")
        alloc_str = "  ".join(alloc_parts)

        content = Text.from_markup(
            f"{alloc_str}    [dim]cost={self._current_allocation_cost:.4f}[/dim]"
        )

        panel = Panel(
            content,
            title=f"[bold yellow]Allocation {self._allocation_index}/{self._max_allocations}[/bold yellow]",
            border_style="yellow",
        )
        self._console.print()
        self._console.print(panel)

    def _print_phase_header(self) -> None:
        """Start a new phase tree."""
        # Create a new tree for this phase
        self._phase_tree = Tree(
            f"[bold green]Phase {self._phase_index}/{self._total_phases}[/bold green]: "
            f"Place [cyan]{self._phase_mixer}[/cyan] (budget={self._phase_budget})"
        )

    def _print_beam_step(self, event: BeamGrowthStepEvent) -> None:
        """Add beam growth step to the phase tree."""
        if self._phase_tree is None:
            return

        # Format tried candidates
        tried_parts = []
        for i, t in enumerate(event.tried[:6]):
            if i == 0:
                tried_parts.append(f"[green]@{t['layer']}→{t['score']:.1f}★[/green]")
            else:
                tried_parts.append(f"@{t['layer']}→{t['score']:.1f}")
        tried_str = "  ".join(tried_parts)
        if len(event.tried) > 6:
            tried_str += "  [dim]...[/dim]"

        # Format beam state
        beam_parts = []
        for i, b in enumerate(event.beam[:3]):
            layers_str = ",".join(str(l) for l in b["layers"])
            if i == 0:
                beam_parts.append(f"[green]\\[{layers_str}]→{b['score']:.1f}★[/green]")
            else:
                beam_parts.append(f"\\[{layers_str}]→{b['score']:.1f}")
        beam_str = "  ".join(beam_parts)

        # Build step description
        if event.base_layers:
            base_str = ",".join(str(l) for l in event.base_layers)
            step_label = f"[bold]Step {event.step}[/bold]: Expand \\[{base_str}]"
        else:
            step_label = f"[bold]Step {event.step}[/bold]: Initial"

        # Add to tree
        step_branch = self._phase_tree.add(step_label)
        step_branch.add(f"Tried: {tried_str}")
        step_branch.add(f"Beam:  {beam_str}")

    def _print_phase_complete(self, event: PhaseCompletedEvent) -> None:
        """Complete and print the phase tree."""
        if self._phase_tree is None:
            return

        # Add completion summary
        if len(event.final_layers) <= 10:
            layers_str = ",".join(str(l) for l in event.final_layers)
        else:
            layers_str = ",".join(str(l) for l in event.final_layers[:5])
            layers_str += f",...,{event.final_layers[-1]}"

        self._phase_tree.add(
            f"[bold green]✓ Complete[/bold green]: "
            f"{event.mixer_name}@\\[{layers_str}] → {event.final_score:.1f}"
        )

        # Print the completed tree
        self._console.print(self._phase_tree)
        self._phase_tree = None

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

        # Print search header using Rich panel
        self._print_search_header()

        self._update_live()

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
        self._current_beam = []
        self._current_tried = []
        self._current_base_layers = []
        self._phase_tree = None

        # Print allocation header
        self._print_allocation_header()

        self._update_live()

    def on_phase_started(self, event: PhaseStartedEvent) -> None:
        """Handle phase started event."""
        self._phase_index = event.phase_index
        self._total_phases = event.total_phases
        self._phase_mixer = event.mixer_name
        self._phase_budget = event.budget
        self._beam_step = 0
        self._beam_total_steps = event.budget - 1 if event.budget > 1 else 0
        self._current_beam = []
        self._current_tried = []
        self._current_base_layers = []

        # Start a new phase tree
        self._print_phase_header()

        self._update_live()

    def on_beam_growth_step(self, event: BeamGrowthStepEvent) -> None:
        """Handle beam growth step event."""
        self._beam_step = event.step
        self._beam_total_steps = event.total_steps
        self._current_beam = event.beam
        self._current_tried = event.tried
        self._current_base_layers = event.base_layers

        # Add step to phase tree
        self._print_beam_step(event)

        self._update_live()

    def on_phase_completed(self, event: PhaseCompletedEvent) -> None:
        """Handle phase completed event."""
        # Complete and print the phase tree
        self._print_phase_complete(event)

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
