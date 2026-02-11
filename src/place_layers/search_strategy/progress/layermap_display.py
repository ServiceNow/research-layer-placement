"""Layermap ASCII visualization of placement search.

This module provides a visual display showing layer placements as character grids
that animate during evaluation. Each layer position is shown as a character, with
color indicating mixer type and highlighting for the layer being evaluated.

Two character styles are available:
- blocks: Dense visual using solid (assigned) and dim (fallback) block characters
- letters: Explicit mixer names using first letter of each mixer

Supports three search strategies:

**Hierarchical beam search** (default):
    Allocation 1/10: FA:5 SWA:3 GDN:40  |  Phase 1/2: +FA  |  Step 3/5
    ────────────────────────────────────────────────────────────────────
    B0: ██░░██░░░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  -4.52
    B1: ██░░░░██░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  -4.38 ★
    B2: ░░██░░██░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  -4.61
             ▲ evaluating

**Swap-beam search**:
    Allocation 1/10: FA:5 SWA:3 GDN:40  |  Swap-Beam Iter 3/10
    ────────────────────────────────────────────────────────────────────
    B0: ██░░██░░░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.823 ★
    B1: ██░░░░██░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.816
        ↑     ↑──────────────────────────────────────────── swap(2,5)
    Evaluating 12/48 best: 0.830

**Simulated annealing**:
    Allocation 1/10: FA:5 SWA:3 GDN:40  |  Annealing Step 150/500
    ────────────────────────────────────────────────────────────────────
    Current: ██░░██░░░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.823
       Best: ██░░░░██░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.857 ★
    T=1.23  Accept: 45%

Usage:
    from place_layers.search_strategy.progress import LayermapDisplay, LayermapDisplayConfig

    config = LayermapDisplayConfig(char_style="blocks")
    with config.create_observer() as display:
        search_strategy = create_search_strategy(observers=[display])
        result = await search_strategy.search(problem)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
    AnnealingCompletedEvent,
    AnnealingReheatEvent,
    AnnealingRestartEvent,
    AnnealingStepEvent,
    BeamCandidateEvaluatedEvent,
    BeamGrowthStepEvent,
    BeamGrowthStepStartedEvent,
    BeamPhaseCompletedEvent,
    BeamPhaseStartedEvent,
    BoundsUpdatedEvent,
    SearchCompletedEvent,
    SearchStartedEvent,
    SwapBeamIterationEvent,
    SwapBeamIterationStartedEvent,
    SwapCandidate,
    SwapEvaluatedEvent,
)
from place_layers.search_strategy.progress.formatter import PlacementFormatterMixin

# Active strategy types for layermap display
ActiveStrategy = Literal["hierarchical", "swap_beam", "annealing"]

if TYPE_CHECKING:
    from types import TracebackType

    from place_layers.search_strategy.progress.config import LayermapDisplayConfig
    from place_layers.search_strategy.progress.events import BestUpdatedEvent


# Mixer color palette - visually distinct colors for up to 6 mixers
MIXER_COLORS = [
    "bright_cyan",  # First mixer (often FA)
    "bright_yellow",  # Second mixer (often SWA)
    "bright_magenta",  # Third mixer (often GDN)
    "bright_green",
    "bright_blue",
    "bright_red",
]


class BeamDisplayEntry(BaseModel):
    """A beam entry ready for display."""

    model_config = {"frozen": True}

    idx: int
    placement: tuple[str | None, ...]  # Per-layer mixer assignments
    score: float | None
    is_best: bool = False


class LeaderboardEntry(BaseModel):
    """An entry in the results leaderboard."""

    score: float
    cost: float
    allocation: dict[str, int]
    assignments: tuple[str, ...]


class LayermapDisplay(PlacementFormatterMixin):
    """Layermap ASCII visualization of placement search.

    Uses the observer pattern to receive events from the search strategy
    and displays layer placements as animated character grids.

    The display uses Rich Live for smooth animation, with each layer
    shown as a colored character indicating its mixer assignment.

    Usage:
        with LayermapDisplay(config) as display:
            search_strategy = create_search_strategy(observer=display)
            await search_strategy.search(problem)
    """

    # Throttle eval updates to avoid display flicker
    _EVAL_UPDATE_INTERVAL = 5

    def __init__(
        self,
        config: "LayermapDisplayConfig",
        console: Console | None = None,
    ) -> None:
        self._config = config
        self._console = console or Console()
        self._live: Live | None = None

        # Mixin attributes (set in on_search_started)
        self._mixer_names: tuple[str, ...] = ()
        self._num_layers: int = 0

        # Mixer to color mapping
        self._mixer_styles: dict[str, str] = {}
        # Mixer to letter mapping (for letters mode)
        self._mixer_letters: dict[str, str] = {}

        # Search state
        self._mode: str = ""
        self._constraint_value: float = 0.0
        self._min_cost: float = 0.0
        self._max_cost: float = 0.0
        self._max_allocations: int = 0

        # Current allocation state
        self._allocation_index: int = 0
        self._current_allocation: dict[str, int] = {}
        self._current_allocation_cost: float = 0.0

        # Current phase state
        self._phase_index: int = 0
        self._total_phases: int = 0
        self._phase_mixer: str = ""

        # Current step state
        self._current_step: int = 0
        self._total_steps: int = 0

        # Beam entries for display
        self._beam_entries: list[BeamDisplayEntry] = []
        self._evaluating_layer: int | None = None
        self._best_beam_idx: int = 0

        # Leaderboard
        self._leaderboard: list[LeaderboardEntry] = []

        # Progress tracking
        self._eval_count: int = 0
        self._last_eval_update: int = 0
        self._eval_expected: int | None = None
        self._alloc_eval_start: int = 0
        self._alloc_expected: int | None = None

        # Overall bounds
        self._eval_lower_bound: int | None = None
        self._eval_upper_bound: int | None = None

        # Step progress
        self._step_eval_index: int = 0
        self._step_total_evals: int = 0
        self._step_best_score: float | None = None

        # Active search strategy
        self._active_strategy: ActiveStrategy | None = None

        # Swap-beam state
        self._swap_beam_iteration: int = 0
        self._swap_beam_max_iterations: int = 0
        self._swap_beam_beam: list[SwapCandidate] = []
        self._swap_current_swap: tuple[int, int] | None = None
        self._swap_eval_index: int = 0
        self._swap_total_evals: int = 0
        self._swap_best_score: float | None = None

        # Annealing state
        self._annealing_step: int = 0
        self._annealing_max_steps: int = 0
        self._annealing_temperature: float = 0.0
        self._annealing_current_placement: tuple[str, ...] = ()
        self._annealing_current_score: float = 0.0
        self._annealing_best_placement: tuple[str, ...] = ()
        self._annealing_best_score: float = 0.0
        self._annealing_acceptance_count: int = 0
        self._annealing_restart_index: int = 0
        self._annealing_num_restarts: int = 1
        self._annealing_last_k: int = 0
        self._annealing_reheat_count: int = 0

    def __enter__(self) -> "LayermapDisplay":
        """Start the Live display."""
        self._live = Live(
            self._build_display(),
            console=self._console,
            refresh_per_second=8,  # Faster refresh for smooth animation
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
        """Build the complete display from current state."""
        components = []

        # Main layermap panel - strategy-specific content
        if self._allocation_index > 0:
            if self._active_strategy == "swap_beam" and self._swap_beam_beam:
                components.append(self._build_swap_beam_content())
            elif self._active_strategy == "annealing" and self._annealing_current_placement:
                components.append(self._build_annealing_content())
            elif self._beam_entries:  # hierarchical (default)
                components.append(self._build_layermap_content())

        # Leaderboard panel
        if self._leaderboard:
            components.append(self._render_leaderboard())

        # Progress bar
        components.append(self._render_progress())

        return Group(*components)

    def _build_layermap_content(self) -> Panel:
        """Build the main layermap visualization panel."""
        lines = []

        # Header line: Allocation | Phase | Step
        alloc_str = self._format_allocation(self._current_allocation)
        header_parts = [
            f"[bold cyan]Allocation {self._allocation_index}/{self._max_allocations}[/bold cyan]: {alloc_str}",
        ]
        if self._phase_index > 0:
            header_parts.append(
                f"[bold green]Phase {self._phase_index}/{self._total_phases}[/bold green]: +{self._phase_mixer}"
            )
        if self._current_step > 0:
            header_parts.append(
                f"[bold yellow]Step {self._current_step}/{self._total_steps}[/bold yellow]"
            )
        header = Text.from_markup("  |  ".join(header_parts))
        lines.append(header)

        # Separator
        lines.append(Text("─" * min(80, self._num_layers + 20), style="dim"))

        # Render each beam entry
        for entry in self._beam_entries:
            row = self._render_beam_entry(
                entry, self._evaluating_layer, entry.idx == self._best_beam_idx
            )
            lines.append(row)

        # Evaluating indicator
        if self._evaluating_layer is not None:
            indicator = self._render_eval_indicator()
            lines.append(indicator)

        # Step progress
        if self._step_total_evals > 0:
            progress_text = f"[dim]Evaluating {self._step_eval_index}/{self._step_total_evals}[/dim]"
            if self._step_best_score is not None:
                progress_text += f" [green]best: {self._step_best_score:.4g}[/green]"
            lines.append(Text.from_markup(progress_text))

        return Panel(
            Group(*lines),
            title="[bold]Layer Placement[/bold]",
            border_style="blue",
        )

    def _render_layer(
        self,
        mixer: str | None,
        is_evaluating: bool = False,
    ) -> Text:
        """Render a single layer position as a colored character.

        Args:
            mixer: Mixer name assigned to this layer, or None for fallback
            is_evaluating: Whether this layer is currently being evaluated

        Returns:
            Rich Text object for this layer position
        """
        if self._config.char_style == "blocks":
            # Block mode: solid for assigned, dim for fallback
            if mixer is not None:
                char = "█"
            else:
                char = "░"
        else:
            # Letters mode: first letter of mixer name, or dot for fallback
            if mixer is not None:
                char = self._mixer_letters.get(mixer, mixer[0])
            else:
                char = "·"

        # Get style based on mixer
        if mixer is not None:
            style = self._mixer_styles.get(mixer, "white")
        else:
            style = "dim"

        # Highlight evaluating layer with reverse video
        if is_evaluating:
            style = f"reverse {style}"

        return Text(char, style=style)

    def _render_beam_entry(
        self,
        entry: BeamDisplayEntry,
        eval_layer: int | None,
        is_best: bool,
    ) -> Text:
        """Render one beam entry as a row of layer characters.

        Args:
            entry: The beam entry to render
            eval_layer: The layer index being evaluated (for highlighting)
            is_best: Whether this is the current best entry

        Returns:
            Rich Text object for the complete row
        """
        result = Text()

        # Beam index label
        result.append(f"B{entry.idx}: ", style="bold" if is_best else "")

        # Render each layer position
        for layer_idx in range(self._num_layers):
            mixer = entry.placement[layer_idx] if layer_idx < len(entry.placement) else None
            is_eval = layer_idx == eval_layer
            result.append_text(self._render_layer(mixer, is_eval))

        # Score
        if entry.score is not None:
            result.append(f"  {entry.score:.3g}", style="green" if is_best else "")

        # Best marker
        if is_best:
            result.append(" ★", style="bold yellow")

        return result

    def _render_eval_indicator(self) -> Text:
        """Render the evaluating layer indicator arrow."""
        if self._evaluating_layer is None:
            return Text("")

        # Calculate position: "B0: " = 4 chars, then layer positions
        prefix_len = 4
        arrow_pos = prefix_len + self._evaluating_layer

        result = Text(" " * arrow_pos)
        result.append("▲", style="bold bright_white")
        result.append(" evaluating", style="dim")
        return result

    # =========================================================================
    # Swap-beam visualization methods
    # =========================================================================

    def _build_swap_beam_content(self) -> Panel:
        """Build the swap-beam search visualization panel."""
        lines = []

        # Header line: Allocation | Swap-Beam Iteration
        alloc_str = self._format_allocation(self._current_allocation)
        header_parts = [
            f"[bold cyan]Allocation {self._allocation_index}/{self._max_allocations}[/bold cyan]: {alloc_str}",
            f"[bold magenta]Swap-Beam[/bold magenta] Iter {self._swap_beam_iteration}/{self._swap_beam_max_iterations}",
        ]
        header = Text.from_markup("  |  ".join(header_parts))
        lines.append(header)

        # Separator
        lines.append(Text("─" * min(80, self._num_layers + 20), style="dim"))

        # Render each beam entry
        for i, candidate in enumerate(self._swap_beam_beam):
            is_best = i == 0
            row = self._render_swap_beam_entry(candidate, is_best)
            lines.append(row)

        # Swap indicator if active
        if self._swap_current_swap and self._swap_current_swap != (-1, -1):
            indicator = self._render_swap_indicator()
            lines.append(indicator)

        # Progress within iteration
        if self._swap_total_evals > 0:
            progress_text = f"[dim]Evaluating {self._swap_eval_index}/{self._swap_total_evals}[/dim]"
            if self._swap_best_score is not None:
                progress_text += f" [green]best: {self._swap_best_score:.4g}[/green]"
            lines.append(Text.from_markup(progress_text))

        return Panel(
            Group(*lines),
            title="[bold]Swap-Beam Search[/bold]",
            border_style="magenta",
        )

    def _render_swap_beam_entry(
        self,
        candidate: SwapCandidate,
        is_best: bool,
    ) -> Text:
        """Render one swap-beam candidate as a row of layer characters."""
        result = Text()

        # Beam index label
        result.append(f"B{candidate.id.seq}: ", style="bold" if is_best else "")

        # Render each layer position, highlighting swap positions
        for layer_idx, mixer in enumerate(candidate.assignments):
            is_swap_pos = (
                self._swap_current_swap is not None
                and layer_idx in self._swap_current_swap
            )
            result.append_text(self._render_layer(mixer, is_evaluating=is_swap_pos))

        # Score
        result.append(f"  {candidate.score:.3g}", style="green" if is_best else "")

        # Best marker
        if is_best:
            result.append(" ★", style="bold yellow")

        return result

    def _render_swap_indicator(self) -> Text:
        """Render the swap indicator showing which layers are being swapped."""
        if not self._swap_current_swap or self._swap_current_swap == (-1, -1):
            return Text("")

        layer_a, layer_b = self._swap_current_swap
        min_layer = min(layer_a, layer_b)
        max_layer = max(layer_a, layer_b)

        # Build indicator line: arrows at swap positions connected by dashes
        prefix_len = 4  # "B0: "
        result = Text(" " * prefix_len)

        for i in range(self._num_layers):
            if i == min_layer:
                result.append("↑", style="bold bright_white")
            elif i == max_layer:
                result.append("↑", style="bold bright_white")
            elif min_layer < i < max_layer:
                result.append("─", style="dim")
            else:
                result.append(" ")

        result.append(f" swap({layer_a},{layer_b})", style="dim")
        return result

    # =========================================================================
    # Annealing visualization methods
    # =========================================================================

    def _build_annealing_content(self) -> Panel:
        """Build the annealing search visualization panel."""
        lines = []

        # Header line: Allocation | Annealing Step (with optional restart info)
        alloc_str = self._format_allocation(self._current_allocation)
        restart_str = ""
        if self._annealing_num_restarts > 1:
            restart_str = f"  [dim]Restart {self._annealing_restart_index + 1}/{self._annealing_num_restarts}[/dim]"
        header_parts = [
            f"[bold cyan]Allocation {self._allocation_index}/{self._max_allocations}[/bold cyan]: {alloc_str}",
            f"[bold yellow]Annealing[/bold yellow] Step {self._annealing_step + 1}/{self._annealing_max_steps}{restart_str}",
        ]
        header = Text.from_markup("  |  ".join(header_parts))
        lines.append(header)

        # Separator
        lines.append(Text("─" * min(80, self._num_layers + 20), style="dim"))

        # Current placement row
        current_row = self._render_annealing_placement_row(
            self._annealing_current_placement,
            self._annealing_current_score,
            label="Current",
            is_best=False,
        )
        lines.append(current_row)

        # Best placement row
        best_row = self._render_annealing_placement_row(
            self._annealing_best_placement,
            self._annealing_best_score,
            label="Best",
            is_best=True,
        )
        lines.append(best_row)

        # Temperature and acceptance info
        if self._annealing_step > 0:
            accept_rate = self._annealing_acceptance_count / self._annealing_step * 100
        else:
            accept_rate = 0
        temp_text = f"T={self._annealing_temperature:.3g}  k={self._annealing_last_k}  Accept: {accept_rate:.0f}%"
        if self._annealing_reheat_count > 0:
            temp_text += f"  [yellow]Reheats: {self._annealing_reheat_count}[/yellow]"
        lines.append(Text.from_markup(f"[dim]{temp_text}[/dim]"))

        return Panel(
            Group(*lines),
            title="[bold]Simulated Annealing[/bold]",
            border_style="yellow",
        )

    def _render_annealing_placement_row(
        self,
        placement: tuple[str, ...],
        score: float,
        label: str,
        is_best: bool,
    ) -> Text:
        """Render an annealing placement row (current or best)."""
        result = Text()

        # Label with padding for alignment
        label_style = "bold" if is_best else ""
        result.append(f"{label:>7}: ", style=label_style)

        # Render each layer position
        for mixer in placement:
            result.append_text(self._render_layer(mixer, is_evaluating=False))

        # Score
        result.append(f"  {score:.3g}", style="green" if is_best else "")

        # Best marker
        if is_best:
            result.append(" ★", style="bold yellow")

        return result

    def _render_leaderboard(self) -> Panel:
        """Render the unified summary + leaderboard panel."""
        components = []

        # Summary stats
        if self._leaderboard:
            best = self._leaderboard[0]
            summary = Table.grid(padding=(0, 2))
            summary.add_column(style="bold")
            summary.add_column(style="green")

            summary.add_row("Best score", f"{best.score:.4g}")
            summary.add_row("Best cost", f"{best.cost:.4g}")
            summary.add_row("Evaluations", f"{self._eval_count:,}")
            summary.add_row("Allocations explored", str(self._allocation_index))

            components.append(summary)
            components.append(Text())  # Spacer

        # Leaderboard table
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", justify="right", style="dim")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Cost", justify="right")
        table.add_column("Allocation")
        table.add_column("Placement")

        for i, entry in enumerate(self._leaderboard[:5], 1):
            alloc_str = self._format_allocation(entry.allocation)
            viz = Text()
            for mixer in entry.assignments:
                viz.append_text(self._render_layer(mixer, is_evaluating=False))
            table.add_row(
                str(i),
                f"{entry.score:.4g}",
                f"{entry.cost:.4g}",
                alloc_str,
                viz,
            )

        components.append(table)

        return Panel(Group(*components), title="[bold]Leaderboard[/bold]", border_style="green")

    def _render_progress(self) -> Group:
        """Render the dual progress bars (overall and per-allocation)."""
        components = []

        # Overall progress bar
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

        # Per-allocation progress bar
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

    def _init_mixer_mappings(self) -> None:
        """Initialize mixer to color and letter mappings."""
        self._mixer_styles = {}
        self._mixer_letters = {}

        for i, mixer in enumerate(self._mixer_names):
            # Assign color from palette (cycling if more mixers than colors)
            self._mixer_styles[mixer] = MIXER_COLORS[i % len(MIXER_COLORS)]
            # Use first letter of mixer name for letters mode
            self._mixer_letters[mixer] = mixer[0].upper()

    # =========================================================================
    # Print methods for permanent output
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

        # Mixer legend with colors
        mixer_parts = []
        for mixer in self._mixer_names:
            color = self._mixer_styles.get(mixer, "white")
            if self._config.char_style == "blocks":
                mixer_parts.append(f"[{color}]█[/{color}]={mixer}")
            else:
                letter = self._mixer_letters.get(mixer, mixer[0])
                mixer_parts.append(f"[{color}]{letter}[/{color}]={mixer}")
        table.add_row("Mixers", " ".join(mixer_parts))

        if self._eval_expected is not None:
            table.add_row("Expected evals", f"~{self._eval_expected:,}")

        panel = Panel(
            table,
            title="[bold]Layer Placement Optimization (Layermap View)[/bold]",
            border_style="blue",
        )
        self._console.print()
        self._console.print(panel)

    def _print_allocation_complete(self, event: AllocationCompletedEvent) -> None:
        """Print allocation completion summary with layermap visualization."""
        if event.assignments is None or event.score is None:
            self._console.print("[dim]  No valid placement found for this allocation[/dim]")
            return

        # Build a single-line layermap visualization of the final placement
        viz = Text("  ")
        for mixer in event.assignments:
            viz.append_text(self._render_layer(mixer, is_evaluating=False))

        viz.append(f"  → {event.score:.4g}", style="bold green")

        self._console.print(
            f"[bold cyan]✓ Allocation {event.allocation_index} complete[/bold cyan]"
        )
        self._console.print(viz)

    def _print_search_complete(self, event: SearchCompletedEvent) -> None:
        """Print search completion summary."""
        # Update final counts from event
        self._eval_count = event.total_evaluations
        self._allocation_index = event.allocations_explored

        # Reuse the unified leaderboard render
        panel = self._render_leaderboard()
        self._console.print()
        self._console.print(panel)

    # =========================================================================
    # Common/outer-loop event handlers (all strategies)
    # =========================================================================

    def on_search_started(self, event: SearchStartedEvent) -> None:
        """Handle search started event."""
        self._mixer_names = event.mixer_names
        self._num_layers = event.num_layers
        self._mode = event.mode
        self._constraint_value = event.constraint_value
        self._min_cost = event.min_cost
        self._max_cost = event.max_cost
        self._max_allocations = event.max_allocations

        # Initialize bounds
        self._eval_lower_bound = event.eval_count_lower_bound
        self._eval_upper_bound = event.eval_count_upper_bound
        self._eval_expected = event.eval_count_expected

        # Initialize mixer mappings
        self._init_mixer_mappings()

        # Reset state
        self._beam_entries = []
        self._leaderboard = []
        self._evaluating_layer = None

        # Print header
        self._print_search_header()

        self._update_live()

    def on_allocation_plan(self, event: AllocationPlanEvent) -> None:
        """Handle allocation plan event (show summary table)."""
        if len(event.allocation_info) > 20:
            return  # Too many to display

        table = Table(show_header=True, header_style="bold", padding=(0, 1))
        table.add_column("#", justify="right", style="dim")
        table.add_column("Allocation", no_wrap=True)
        table.add_column("Cost", justify="right")
        table.add_column("Expected evals", justify="right")

        for i, info in enumerate(event.allocation_info, 1):
            alloc_str = self._format_allocation(info.allocation)
            table.add_row(
                str(i),
                alloc_str,
                f"{info.cost:.4f}",
                f"~{int(info.expected):,}",
            )

        self._console.print()
        self._console.print(
            Panel(table, title="[bold]Allocation Plan[/bold]", border_style="yellow")
        )

    def on_allocation_started(self, event: AllocationStartedEvent) -> None:
        """Handle allocation started event."""
        self._allocation_index = event.allocation_index
        self._current_allocation = event.allocation
        self._current_allocation_cost = event.allocation_cost

        # Reset active strategy (will be set by first strategy-specific event)
        self._active_strategy = None

        # Reset hierarchical phase/step state
        self._phase_index = 0
        self._total_phases = 0
        self._current_step = 0
        self._total_steps = 0
        self._beam_entries = []
        self._evaluating_layer = None

        # Reset swap-beam state
        self._swap_beam_iteration = 0
        self._swap_beam_max_iterations = 0
        self._swap_beam_beam = []
        self._swap_current_swap = None
        self._swap_eval_index = 0
        self._swap_total_evals = 0
        self._swap_best_score = None

        # Reset annealing state
        self._annealing_step = 0
        self._annealing_max_steps = 0
        self._annealing_temperature = 0.0
        self._annealing_current_placement = ()
        self._annealing_current_score = 0.0
        self._annealing_best_placement = ()
        self._annealing_best_score = 0.0
        self._annealing_acceptance_count = 0
        self._annealing_restart_index = 0
        self._annealing_num_restarts = 1
        self._annealing_last_k = 0
        self._annealing_reheat_count = 0

        # Per-allocation progress
        self._alloc_eval_start = self._eval_count
        self._alloc_expected = event.eval_expected

        self._update_live()

    def on_allocation_completed(self, event: AllocationCompletedEvent) -> None:
        """Handle allocation completed event."""
        # Print completion summary
        self._print_allocation_complete(event)

        # Add ALL placements from this allocation to leaderboard
        for assignments, score in event.top_placements:
            entry = LeaderboardEntry(
                score=score,
                cost=event.allocation_cost,
                allocation=event.allocation,
                assignments=assignments,
            )
            self._leaderboard.append(entry)

        # Sort globally by score descending
        self._leaderboard.sort(key=lambda e: e.score, reverse=True)

        # Trim to bound memory (keep top 50)
        self._leaderboard = self._leaderboard[:50]

        # Clear beam for next allocation
        self._beam_entries = []
        self._evaluating_layer = None

        self._update_live()

    # =========================================================================
    # Global best tracking event handlers (all strategies)
    # =========================================================================

    def on_best_updated(self, event: "BestUpdatedEvent") -> None:
        """Handle best updated event (leaderboard tracks via allocation_completed)."""
        del event  # Unused

    def on_bounds_updated(self, event: BoundsUpdatedEvent) -> None:
        """Handle bounds refinement during search."""
        self._eval_lower_bound = event.new_lower
        self._eval_upper_bound = event.new_upper
        self._update_live()

    def on_search_completed(self, event: SearchCompletedEvent) -> None:
        """Handle search completed event."""
        self._eval_count = event.total_evaluations
        self._print_search_complete(event)
        self._update_live()

    def on_eval_completed(self, eval_count: int) -> None:
        """Handle evaluation completed - update overall progress."""
        self._eval_count = eval_count

    # =========================================================================
    # Hierarchical beam search event handlers
    # =========================================================================

    def on_beam_phase_started(self, event: BeamPhaseStartedEvent) -> None:
        """Handle beam phase started event."""
        self._active_strategy = "hierarchical"
        self._phase_index = event.phase_index + 1  # 1-indexed for display
        self._total_phases = event.total_phases
        self._phase_mixer = event.mixer_name
        self._current_step = 0
        self._total_steps = event.budget

        self._update_live()

    def on_beam_phase_completed(self, event: BeamPhaseCompletedEvent) -> None:
        """Handle beam phase completed event."""
        # Phase complete - will transition to next phase or allocation complete
        del event  # Info captured in allocation_completed
        self._update_live()

    def on_beam_growth_step_started(self, event: BeamGrowthStepStartedEvent) -> None:
        """Handle beam growth step started event."""
        self._current_step = event.step
        self._total_steps = event.total_steps
        self._step_eval_index = 0
        self._step_total_evals = event.num_candidates
        self._step_best_score = None

        # Initialize beam entries from parents
        self._beam_entries = []
        for idx, parent in enumerate(event.parents):
            self._beam_entries.append(
                BeamDisplayEntry(
                    idx=idx,
                    placement=parent.partial_placement,
                    score=parent.score,
                    is_best=(idx == 0),
                )
            )
        self._best_beam_idx = 0
        self._evaluating_layer = None

        self._update_live()

    def on_beam_candidate_evaluated(self, event: BeamCandidateEvaluatedEvent) -> None:
        """Handle beam candidate evaluated event - update highlighted layer."""
        self._step_eval_index = event.eval_index
        self._evaluating_layer = event.layer

        # Track best score in this step
        if self._step_best_score is None or event.score > self._step_best_score:
            self._step_best_score = event.score

        # Throttle display updates
        if event.eval_index - self._last_eval_update >= self._EVAL_UPDATE_INTERVAL:
            self._last_eval_update = event.eval_index
            self._update_live()

    def on_beam_growth_step(self, event: BeamGrowthStepEvent) -> None:
        """Handle beam growth step completed - update beam with pruned entries."""
        # Update beam entries with selected entries after pruning
        self._beam_entries = []
        for idx, entry in enumerate(event.beam):
            self._beam_entries.append(
                BeamDisplayEntry(
                    idx=idx,
                    placement=entry.partial_placement,
                    score=entry.score,
                    is_best=(idx == 0),
                )
            )
        self._best_beam_idx = 0

        # Clear evaluating state
        self._evaluating_layer = None
        self._step_eval_index = 0
        self._step_total_evals = 0
        self._step_best_score = None

        self._update_live()

    # =========================================================================
    # Swap-beam search event handlers
    # =========================================================================

    def on_swap_beam_iteration_started(
        self, event: SwapBeamIterationStartedEvent
    ) -> None:
        """Handle swap-beam iteration started event."""
        self._active_strategy = "swap_beam"
        self._swap_beam_iteration = event.iteration
        self._swap_beam_max_iterations = event.max_iterations
        self._swap_beam_beam = list(event.beam)
        self._swap_current_swap = None
        self._swap_eval_index = 0
        self._swap_total_evals = event.num_candidates
        self._swap_best_score = (
            event.beam[0].score if event.beam else None
        )

        self._update_live()

    def on_swap_evaluated(self, event: SwapEvaluatedEvent) -> None:
        """Handle swap evaluated event - update swap highlight and progress."""
        self._swap_current_swap = event.swap
        self._swap_eval_index = event.eval_index

        # Track best score in this iteration
        if self._swap_best_score is None or event.score > self._swap_best_score:
            self._swap_best_score = event.score

        # Throttle display updates
        if event.eval_index - self._last_eval_update >= self._EVAL_UPDATE_INTERVAL:
            self._last_eval_update = event.eval_index
            self._update_live()

    def on_swap_beam_iteration(self, event: SwapBeamIterationEvent) -> None:
        """Handle swap-beam iteration completed - update beam with results."""
        self._swap_beam_iteration = event.iteration
        self._swap_beam_beam = list(event.beam)
        self._swap_current_swap = None
        self._swap_eval_index = 0
        self._swap_total_evals = 0

        self._update_live()

    # =========================================================================
    # Annealing search event handlers
    # =========================================================================

    def on_annealing_step(self, event: AnnealingStepEvent) -> None:
        """Handle annealing step event - update current/best state."""
        self._active_strategy = "annealing"
        self._annealing_step = event.step
        self._annealing_max_steps = event.max_steps
        self._annealing_temperature = event.temperature
        self._annealing_last_k = event.perturbation_size
        self._annealing_current_placement = event.current_assignments
        self._annealing_current_score = event.current_score
        self._annealing_best_placement = event.best_assignments
        self._annealing_best_score = event.best_score

        if event.accepted:
            self._annealing_acceptance_count += 1

        # Throttle display updates (every 5 steps for annealing)
        if event.step % 5 == 0 or event.step == event.max_steps - 1:
            self._update_live()

    def on_annealing_restart(self, event: AnnealingRestartEvent) -> None:
        """Handle annealing restart - reset per-restart state."""
        self._active_strategy = "annealing"
        self._annealing_restart_index = event.restart_index
        self._annealing_num_restarts = event.num_restarts
        self._annealing_max_steps = event.steps_budget
        self._annealing_step = 0
        self._annealing_acceptance_count = 0
        self._annealing_reheat_count = 0
        self._annealing_last_k = 0
        self._annealing_current_placement = event.initial_assignments
        self._annealing_current_score = 0.0
        self._update_live()

    def on_annealing_reheat(self, event: AnnealingReheatEvent) -> None:
        """Handle annealing reheat - update temperature."""
        self._annealing_reheat_count += 1
        self._annealing_temperature = event.new_temperature
        self._update_live()

    def on_annealing_completed(self, event: AnnealingCompletedEvent) -> None:
        """Handle annealing completed event."""
        # Update final state
        self._annealing_best_placement = event.best_assignments
        self._annealing_best_score = event.best_score
        self._annealing_last_k = 0
        self._annealing_reheat_count = 0

        self._update_live()
