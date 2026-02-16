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
    SurrogateRoundCompletedEvent,
    SurrogateRoundStartedEvent,
    SwapBeamIterationEvent,
    SwapBeamIterationStartedEvent,
    SwapEvaluatedEvent,
)
from place_layers.search_strategy.progress.formatter import PlacementFormatterMixin

if TYPE_CHECKING:
    from types import TracebackType

    from place_layers.search_strategy.progress.events import (
        BestUpdatedEvent,
        SurrogateDatasetLoadedEvent,
    )


class LeaderboardEntry(BaseModel):
    """An entry in the results leaderboard."""

    score: float
    cost: float
    allocation: dict[str, int]  # e.g., {"FA": 2, "SWA": 1, "GDN": 7}
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


class SwapBeamIterationData(BaseModel):
    """Data for a completed swap-beam iteration."""

    iteration: int
    max_iterations: int
    num_tried: int
    best_score: float | None
    improved: bool
    converged: bool


class AnnealingStepSample(BaseModel):
    """Sampled annealing step data for history display."""

    step: int
    temperature: float
    best_score: float
    accepted: bool
    delta: float


class RichDisplay(PlacementFormatterMixin):
    """Rich Live display for search progress.

    Uses the observer pattern to receive events from the search strategy
    and updates the display accordingly. Thread-safe through Rich's Live.

    The display uses a hybrid approach:
    - Completed steps are printed permanently to the terminal
    - Current state is shown in a live display at the bottom

    Usage:
        with RichDisplay() as display:
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
        # Mixin attributes (set in on_search_started)
        self._mixer_names: tuple[str, ...] = ()
        self._num_layers: int = 0

        # State from events
        self._mode: str = ""
        self._constraint_value: float = 0.0
        self._min_cost: float = 0.0
        self._max_cost: float = 0.0
        self._max_allocations: int | None = 0

        self._allocation_index: int = 0
        self._current_allocation: dict[str, int] = {}
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

        # Swap-beam search state
        self._swap_beam_iterations: list[SwapBeamIterationData] = []
        self._swap_beam_current_iteration: int = 0
        self._swap_beam_max_iterations: int = 0
        self._swap_beam_best_score: float | None = None
        self._swap_beam_eval_count: int = 0
        self._swap_beam_total_evals: int = 0

        # Annealing search state
        self._annealing_steps: list[AnnealingStepSample] = []
        self._annealing_current_step: int = 0
        self._annealing_max_steps: int = 0
        self._annealing_temperature: float = 0.0
        self._annealing_acceptance_count: int = 0
        self._annealing_best_score: float | None = None
        self._annealing_restart_index: int = 0
        self._annealing_num_restarts: int = 1
        self._annealing_last_k: int = 0
        self._annealing_reheat_count: int = 0

        # Track which search strategy is active (for display selection)
        self._active_strategy: str | None = None  # "hierarchical", "swap_beam", "annealing"

    def __enter__(self) -> "RichDisplay":
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
                title=f"[bold yellow]Allocation {self._allocation_index}/{self._max_allocations or '?'}[/bold yellow]",
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
        """Build content for allocation panel: phases with nested steps.

        Detects the active strategy and renders the appropriate visualization:
        - Hierarchical: Phase panels with nested step panels
        - Swap-beam: Iteration history and current progress
        - Annealing: Temperature decay and acceptance tracking
        """
        components: list[Text | Panel] = []

        # Allocation summary line
        alloc_str = self._format_allocation(self._current_allocation)
        components.append(
            Text.from_markup(f"{alloc_str}  [dim]cost={self._current_allocation_cost:.4f}[/dim]")
        )

        # Render based on active strategy
        if self._active_strategy == "swap_beam":
            # Swap-beam visualization
            swap_panel = self._build_swap_beam_panel()
            if swap_panel:
                components.append(swap_panel)
        elif self._active_strategy == "annealing":
            # Annealing visualization
            annealing_panel = self._build_annealing_panel()
            if annealing_panel:
                components.append(annealing_panel)
        else:
            # Hierarchical beam search (default)
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

    def _build_swap_beam_panel(self) -> Panel | None:
        """Build swap-beam search visualization panel."""
        if not self._swap_beam_iterations and self._swap_beam_current_iteration == 0:
            return None

        lines: list[Text] = []

        # Current progress
        if self._swap_beam_current_iteration > 0:
            progress_text = (
                f"[dim]Iteration {self._swap_beam_current_iteration}/"
                f"{self._swap_beam_max_iterations}: "
                f"evaluating {self._swap_beam_eval_count}/{self._swap_beam_total_evals}[/dim]"
            )
            if self._swap_beam_best_score is not None:
                progress_text += f" [green]best: {self._swap_beam_best_score:.4g}[/green]"
            lines.append(Text.from_markup(progress_text))

        # Iteration history
        if self._swap_beam_iterations:
            lines.append(Text.from_markup("\n[bold]Iterations:[/bold]"))
            for iter_data in self._swap_beam_iterations[-5:]:  # Show last 5
                if iter_data.converged:
                    status = "[green]✓ converged[/green]"
                elif iter_data.improved:
                    status = "[green]✓ improved[/green]"
                else:
                    status = "[dim]no improvement[/dim]"

                score_str = f"{iter_data.best_score:.4g}" if iter_data.best_score else "N/A"
                lines.append(
                    Text.from_markup(
                        f"  {iter_data.iteration}. {status} → {score_str}"
                    )
                )

        return Panel(
            Group(*lines) if lines else Text(""),
            title="[bold cyan]Swap-Beam Search[/bold cyan]",
            border_style="cyan",
        )

    def _build_annealing_panel(self) -> Panel | None:
        """Build annealing search visualization panel."""
        if self._annealing_max_steps == 0:
            return None

        lines: list[Text] = []

        # Current step progress
        progress_pct = (self._annealing_current_step + 1) / self._annealing_max_steps * 100
        accept_rate = (
            self._annealing_acceptance_count / (self._annealing_current_step + 1) * 100
            if self._annealing_current_step > 0
            else 0
        )

        lines.append(
            Text.from_markup(
                f"[dim]Step {self._annealing_current_step + 1}/{self._annealing_max_steps}[/dim] | "
                f"T={self._annealing_temperature:.3g} | k={self._annealing_last_k} | "
                f"Accept: {accept_rate:.0f}%"
            )
        )

        # Temperature decay bar
        bar_width = 20
        filled = int(progress_pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(Text.from_markup(f"  {bar} ({progress_pct:.0f}%)"))

        # Best score
        if self._annealing_best_score is not None:
            lines.append(
                Text.from_markup(f"  [green]Best: {self._annealing_best_score:.4g}[/green]")
            )

        if self._annealing_reheat_count > 0:
            lines.append(
                Text.from_markup(f"  [yellow]Reheats: {self._annealing_reheat_count}[/yellow]")
            )

        if self._annealing_num_restarts > 1:
            title = (
                f"[bold magenta]Simulated Annealing[/bold magenta] "
                f"[dim]Restart {self._annealing_restart_index + 1}/"
                f"{self._annealing_num_restarts}[/dim]"
            )
        else:
            title = "[bold magenta]Simulated Annealing[/bold magenta]"

        return Panel(
            Group(*lines) if lines else Text(""),
            title=title,
            border_style="magenta",
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
            by_mixer = self._group_layers_by_mixer(entry.assignments)
            placement_str = self._format_placement(by_mixer)
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

        # Format assignments by mixer in canonical order
        by_mixer = self._group_layers_by_mixer(event.assignments)
        placement_str = self._format_placement(
            by_mixer, max_layers_per_mixer=8, escape_brackets=True
        )

        self._console.print(
            f"[bold cyan]✓ Allocation {event.allocation_index} complete[/bold cyan]: "
            f"{placement_str} → {event.score:.4g}"
        )

    def _format_expand_section(self, event: BeamGrowthStepEvent) -> str:
        """Format the expand section of a beam step as text."""
        from place_layers.search_strategy.progress.events import (
            BeamEntry,
            BeamNodeId,
            EvaluatedCandidate,
        )

        lines = []
        # Use unified B naming and show step context
        if event.step == 1:
            lines.append("Expand initial beam:")
        else:
            lines.append(f"Expand beam from step {event.step - 1}:")

        # Build lookup from parent ID to index and parent entry
        parent_id_to_idx: dict[BeamNodeId, int] = {}
        parent_id_to_entry: dict[BeamNodeId, BeamEntry] = {}
        for idx, parent in enumerate(event.parents):
            parent_id_to_idx[parent.id] = idx
            parent_id_to_entry[parent.id] = parent

        # Group tried candidates by parent_id
        tried_by_parent: dict[BeamNodeId, list[EvaluatedCandidate]] = {}
        for t in event.tried:
            if t.parent_id is None:
                continue
            if t.parent_id not in tried_by_parent:
                tried_by_parent[t.parent_id] = []
            tried_by_parent[t.parent_id].append(t)

        # Sort each parent's tried list by score descending
        for parent_id in tried_by_parent:
            tried_by_parent[parent_id].sort(key=lambda x: x.score, reverse=True)

        # Use best_candidate from event for highlighting
        best = event.best_candidate
        global_best_id = best.id if best else None

        # Show each parent and its tried candidates
        for parent_idx, parent in enumerate(event.parents):
            # Format parent's partial placement
            by_mixer = self._group_layers_by_mixer(parent.partial_placement)
            parent_placement_str = self._format_placement(
                by_mixer, max_layers_per_mixer=6, escape_brackets=True
            )
            context_str = f" ({parent_placement_str})" if parent_placement_str else ""

            lines.append(f"  B{parent_idx}{context_str}:")

            # Format tried candidates for this parent, one per line
            tried_list = tried_by_parent.get(parent.id, [])
            for t in tried_list[:5]:
                is_global_best = t.id == global_best_id
                diff = self._format_placement_diff(parent.partial_placement, t.placement)
                if is_global_best:
                    lines.append(f"    [green]B{parent_idx} + {diff} ({t.score:.4g})★[/green]")
                else:
                    lines.append(f"    B{parent_idx} + {diff} ({t.score:.4g})")
            if len(tried_list) > 5:
                lines.append(f"    [dim]...+{len(tried_list) - 5} more[/dim]")

        return "\n".join(lines)

    def _format_prune_section(self, event: BeamGrowthStepEvent) -> str:
        """Format the prune section of a beam step as text."""
        from place_layers.search_strategy.progress.events import BeamEntry, BeamNodeId

        lines = []
        beam_size = len(event.beam)
        lines.append(f"Prune to {beam_size}:")

        # Build lookup from parent ID to index
        parent_id_to_idx: dict[BeamNodeId, int] = {}
        parent_id_to_entry: dict[BeamNodeId, BeamEntry] = {}
        for idx, parent in enumerate(event.parents):
            parent_id_to_idx[parent.id] = idx
            parent_id_to_entry[parent.id] = parent

        for i, b in enumerate(event.beam):
            # Show lineage using diff from parent: B0 ← B0 + FA@6 (score)
            parent_idx = parent_id_to_idx.get(b.parent_id, 0) if b.parent_id else 0
            parent_entry = parent_id_to_entry.get(b.parent_id) if b.parent_id else None
            parent_partial = parent_entry.partial_placement if parent_entry else ()
            diff = self._format_placement_diff(parent_partial, b.partial_placement)

            if i == 0:
                lines.append(f"  [green]B{i} ← B{parent_idx} + {diff} ({b.score:.4g})★[/green]")
            else:
                lines.append(f"  B{i} ← B{parent_idx} + {diff} ({b.score:.4g})")

        # Use cut_candidates from event (already sorted by score descending)
        cut_candidates = event.cut_candidates

        if cut_candidates:
            # Show top 2-3 runners-up in compact format
            shown = cut_candidates[:3]
            cut_parts = [f"+{c.layer}→{c.score:.4g}" for c in shown]
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
                by_mixer = self._group_layers_by_mixer(entry.assignments)
                placement_str = self._format_placement(by_mixer)
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

        # Reset swap-beam state
        self._swap_beam_iterations = []
        self._swap_beam_current_iteration = 0
        self._swap_beam_max_iterations = 0
        self._swap_beam_best_score = None
        self._swap_beam_eval_count = 0
        self._swap_beam_total_evals = 0

        # Reset annealing state
        self._annealing_steps = []
        self._annealing_current_step = 0
        self._annealing_max_steps = 0
        self._annealing_temperature = 0.0
        self._annealing_acceptance_count = 0
        self._annealing_best_score = None
        self._annealing_restart_index = 0
        self._annealing_num_restarts = 1
        self._annealing_last_k = 0
        self._annealing_reheat_count = 0

        # Reset active strategy (will be set by first inner-loop event)
        self._active_strategy = None

        # Capture per-allocation bounds for progress tracking
        self._alloc_eval_start = self._eval_count
        self._alloc_expected = event.eval_expected

        self._update_live()

    def on_allocation_completed(self, event: AllocationCompletedEvent) -> None:
        """Handle allocation completed event."""
        # Print allocation completion summary
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

        # Clear phases for next allocation
        self._completed_phases = []
        self._current_phase = None
        self._current_step_content = ""

        self._update_live()

    # =========================================================================
    # Global best tracking event handlers (all strategies)
    # =========================================================================

    def on_best_updated(self, event: "BestUpdatedEvent") -> None:
        """Handle best updated event (no-op, leaderboard tracks results)."""
        del event  # Unused - leaderboard tracks results via on_allocation_completed

    def on_bounds_updated(self, event: BoundsUpdatedEvent) -> None:
        """Handle bounds refinement during search.

        Updates the progress bar to reflect more accurate bounds as
        search progresses and actual eval counts replace estimates.
        """
        self._eval_lower_bound = event.new_lower
        self._eval_upper_bound = event.new_upper
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

    # =========================================================================
    # Hierarchical beam search event handlers
    # =========================================================================

    def on_beam_phase_started(self, event: BeamPhaseStartedEvent) -> None:
        """Handle beam phase started event."""
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

    def on_beam_candidate_evaluated(self, event: BeamCandidateEvaluatedEvent) -> None:
        """Handle individual candidate result - stream into display."""
        # Accumulate result
        self._step_results.append({
            "layer": event.layer,
            "score": event.score,
            "id": event.id,
            "parent_id": event.parent_id,
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

    def on_beam_phase_completed(self, event: BeamPhaseCompletedEvent) -> None:
        """Handle beam phase completed event."""
        if self._current_phase is not None:
            # Use formatter for consistent formatting
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

    # =========================================================================
    # Swap-beam search event handlers
    # =========================================================================

    def on_swap_beam_iteration_started(
        self, event: SwapBeamIterationStartedEvent
    ) -> None:
        """Handle swap-beam iteration started - initialize iteration tracking."""
        self._active_strategy = "swap_beam"
        self._swap_beam_current_iteration = event.iteration
        self._swap_beam_max_iterations = event.max_iterations
        self._swap_beam_total_evals = event.num_candidates
        self._swap_beam_eval_count = 0

        # Set initial best score from beam if available
        if event.beam:
            self._swap_beam_best_score = event.beam[0].score

        self._update_live()

    def on_swap_evaluated(self, event: SwapEvaluatedEvent) -> None:
        """Handle swap evaluated - track progress and best score."""
        self._swap_beam_eval_count = event.eval_index

        # Track best score seen
        if self._swap_beam_best_score is None or event.score > self._swap_beam_best_score:
            self._swap_beam_best_score = event.score

        # Throttle updates
        if event.eval_index % self._EVAL_UPDATE_INTERVAL == 0:
            self._update_live()

    def on_swap_beam_iteration(self, event: SwapBeamIterationEvent) -> None:
        """Handle swap-beam iteration completed - add to history and print summary."""
        # Get best score for this iteration
        best_score = event.best_candidate.score if event.best_candidate else None

        # Add to history
        iter_data = SwapBeamIterationData(
            iteration=event.iteration,
            max_iterations=event.max_iterations,
            num_tried=len(event.tried),
            best_score=best_score,
            improved=event.improved,
            converged=event.converged,
        )
        self._swap_beam_iterations.append(iter_data)

        # Print permanent summary line
        if event.iteration > 0:  # Skip iteration 0 (initial eval)
            if event.converged:
                status = "[green]✓ converged[/green]"
            elif event.improved:
                status = "[green]✓ improved[/green]"
            else:
                status = "[dim]no improvement[/dim]"

            score_str = f"{best_score:.4g}" if best_score else "N/A"
            self._console.print(
                f"  [cyan]Iter {event.iteration}[/cyan]: {status} → {score_str}"
            )

        # Reset for next iteration
        self._swap_beam_eval_count = 0
        self._swap_beam_total_evals = 0

        self._update_live()

    # =========================================================================
    # Annealing search event handlers
    # =========================================================================

    def on_annealing_step(self, event: AnnealingStepEvent) -> None:
        """Handle annealing step - track temperature and acceptance."""
        self._active_strategy = "annealing"
        self._annealing_current_step = event.step
        self._annealing_max_steps = event.max_steps
        self._annealing_temperature = event.temperature
        self._annealing_best_score = event.best_score
        self._annealing_last_k = event.perturbation_size

        if event.accepted:
            self._annealing_acceptance_count += 1

        # Sample steps for history (every 10% of total)
        sample_interval = max(1, event.max_steps // 10)
        if event.step % sample_interval == 0:
            self._annealing_steps.append(
                AnnealingStepSample(
                    step=event.step,
                    temperature=event.temperature,
                    best_score=event.best_score,
                    accepted=event.accepted,
                    delta=event.delta,
                )
            )

        # Throttle display updates
        if event.step % self._EVAL_UPDATE_INTERVAL == 0:
            self._update_live()

    def on_annealing_restart(self, event: AnnealingRestartEvent) -> None:
        """Handle annealing restart - reset per-restart tracking."""
        self._annealing_restart_index = event.restart_index
        self._annealing_num_restarts = event.num_restarts
        self._annealing_max_steps = event.steps_budget
        # Reset per-restart counters
        self._annealing_steps = []
        self._annealing_acceptance_count = 0
        self._annealing_current_step = 0
        self._annealing_last_k = 0
        self._annealing_reheat_count = 0
        if event.restart_index > 0:
            self._console.print(
                f"  [magenta]Restart {event.restart_index + 1}/{event.num_restarts}[/magenta]"
            )
        self._update_live()

    def on_annealing_reheat(self, event: AnnealingReheatEvent) -> None:
        """Handle annealing reheat - print temperature change."""
        self._annealing_reheat_count += 1
        self._console.print(
            f"    [yellow]Reheat[/yellow]: T {event.old_temperature:.3g} → "
            f"{event.new_temperature:.3g} (accept: {event.acceptance_rate:.1%})"
        )

    def on_annealing_completed(self, event: AnnealingCompletedEvent) -> None:
        """Handle annealing completed - print summary."""
        parts = [f"{event.steps_taken} steps"]
        if event.restarts_completed > 1:
            parts.append(f"{event.restarts_completed} restarts (best: #{event.best_restart + 1})")
        if event.total_reheats > 0:
            parts.append(f"{event.total_reheats} reheats")
        detail = ", ".join(parts)

        self._console.print(
            f"  [magenta]Annealing[/magenta] ({event.termination_reason}): "
            f"{detail} → [green]{event.best_score:.4g}[/green]"
        )

        # Reset state for next allocation
        self._annealing_steps = []
        self._annealing_acceptance_count = 0
        self._annealing_current_step = 0
        self._annealing_max_steps = 0
        self._annealing_restart_index = 0
        self._annealing_num_restarts = 1
        self._annealing_last_k = 0
        self._annealing_reheat_count = 0

        self._update_live()

    # =========================================================================
    # Surrogate search event handlers
    # =========================================================================

    def on_surrogate_dataset_loaded(self, event: "SurrogateDatasetLoadedEvent") -> None:
        """Handle surrogate dataset loaded - print permanent context line."""
        self._console.print(
            f"  Surrogate: {event.num_samples} samples, exploration rho={event.exploration_rho:.2f}"
        )

    def on_surrogate_round_started(self, event: SurrogateRoundStartedEvent) -> None:
        """Handle surrogate round started - print permanent line."""
        self._console.print(
            f"  [cyan]ROUND {event.round}/{event.total_rounds}[/cyan]: "
            f"{event.num_candidates} candidates, "
            f"evaluating top {event.evals_planned} by EI"
        )

    def on_surrogate_round_completed(self, event: SurrogateRoundCompletedEvent) -> None:
        """Handle surrogate round completed - print permanent line."""
        self._console.print(
            f"  [cyan]ROUND {event.round}/{event.total_rounds} complete[/cyan]: "
            f"best={event.best_score:.4g}, pred_corr={event.prediction_correlation:.2f}, "
            f"rho={event.rho:.4f}, dataset={event.dataset_size}"
        )
