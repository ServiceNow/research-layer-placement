"""Configuration for search observers.

This module provides Pydantic configuration classes for search observers,
following the Hydra+Pydantic discriminated union pattern used elsewhere
in the codebase.

The key design is:
- Single Hydra `observer/` group with all observer configs
- Pydantic container class (`ObserversConfig`) with typed fields enforcing semantics
- Smart constructor on container is a context manager handling ExitStack internally

YAML structure:
    observers:
      display:
        observer_type: rich      # or "plain" for headless
      log:
        - observer_type: disk
          filename: search_events.jsonl

CLI usage:
    # Use default preset (rich + disk)
    python -m place_layers +experiments=optimize_mmlu

    # Use headless preset (plain + disk)
    python -m place_layers +experiments=optimize_mmlu observers=headless

    # Use silent preset (no display, disk only)
    python -m place_layers +experiments=optimize_mmlu observers=silent

Code usage:
    with config.observers.create_observers() as observers:
        search_strategy = config.create_search_strategy(observers=observers)
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, ClassVar, Generator, Literal, Protocol, Self

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from place_layers.search_strategy.progress.event_disk_writer import EventDiskWriter
    from place_layers.search_strategy.progress.layermap_display import LayermapDisplay
    from place_layers.search_strategy.progress.plain_display import PlainDisplay
    from place_layers.search_strategy.progress.protocol import SearchObserver
    from place_layers.search_strategy.progress.rich_display import RichDisplay


class ObserverContextManager(Protocol):
    """Protocol for observers that are also context managers.

    Observers must implement both the SearchObserver protocol and be usable
    as context managers for proper lifecycle management.
    """

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


class BaseObserverConfig(BaseModel):
    """Base configuration for search observers."""

    model_config = {"frozen": True, "extra": "forbid"}

    observer_type: str = Field(description="Type identifier for observer")

    # Subclasses must set this to their implementation class
    _impl_class: ClassVar[type[ObserverContextManager]]

    def create_observer(self) -> "SearchObserver":
        """Smart constructor. No args - all config-driven.

        Returns an object that implements the SearchObserver protocol
        and the context manager protocol (__enter__/__exit__).
        """
        raise NotImplementedError


class RichDisplayConfig(BaseObserverConfig):
    """Rich terminal display with live panels.

    Provides an interactive terminal display using Rich's Live component
    with progress bars, beam visualization, and leaderboards.
    """

    observer_type: Literal["rich"] = "rich"
    surrogate_top_k: int | None = Field(
        default=10,
        description="Max surrogate candidates to display per round (null = no limit)",
        ge=1,
    )

    def create_observer(self) -> "RichDisplay":
        """Create RichDisplay instance."""
        from place_layers.search_strategy.progress.rich_display import (
            RichDisplay,
        )

        return RichDisplay(surrogate_top_k=self.surrogate_top_k)


class LayermapDisplayConfig(BaseObserverConfig):
    """Layermap ASCII visualization of placement search.

    Displays layer placements as character grids with animation during
    evaluation. Each layer is shown as a character, with color indicating
    mixer type and highlighting for the layer being evaluated.

    Two character styles are available:
    - blocks: Dense visual using block characters (default)
    - letters: Explicit mixer names using first letter of each mixer

    Example output (blocks mode):
        Allocation 1/10: FA:5 SWA:3 GDN:40  |  Phase 1/2: +FA  |  Step 3/5
        ────────────────────────────────────────────────────────────────────
        B0: ██░░██░░░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  -4.52
        B1: ██░░░░██░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  -4.38 ★
        B2: ░░██░░██░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  -4.61
                 ▲ evaluating
    """

    observer_type: Literal["layermap"] = "layermap"
    char_style: Literal["blocks", "letters"] = Field(
        default="blocks",
        description="Character style: 'blocks' (solid/dim) or 'letters' (F/S/G)",
    )
    surrogate_top_k: int | None = Field(
        default=10,
        description="Max surrogate candidates to display per round (null = no limit)",
        ge=1,
    )

    def create_observer(self) -> "LayermapDisplay":
        """Create LayermapDisplay instance."""
        from place_layers.search_strategy.progress.layermap_display import LayermapDisplay

        return LayermapDisplay(config=self)


class PlainDisplayConfig(BaseObserverConfig):
    """Plain text output for headless jobs.

    Produces simple timestamped log lines without cursor manipulation,
    spinners, or progress bars. Suitable for CI/CD pipelines, batch jobs,
    and environments without rich terminal support.

    Output format (with phases/steps/leaderboard enabled):
        [14:30:22] SEARCH_STARTED: mode=max_performance, layers=48, allocations=10
        [14:30:22]   Budget <= 8.0000, cost bounds [4.0000, 20.0000]
        [14:30:22]   Mixers: FA, SWA, GDN | Expected evals: ~1,797
        [14:30:22] ALLOCATION #1/10: FA:12 SWA:8 GDN:28 (cost=8.0000)
        [14:30:23]   PHASE 1/2: Place FA (budget=12), filler=GDN
        [14:30:23]     Starting beam:
        [14:30:23]       B0: GDN@{0,1,2,3,...,47}
        [14:30:23]     Step 1/12: Evaluating 48 candidates...
        [14:30:23]     Step 1/12: [10/48] best B0+FA@6 (5.2)
        [14:30:23]     Step 1/12: [48/48] done, best B0+FA@6 (5.2)
        [14:30:23]     Step 1/12: Prune ->
        [14:30:23]       B0 <- B0+FA@6 (5.2)*
        [14:30:23]       B1 <- B0+FA@16 (4.8)
        [14:30:45]   PHASE 1/2 COMPLETE: FA@[0,2,4,6,...] -> 0.8234
        [14:30:45] ALLOCATION #1 COMPLETE: FA@[...], SWA@[...], GDN@[...] -> 0.8234
        [14:31:02] BEST_UPDATED: score=0.8567, cost=0.4500
        [14:35:00] SEARCH_COMPLETE: best=0.8567, evals=1234, allocations=10
        [14:35:00]   Leaderboard:
        [14:35:00]     1. 0.8567 (cost=0.45) FA:12 SWA:8 GDN:28
    """

    observer_type: Literal["plain"] = "plain"
    surrogate_top_k: int | None = Field(
        default=10,
        description="Max surrogate candidates to display per round (null = no limit)",
        ge=1,
    )
    show_allocation_start: bool = Field(
        default=True,
        description="Whether to print when allocations start",
    )
    show_allocation_complete: bool = Field(
        default=True,
        description="Whether to print when allocations complete",
    )
    show_best_updated: bool = Field(
        default=True,
        description="Whether to print when a new best is found",
    )
    show_phases: bool = Field(
        default=True,
        description="Whether to print phase start/complete events",
    )
    show_steps: bool = Field(
        default=True,
        description="Whether to print beam growth step events",
    )
    show_leaderboard: bool = Field(
        default=True,
        description="Whether to print leaderboard at search completion",
    )
    eval_progress_interval: int = Field(
        description="Print progress every N evaluations (0 = only at end)",
    )
    timestamp_format: str = Field(
        default="%H:%M:%S",
        description="strftime format for timestamps",
    )

    def create_observer(self) -> "PlainDisplay":
        """Create PlainDisplay instance."""
        from place_layers.search_strategy.progress.plain_display import PlainDisplay

        return PlainDisplay(config=self)


class DiskWriterConfig(BaseObserverConfig):
    """JSONL event writer for offline analysis.

    Writes all search events to a JSONL file, enabling post-hoc analysis
    and visualization of the search process.
    """

    observer_type: Literal["disk"] = "disk"
    output_dir: Path = Field(description="Output directory (typically from Hydra)")
    filename: str = Field(
        default="search_events.jsonl",
        description="Name of the output file",
    )

    def create_observer(self) -> "EventDiskWriter":
        """Create EventDiskWriter instance."""
        from place_layers.search_strategy.progress.event_disk_writer import (
            EventDiskWriter,
        )

        return EventDiskWriter(output_dir=self.output_dir, filename=self.filename)


# === Typed unions per semantic role ===

DisplayObserverConfig = Annotated[
    RichDisplayConfig | PlainDisplayConfig | LayermapDisplayConfig,
    Field(discriminator="observer_type"),
]
"""Union of display observer configs (rich | plain | layermap).

Display observers provide terminal output for interactive use.
Only one display observer should be active at a time.
"""

LogObserverConfig = Annotated[
    DiskWriterConfig,
    Field(discriminator="observer_type"),
]
"""Union of log observer configs (currently just disk).

Log observers write events for offline analysis.
Multiple log observers can be active simultaneously.
"""


# === Container class with context-managed smart constructor ===


class ObserversConfig(BaseModel):
    """Container for all observer configurations.

    Enforces semantics via typed fields:
    - display: single optional display observer (rich, plain, or layermap)
    - log: dict of named log observers (e.g. {"disk": DiskWriterConfig(...)})

    All observers are context managers by contract.

    Usage:
        with config.observers.create_observers() as observers:
            search_strategy = create_search_strategy(observers=observers)
    """

    model_config = {"frozen": True}

    display: DisplayObserverConfig | None = None
    log: dict[str, LogObserverConfig] = {}

    @contextmanager
    def create_observers(self) -> Generator[list["SearchObserver"], None, None]:
        """Context manager: creates all observers and manages their lifecycle.

        All observers are context managers by contract. This method handles
        the ExitStack internally, entering each observer context and ensuring
        proper cleanup on exit.

        Yields:
            List of instantiated observers ready for use. Each observer
            implements the SearchObserver protocol.

        Example:
            with config.observers.create_observers() as observers:
                search_strategy = config.create_search_strategy(
                    observers=observers,
                    progress_estimator=progress_estimator,
                )
                result = await search_strategy.search(problem)
        """
        with ExitStack() as stack:
            observers: list[SearchObserver] = []
            if self.display is not None:
                observers.append(stack.enter_context(self.display.create_observer()))
            for log_config in self.log.values():
                observers.append(stack.enter_context(log_config.create_observer()))
            yield observers
