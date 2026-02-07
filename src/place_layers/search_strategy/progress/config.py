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
from typing import Annotated, Any, Generator, Literal

from pydantic import BaseModel, Field, field_validator


class BaseObserverConfig(BaseModel):
    """Base configuration for search observers."""

    model_config = {"frozen": True, "extra": "forbid"}

    observer_type: str = Field(description="Type identifier for observer")

    def create_observer(self) -> Any:
        """Smart constructor. No args - all config-driven.

        Returns an object that implements both SearchObserver protocol
        and the context manager protocol (__enter__/__exit__).

        Note: Return type is Any because the concrete types implement both
        SearchObserver (a Protocol) and context manager, but there's no
        way to express this intersection in Python's type system without
        creating a combined Protocol.
        """
        raise NotImplementedError


class RichDisplayConfig(BaseObserverConfig):
    """Rich terminal display with live panels.

    Provides an interactive terminal display using Rich's Live component
    with progress bars, beam visualization, and leaderboards.
    """

    observer_type: Literal["rich"] = "rich"

    def create_observer(self) -> Any:
        """Create RichSearchDisplay instance."""
        from place_layers.search_strategy.progress.rich_display import (
            RichSearchDisplay,
        )

        return RichSearchDisplay()


class PlainObserverConfig(BaseObserverConfig):
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

    def create_observer(self) -> Any:
        """Create PlainObserver instance."""
        from place_layers.search_strategy.progress.plain_observer import PlainObserver

        return PlainObserver(config=self)


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

    def create_observer(self) -> Any:
        """Create EventDiskWriter instance."""
        from place_layers.search_strategy.progress.event_disk_writer import (
            EventDiskWriter,
        )

        return EventDiskWriter(output_dir=self.output_dir, filename=self.filename)


# === Typed unions per semantic role ===

DisplayObserverConfig = Annotated[
    RichDisplayConfig | PlainObserverConfig,
    Field(discriminator="observer_type"),
]
"""Union of display observer configs (rich | plain).

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
    - display: single optional display observer (rich or plain)
    - log: sequence of log observers (disk, etc.)

    All observers are context managers by contract.

    Usage:
        with config.observers.create_observers() as observers:
            search_strategy = create_search_strategy(observers=observers)
    """

    model_config = {"frozen": True}

    display: DisplayObserverConfig | None = None
    log: list[LogObserverConfig] = []

    @field_validator("log", mode="before")
    @classmethod
    def convert_log_dict_to_list(cls, v: Any) -> Any:
        """Convert dict with numeric string keys to list.

        Hydra converts YAML lists with @package notation to dicts with
        keys like {'0': {...}, '1': {...}}. This validator handles that.
        """
        if isinstance(v, dict):
            # Sort by numeric key to maintain order
            try:
                sorted_items = sorted(v.items(), key=lambda x: int(x[0]))
                return [item[1] for item in sorted_items]
            except (ValueError, TypeError):
                # Keys aren't numeric, let Pydantic handle the error
                return v
        return v

    @contextmanager
    def create_observers(self) -> Generator[list[Any], None, None]:
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
            observers: list[Any] = []
            if self.display is not None:
                observers.append(stack.enter_context(self.display.create_observer()))
            for log_config in self.log:
                observers.append(stack.enter_context(log_config.create_observer()))
            yield observers
