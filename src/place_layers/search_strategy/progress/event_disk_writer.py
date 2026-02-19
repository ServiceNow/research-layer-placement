"""Disk-based event writer for offline analysis.

This module provides an observer that writes all search events to a JSONL file
for offline analysis. Each line contains a timestamped event record with full
type information preserved through Pydantic's discriminated union support.

Usage:
    with EventDiskWriter(output_dir) as disk_writer:
        search_strategy = config.create_search_strategy(observers=[disk_writer])
        result = await search_strategy.search(problem)

    # Read events back:
    with open(output_dir / "search_events.jsonl") as f:
        for line in f:
            record = EventRecord.model_validate_json(line)
            match record.event:
                case SearchStartedEvent():
                    print(f"Search started: {record.event.mode}")
                case BeamGrowthStepEvent():
                    print(f"Beam step: {record.event.step}")
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from pydantic import BaseModel

from place_layers.search_strategy.progress.events import (
    AllocationCompletedEvent,
    AllocationPlanEvent,
    AllocationStartedEvent,
    AnnealingCompletedEvent,
    AnnealingReheatEvent,
    AnnealingRestartEvent,
    AnnealingStepEvent,
    AnySearchEvent,
    BeamCandidateEvaluatedEvent,
    BeamGrowthStepEvent,
    BeamGrowthStepStartedEvent,
    BeamPhaseCompletedEvent,
    BeamPhaseStartedEvent,
    BestUpdatedEvent,
    BoundsUpdatedEvent,
    SearchCompletedEvent,
    SearchStartedEvent,
    SurrogateCandidateEvaluatedEvent,
    SurrogateDatasetLoadedEvent,
    SurrogateRoundCompletedEvent,
    SurrogateRoundStartedEvent,
)

if TYPE_CHECKING:
    from types import TracebackType


class EventRecord(BaseModel):
    """Strongly-typed record wrapping an event with metadata.

    This wrapper adds a sequence number and timestamp to each event,
    while preserving full type information through the discriminated union.
    """

    model_config = {"frozen": True}

    seq: int  # Monotonically increasing sequence number
    timestamp: datetime  # When recorded (UTC)
    event: AnySearchEvent  # Discriminated union - Pydantic handles correctly


class EventDiskWriter:
    """Observer that writes events to JSONL for offline analysis.

    Context manager - opens file on enter, closes on exit.
    Implements all SearchObserver methods via shared _write_event().

    Output Format
    -------------

    One JSON object per line (JSONL):

    .. code-block:: json

        {"seq": 1, "timestamp": "...", "event": {"event_type": "search_started", ...}}
        {"seq": 2, "timestamp": "...", "event": {"event_type": "allocation_started", ...}}

    The ``event_type`` discriminator field is automatically included by Pydantic,
    enabling correct deserialization back to the original event types.

    File Size Estimates
    -------------------

    Event counts scale primarily with evaluations (see ``events.py`` for formula).
    Typical event sizes:

    - ``candidate_evaluated``: ~200 bytes (93-99% of events)
    - ``beam_growth_step``: ~500-2000 bytes (depends on beam width)
    - Other events: ~100-400 bytes

    Example: 10 allocations × 8000 evaluations ≈ 80,000 events ≈ 16-20 MB

    For long-running searches, consider:

    - Filtering events when reading (skip ``candidate_evaluated`` if not needed)
    - Streaming processing rather than loading entire file
    - Compressing output: ``gzip search_events.jsonl``

    Processing Tips
    ---------------

    Filter to structural events only (skip per-evaluation noise)::

        for line in open("search_events.jsonl"):
            record = EventRecord.model_validate_json(line)
            if record.event.event_type != "candidate_evaluated":
                process(record)

    Extract beam search lineage::

        for line in open("search_events.jsonl"):
            record = EventRecord.model_validate_json(line)
            if record.event.event_type == "beam_growth_step":
                for entry in record.event.beam:
                    print(f"Step {record.event.step}: id={entry.id} parent_id={entry.parent_id}")
    """

    def __init__(
        self,
        output_dir: Path,
        filename: str = "search_events.jsonl",
    ) -> None:
        """Initialize the event writer.

        Args:
            output_dir: Directory where the JSONL file will be written.
            filename: Name of the output file (default: search_events.jsonl).
        """
        self._output_path = output_dir / filename
        self._file: TextIO | None = None
        self._seq = 0

    def __enter__(self) -> "EventDiskWriter":
        """Open the output file for writing."""
        # Ensure parent directory exists
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._output_path.open("a")
        self._seq = 0
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Close the output file."""
        del exc_type, exc_val, exc_tb  # Unused
        if self._file is not None:
            self._file.close()
            self._file = None

    def _write_event(self, event: AnySearchEvent) -> None:
        """Write an event record to the JSONL file.

        Args:
            event: The search event to record.
        """
        if self._file is None:
            return

        self._seq += 1
        record = EventRecord(
            seq=self._seq,
            timestamp=datetime.now(timezone.utc),
            event=event,
        )
        self._file.write(record.model_dump_json() + "\n")
        self._file.flush()  # Ensure durability on crash

    # =========================================================================
    # Common/outer-loop event handlers (all strategies)
    # =========================================================================

    def on_search_started(self, event: SearchStartedEvent) -> None:
        """Handle search started event."""
        self._write_event(event)

    def on_allocation_plan(self, event: AllocationPlanEvent) -> None:
        """Handle allocation plan event."""
        self._write_event(event)

    def on_allocation_started(self, event: AllocationStartedEvent) -> None:
        """Handle allocation started event."""
        self._write_event(event)

    def on_allocation_completed(self, event: AllocationCompletedEvent) -> None:
        """Handle allocation completed event."""
        self._write_event(event)

    def on_best_updated(self, event: BestUpdatedEvent) -> None:
        """Handle best updated event."""
        self._write_event(event)

    def on_bounds_updated(self, event: BoundsUpdatedEvent) -> None:
        """Handle bounds updated event."""
        self._write_event(event)

    def on_search_completed(self, event: SearchCompletedEvent) -> None:
        """Handle search completed event."""
        self._write_event(event)

    def on_eval_completed(self, eval_count: int) -> None:
        """Handle evaluation completed - skip writing.

        This is a throttled counter event; detailed info is already
        captured in BeamCandidateEvaluatedEvent.
        """
        del eval_count  # Not written to disk

    # =========================================================================
    # Hierarchical beam search event handlers
    # =========================================================================

    def on_beam_phase_started(self, event: BeamPhaseStartedEvent) -> None:
        """Handle beam phase started event."""
        self._write_event(event)

    def on_beam_phase_completed(self, event: BeamPhaseCompletedEvent) -> None:
        """Handle beam phase completed event."""
        self._write_event(event)

    def on_beam_growth_step_started(self, event: BeamGrowthStepStartedEvent) -> None:
        """Handle beam growth step started event."""
        self._write_event(event)

    def on_beam_growth_step(self, event: BeamGrowthStepEvent) -> None:
        """Handle beam growth step event."""
        self._write_event(event)

    def on_beam_candidate_evaluated(self, event: BeamCandidateEvaluatedEvent) -> None:
        """Handle beam candidate evaluated event."""
        self._write_event(event)

    # =========================================================================
    # Annealing search event handlers
    # =========================================================================

    def on_annealing_step(self, event: AnnealingStepEvent) -> None:
        """Handle annealing step event."""
        self._write_event(event)

    def on_annealing_completed(self, event: AnnealingCompletedEvent) -> None:
        """Handle annealing completed event."""
        self._write_event(event)

    def on_annealing_reheat(self, event: AnnealingReheatEvent) -> None:
        """Handle annealing reheat event."""
        self._write_event(event)

    def on_annealing_restart(self, event: AnnealingRestartEvent) -> None:
        """Handle annealing restart event."""
        self._write_event(event)

    # =========================================================================
    # Surrogate search event handlers
    # =========================================================================

    def on_surrogate_dataset_loaded(self, event: SurrogateDatasetLoadedEvent) -> None:
        """Handle surrogate dataset loaded event."""
        self._write_event(event)

    def on_surrogate_round_started(self, event: SurrogateRoundStartedEvent) -> None:
        """Handle surrogate round started event."""
        self._write_event(event)

    def on_surrogate_round_completed(self, event: SurrogateRoundCompletedEvent) -> None:
        """Handle surrogate round completed event."""
        self._write_event(event)

    def on_surrogate_candidate_evaluated(
        self, event: SurrogateCandidateEvaluatedEvent
    ) -> None:
        """Handle surrogate candidate evaluated event."""
        self._write_event(event)
