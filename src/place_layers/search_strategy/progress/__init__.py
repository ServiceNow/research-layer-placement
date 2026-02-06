"""Progress display for search strategies.

This module provides an observer-based progress display system for cost-constrained
search strategies. The key components are:

- Events: Dataclasses representing key points in the search process
- SearchObserver: Protocol defining the observer interface
- RichSearchDisplay: Rich-based implementation for terminal display
- EventDiskWriter: JSONL-based implementation for offline analysis
- SearchProgressEstimator: Progress estimation and refinement

Progress estimators are created by search strategy configs that support them.
For example, HierarchicalConfig.create_progress_estimator() creates an estimator
for hierarchical beam search.

Usage:
    from place_layers.search_strategy.progress import RichSearchDisplay, EventDiskWriter

    with RichSearchDisplay() as display, EventDiskWriter(output_dir) as disk_writer:
        search_strategy = config.create_search_strategy(observers=[display, disk_writer])
        result = await search_strategy.search(problem)

With progress estimation (via search strategy config):
    from place_layers.search_strategy.progress import RichSearchDisplay, EventDiskWriter

    # Search strategy configs create their own estimators
    progress_estimator = search_strategy_config.create_progress_estimator(problem)

    with RichSearchDisplay() as display, EventDiskWriter(output_dir) as disk_writer:
        search_strategy = search_strategy_config.create_search_strategy(
            observers=[display, disk_writer],
            progress_estimator=progress_estimator,
        )
        result = await search_strategy.search(problem)
"""

from place_layers.search_strategy.progress.estimator import (
    AllocationProgressInfo,
    ProgressEstimate,
    SearchProgressEstimator,
)
from place_layers.search_strategy.progress.event_disk_writer import (
    EventDiskWriter,
    EventRecord,
)
from place_layers.search_strategy.progress.events import (
    AllocationCompletedEvent,
    AllocationPlanEvent,
    AllocationStartedEvent,
    AnySearchEvent,
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
from place_layers.search_strategy.progress.protocol import SearchObserver
from place_layers.search_strategy.progress.rich_display import RichSearchDisplay

__all__ = [
    # Events
    "AllocationCompletedEvent",
    "AllocationPlanEvent",
    "AllocationStartedEvent",
    "AnySearchEvent",
    "BeamGrowthStepEvent",
    "BeamGrowthStepStartedEvent",
    "BestUpdatedEvent",
    "BoundsUpdatedEvent",
    "CandidateEvaluatedEvent",
    "PhaseCompletedEvent",
    "PhaseStartedEvent",
    "SearchCompletedEvent",
    "SearchStartedEvent",
    # Protocol
    "SearchObserver",
    # Display
    "RichSearchDisplay",
    # Disk writer
    "EventDiskWriter",
    "EventRecord",
    # Progress estimation
    "AllocationProgressInfo",
    "ProgressEstimate",
    "SearchProgressEstimator",
]
