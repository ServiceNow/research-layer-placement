"""Progress display for search strategies.

This module provides an observer-based progress display system for cost-constrained
search strategies. The key components are:

- Events: Dataclasses representing key points in the search process
- SearchObserver: Protocol defining the observer interface
- RichDisplay: Rich-based implementation for terminal display
- PlainDisplay: Simple timestamped output for headless jobs
- EventDiskWriter: JSONL-based implementation for offline analysis
- SearchProgressEstimator: Progress estimation and refinement
- ObserversConfig: Hydra+Pydantic configuration container for observers

Progress estimators are created by search strategy configs that support them.
For example, HierarchicalConfig.create_progress_estimator() creates an estimator
for hierarchical beam search.

Usage (config-driven):
    from place_layers.search_strategy.progress import ObserversConfig

    # ObserversConfig is typically populated from Hydra config
    with config.observers.create_observers() as observers:
        search_strategy = config.create_search_strategy(observers=observers)
        result = await search_strategy.search(problem)

Usage (direct):
    from place_layers.search_strategy.progress import RichDisplay, EventDiskWriter

    with RichDisplay() as display, EventDiskWriter(output_dir) as disk_writer:
        search_strategy = config.create_search_strategy(observers=[display, disk_writer])
        result = await search_strategy.search(problem)
"""

from place_layers.search_strategy.progress.config import (
    BaseObserverConfig,
    DiskWriterConfig,
    DisplayObserverConfig,
    LayermapDisplayConfig,
    LogObserverConfig,
    ObserversConfig,
    PlainDisplayConfig,
    RichDisplayConfig,
)
from place_layers.search_strategy.progress.estimator import (
    AllocationProgressInfo,
    ProgressEstimate,
    SearchProgressEstimator,
)
from place_layers.search_strategy.progress.event_disk_writer import (
    EventDiskWriter,
    EventRecord,
)
from place_layers.search_strategy.progress.formatter import PlacementFormatterMixin
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
from place_layers.search_strategy.progress.layermap_display import LayermapDisplay
from place_layers.search_strategy.progress.plain_display import PlainDisplay
from place_layers.search_strategy.progress.protocol import SearchObserver
from place_layers.search_strategy.progress.rich_display import RichDisplay

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
    # Formatting
    "PlacementFormatterMixin",
    # Displays
    "LayermapDisplay",
    "PlainDisplay",
    "RichDisplay",
    "EventDiskWriter",
    "EventRecord",
    # Display configs
    "BaseObserverConfig",
    "DiskWriterConfig",
    "DisplayObserverConfig",
    "LayermapDisplayConfig",
    "LogObserverConfig",
    "ObserversConfig",
    "PlainDisplayConfig",
    "RichDisplayConfig",
    # Progress estimation
    "AllocationProgressInfo",
    "ProgressEstimate",
    "SearchProgressEstimator",
]
