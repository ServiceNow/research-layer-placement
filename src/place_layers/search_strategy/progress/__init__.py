"""Progress display for search strategies.

This module provides an observer-based progress display system for cost-constrained
search strategies. The key components are:

- Events: Dataclasses representing key points in the search process
- SearchObserver: Protocol defining the observer interface
- RichSearchDisplay: Rich-based implementation for terminal display
- SearchProgressEstimator: Progress estimation and refinement
- ProgressEstimatorConfig: Hydra-style configs for progress estimators

Usage:
    from place_layers.search_strategy.progress import RichSearchDisplay

    with RichSearchDisplay() as display:
        search_strategy = config.create_search_strategy(observer=display)
        result = await search_strategy.search(problem)

With progress estimation:
    from place_layers.search_strategy.progress import (
        RichSearchDisplay,
        SearchProgressEstimator,
    )

    # Create estimator from config
    estimator = config.progress_estimator.create_estimator(...)

    with RichSearchDisplay(progress_estimator=estimator) as display:
        search_strategy = config.create_search_strategy(observer=display)
        result = await search_strategy.search(problem)
"""

from place_layers.search_strategy.progress.estimator import (
    AllocationProgressInfo,
    ProgressEstimate,
    ProgressUpdate,
    SearchProgressEstimator,
)
from place_layers.search_strategy.progress.estimator_config import (
    BaseProgressEstimatorConfig,
    HierarchicalProgressEstimatorConfig,
    ProgressEstimatorConfig,
)
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
from place_layers.search_strategy.progress.protocol import SearchObserver
from place_layers.search_strategy.progress.rich_display import RichSearchDisplay

__all__ = [
    # Events
    "AllocationStartedEvent",
    "BeamGrowthStepEvent",
    "BestUpdatedEvent",
    "BoundsUpdatedEvent",
    "PhaseCompletedEvent",
    "PhaseStartedEvent",
    "SearchCompletedEvent",
    "SearchStartedEvent",
    # Protocol
    "SearchObserver",
    # Display
    "RichSearchDisplay",
    # Progress estimation
    "AllocationProgressInfo",
    "ProgressEstimate",
    "ProgressUpdate",
    "SearchProgressEstimator",
    # Config
    "BaseProgressEstimatorConfig",
    "HierarchicalProgressEstimatorConfig",
    "ProgressEstimatorConfig",
]
