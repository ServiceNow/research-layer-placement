"""Base configuration for search strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Sequence

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from place_layers.problem.base_problem import BaseProblem
    from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
    from place_layers.search_strategy.progress import SearchObserver
    from place_layers.search_strategy.progress.estimator import SearchProgressEstimator


class BaseSearchStrategyConfig(BaseModel):
    """Base configuration for all search strategy implementations."""

    model_config = {"arbitrary_types_allowed": True}

    strategy_type: str = Field(description="Type identifier for search strategy")

    # Each subclass must define this
    _impl_class: ClassVar[type[BaseSearchStrategy]]

    def create_search_strategy(
        self,
        observers: "Sequence[SearchObserver]" = (),
        progress_estimator: "SearchProgressEstimator | None" = None,
    ) -> "BaseSearchStrategy":
        """Smart constructor: instantiate search strategy from config.

        Args:
            observers: Sequence of observers to receive progress events during
                search. If any observers are provided but the strategy doesn't
                support them, raises NotImplementedError.
            progress_estimator: Optional progress estimator for tracking search
                progress and refining bounds as allocations complete.

        Returns:
            Instantiated search strategy.

        Raises:
            NotImplementedError: If observers are provided but the strategy
                doesn't support progress events.
        """
        return self._impl_class(
            config=self, observers=observers, progress_estimator=progress_estimator
        )

    def create_progress_estimator(
        self, problem: "BaseProblem"
    ) -> "SearchProgressEstimator | None":
        """Create a progress estimator for this search strategy.

        Default returns None. Subclasses override if they support progress estimation.

        Args:
            problem: The optimization problem.

        Returns:
            Configured SearchProgressEstimator, or None if not supported.
        """
        return None
