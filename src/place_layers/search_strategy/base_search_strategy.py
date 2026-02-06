"""Abstract base class for search strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence, overload


if TYPE_CHECKING:
    from place_layers.core.types import SearchResult
    from place_layers.problem.base_problem import BaseProblem
    from place_layers.search_strategy.base_search_strategy_config import (
        BaseSearchStrategyConfig,
    )
    from place_layers.search_strategy.progress import SearchObserver


class BaseSearchStrategy[ParametersT]:
    """Abstract base class for search strategies."""

    def __init__(
        self,
        config: "BaseSearchStrategyConfig",
        observers: "Sequence[SearchObserver]" = (),
        progress_estimator: Any = None,
    ) -> None:
        """
        Initialize the search strategy with configuration.

        Args:
            config: Search strategy configuration.
            observers: Sequence of observers for progress events.
            progress_estimator: Optional progress estimator (used by subclasses).
        """
        del progress_estimator  # Used by subclasses only
        self.config = config
        self.observers = tuple(observers)

    @overload
    async def search(
        self,
        problem: "BaseProblem[ParametersT]",
        dry_run: Literal[False] = ...,
    ) -> "SearchResult[ParametersT]": ...

    @overload
    async def search(
        self,
        problem: "BaseProblem[ParametersT]",
        dry_run: Literal[True],
    ) -> None: ...

    async def search(
        self,
        problem: "BaseProblem[ParametersT]",
        dry_run: bool = False,
    ) -> "SearchResult[ParametersT] | None":
        """
        Run the search algorithm to find optimal parameters.

        Args:
            problem: Problem definition with embedded evaluator and constraints.
            dry_run: If True, emit initial events only (for preview) and return None.

        Returns:
            Search results with best parameters found, or None if dry_run.
        """
        raise NotImplementedError
