"""Abstract base class for search strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from place_layers.core.types import ParametersT

if TYPE_CHECKING:
    from place_layers.core.types import SearchResult
    from place_layers.problem.base_problem import BaseProblem
    from place_layers.search_strategy.base_search_strategy_config import (
        BaseSearchStrategyConfig,
    )


class BaseSearchStrategy(Generic[ParametersT]):
    """Abstract base class for search strategies."""

    def __init__(self, config: "BaseSearchStrategyConfig") -> None:
        """
        Initialize the search strategy with configuration.

        Args:
            config: Search strategy configuration.
        """
        self.config = config

    async def search(
        self,
        problem: "BaseProblem[ParametersT]",
    ) -> "SearchResult[ParametersT]":
        """
        Run the search algorithm to find optimal parameters.

        Args:
            problem: Problem definition with embedded evaluator and constraints.

        Returns:
            Search results with best parameters found.
        """
        raise NotImplementedError
