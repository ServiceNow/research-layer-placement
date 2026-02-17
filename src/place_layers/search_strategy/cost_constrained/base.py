"""Abstract base class for cost-constrained search strategies.

This module provides the shared cost-constrained search logic: problem validation,
mode dispatch, result creation, and event emission. Subclasses implement the actual
search algorithms via _search_max_performance() and _search_min_cost().
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, Sequence, overload

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
    MinCostProblemConfig,
)
from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
from place_layers.search_strategy.cost_constrained.types import (
    BeamEntry,
    CostConstrainedSearchResult,
)

if TYPE_CHECKING:
    from place_layers.evaluator.evaluators import Evaluators
    from place_layers.problem.base_problem import BaseProblem
    from place_layers.search_strategy.cost_constrained.base_config import (
        BaseCostConstrainedConfig,
    )
    from place_layers.search_strategy.progress import SearchObserver
    from place_layers.search_strategy.progress.estimator import SearchProgressEstimator


logger = logging.getLogger(__name__)


class BaseCostConstrainedSearch(BaseSearchStrategy[LayerPlacement]):
    """
    Abstract base class for cost-constrained search strategies.

    This class provides the shared logic for cost-constrained search:
    - Problem validation (must be CostConstrainedProblem)
    - Mode dispatch (max_performance vs min_cost)
    - Result creation and event emission

    Subclasses implement _search_max_performance() and _search_min_cost()
    with their specific search algorithms.

    Supports two optimization modes:

    1. MAX_PERFORMANCE mode (maximize performance, cost <= budget):
       - Find best assignment within cost budget
       - Track globally best assignment found

    2. MIN_COST mode (minimize cost, performance >= target):
       - Find cheapest assignment meeting performance target
    """

    def __init__(
        self,
        config: "BaseCostConstrainedConfig",
        observers: "Sequence[SearchObserver]" = (),
        progress_estimator: "SearchProgressEstimator | None" = None,
    ) -> None:
        super().__init__(config, observers)
        self.config: BaseCostConstrainedConfig
        self._progress_estimator = progress_estimator
        self.eval_count = 0

    @overload
    async def search(
        self,
        problem: "BaseProblem[LayerPlacement]",
        dry_run: Literal[False] = ...,
    ) -> CostConstrainedSearchResult: ...

    @overload
    async def search(
        self,
        problem: "BaseProblem[LayerPlacement]",
        dry_run: Literal[True],
    ) -> None: ...

    async def search(
        self,
        problem: "BaseProblem[LayerPlacement]",
        dry_run: bool = False,
    ) -> CostConstrainedSearchResult | None:
        """
        Run cost-constrained search.

        Args:
            problem: CostConstrainedProblem with cost model and constraints.
            dry_run: If True, emit initial events only (for preview) and return None.

        Returns:
            Search results with best configuration and metadata, or None if dry_run.
        """
        if not isinstance(problem, CostConstrainedProblem):
            raise TypeError(
                f"Cost-constrained search requires CostConstrainedProblem, "
                f"got {type(problem)}"
            )

        self.eval_count = 0  # Reset for new search

        config = problem.config
        cost_model = problem.cost_model
        num_layers = problem.evaluators.num_layers
        num_mixers = cost_model.num_mixers
        mixer_names = cost_model.phase_order

        self._log_config(config, cost_model, num_layers, num_mixers)

        match config:
            case MaxPerformanceProblemConfig(cost_budget=budget):
                mode: Literal["max_performance", "min_cost"] = "max_performance"
                constraint_value = budget
            case MinCostProblemConfig(performance_target=target):
                mode = "min_cost"
                constraint_value = target
            case _:
                raise TypeError(f"Unexpected config type: {type(config)}")

        self._emit_search_started(
            mode=mode,
            constraint_value=constraint_value,
            cost_model=cost_model,
            num_layers=num_layers,
            mixer_names=tuple(mixer_names),
        )

        if dry_run:
            return None

        cost_floor = config.cost_floor

        match config:
            case MaxPerformanceProblemConfig(cost_budget=budget):
                return await self._search_max_performance(
                    problem=problem,
                    cost_model=cost_model,
                    num_layers=num_layers,
                    cost_budget=budget,
                    cost_floor=cost_floor,
                )
            case MinCostProblemConfig(performance_target=target):
                return await self._search_min_cost(
                    problem=problem,
                    cost_model=cost_model,
                    num_layers=num_layers,
                    performance_target=target,
                    cost_floor=cost_floor,
                )
            case _:
                raise TypeError(f"Unexpected config type: {type(config)}")

    def _emit_search_started(
        self,
        mode: Literal["max_performance", "min_cost"],
        constraint_value: float,
        cost_model: BaseCostModel,
        num_layers: int,
        mixer_names: tuple[str, ...],
    ) -> None:
        """Emit SearchStartedEvent. Subclasses override to add strategy-specific fields."""
        if not self.observers:
            return

        from place_layers.search_strategy.progress import SearchStartedEvent

        event = SearchStartedEvent(
            mode=mode,
            constraint_value=constraint_value,
            min_cost=cost_model.min_cost(num_layers),
            max_cost=cost_model.max_cost(num_layers),
            num_layers=num_layers,
            mixer_names=mixer_names,
            max_allocations=None,
            total_allocations=cost_model.count_allocations(num_layers),
        )
        for observer in self.observers:
            observer.on_search_started(event)

    @abstractmethod
    async def _search_max_performance(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        cost_budget: float,
        cost_floor: float | None = None,
    ) -> CostConstrainedSearchResult:
        """MAX_PERFORMANCE mode: maximize performance subject to cost <= budget."""
        ...

    @abstractmethod
    async def _search_min_cost(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        performance_target: float,
        cost_floor: float | None = None,
    ) -> CostConstrainedSearchResult:
        """MIN_COST mode: minimize cost subject to performance >= target."""
        ...

    @staticmethod
    def _allocation_to_assignment(
        allocation: Allocation,
        num_layers: int,
    ) -> list[str]:
        """
        Convert an allocation to a canonical assignment.

        Creates a deterministic assignment that achieves the given allocation.
        Layer ordering is determined by allocation dict iteration order,
        which comes from config.mixer_costs key order (phase_order).

        Args:
            allocation: Dict mapping mixer name to layer count.
            num_layers: Total number of layers.

        Returns:
            List of mixer names (length = num_layers).
        """
        assignments: list[str] = []
        for mixer, count in allocation.items():
            assignments.extend([mixer] * count)

        assert (
            len(assignments) == num_layers
        ), f"Allocation sum {sum(allocation.values())} != num_layers {num_layers}"
        return assignments

    def _create_result(
        self,
        best_entry: BeamEntry,
        cost_model: BaseCostModel,
        mode: Literal["max_performance", "min_cost"],
        constraint_value: float,
        allocations_explored: int,
    ) -> CostConstrainedSearchResult:
        """Create the final search result."""
        assert best_entry.scores is not None

        # Compute cost from allocation (never stored, always computed fresh)
        final_cost = cost_model.allocation_cost(best_entry.allocation)

        logger.info("\n" + "=" * 60)
        logger.info("SEARCH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best scores: {best_entry.scores}")
        logger.info(f"Best cost: {final_cost:.4f}")
        logger.info(f"Best allocation: {best_entry.allocation}")
        logger.info(f"Allocations explored: {allocations_explored}")
        logger.info(f"Total evaluations: {self.eval_count}")

        # Emit search completed event
        if self.observers:
            from place_layers.search_strategy.progress import SearchCompletedEvent

            event = SearchCompletedEvent(
                best_scores=best_entry.scores,
                best_cost=final_cost,
                total_evaluations=self.eval_count,
                allocations_explored=allocations_explored,
            )
            for observer in self.observers:
                observer.on_search_completed(event)

        return CostConstrainedSearchResult(
            best_parameters=LayerPlacement(assignments=best_entry.assignments),
            best_scores=best_entry.scores,
            total_evaluations=self.eval_count,
            final_cost=final_cost,
            mode=mode,
            constraint_value=constraint_value,
            allocations_explored=allocations_explored,
        )

    def _log_config(self, config, cost_model, num_layers: int, num_mixers: int) -> None:
        """Log search configuration."""
        logger.info("\n" + "=" * 60)
        logger.info("Cost-Constrained Search Configuration")
        logger.info("=" * 60)
        logger.info(f"Strategy: {self.__class__.__name__}")
        logger.info(f"Total layers: {num_layers}")
        logger.info(f"Number of mixer types: {num_mixers}")
        logger.info(f"Mode: {config.problem_type}")
        if config.cost_floor is not None:
            logger.info(f"Cost floor: {config.cost_floor}")
