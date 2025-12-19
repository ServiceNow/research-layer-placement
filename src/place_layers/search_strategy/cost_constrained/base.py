"""Abstract base class for cost-constrained search strategies.

This module provides the shared outer loop logic that iterates through allocations.
Subclasses implement the inner loop (_search_within_allocation) using different strategies.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
)
from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
from place_layers.search_strategy.cost_constrained.types import (
    BeamEntry,
    CostConstrainedSearchResult,
)

if TYPE_CHECKING:
    from place_layers.problem.base_problem import BaseProblem
    from place_layers.search_strategy.cost_constrained.base_config import (
        BaseCostConstrainedConfig,
    )


logger = logging.getLogger(__name__)


class BaseCostConstrainedSearch(BaseSearchStrategy[LayerPlacement]):
    """
    Abstract base class for cost-constrained search strategies.

    This class implements the outer loop that iterates through allocations in cost order.
    Subclasses must implement `_search_within_allocation` to define how assignments
    are optimized within each fixed allocation.

    The key insight is that cost depends only on allocation (counts per mixer),
    not on which specific layers use which mixers. But performance depends on
    the full assignment. So we enumerate allocations in cost order and search
    within each allocation.

    Supports two optimization modes:

    1. MAX_PERFORMANCE mode (maximize performance, cost <= budget):
       - Navigate through allocations at or below budget
       - For each allocation, find best assignment
       - Track globally best assignment found

    2. MIN_COST mode (minimize cost, performance >= target):
       - Navigate through allocations from expensive to cheap
       - Stop when performance falls below target
    """

    def __init__(self, config: "BaseCostConstrainedConfig") -> None:
        super().__init__(config)
        self.config: BaseCostConstrainedConfig
        self.eval_count = 0

    async def search(
        self,
        problem: BaseProblem[LayerPlacement],
    ) -> CostConstrainedSearchResult:
        """
        Run cost-constrained search.

        Args:
            problem: CostConstrainedProblem with cost model and constraints.

        Returns:
            Search results with best configuration and metadata.
        """
        if not isinstance(problem, CostConstrainedProblem):
            raise TypeError(
                f"Cost-constrained search requires CostConstrainedProblem, "
                f"got {type(problem)}"
            )

        self.eval_count = 0  # Reset for new search

        config = problem.config
        cost_model = problem.cost_model
        num_layers = problem.evaluator.num_layers
        num_mixers = problem.evaluator.num_mixers

        self._log_config(config, cost_model, num_layers, num_mixers)

        if isinstance(config, MaxPerformanceProblemConfig):
            return await self._search_max_performance(
                problem=problem,
                cost_model=cost_model,
                num_layers=num_layers,
                num_mixers=num_mixers,
                cost_budget=config.cost_budget,
            )
        else:
            # MinCostProblemConfig
            return await self._search_min_cost(
                problem=problem,
                cost_model=cost_model,
                num_layers=num_layers,
                num_mixers=num_mixers,
                performance_target=config.performance_target,
            )

    async def _search_max_performance(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        num_mixers: int,
        cost_budget: float,
    ) -> CostConstrainedSearchResult:
        """MAX_PERFORMANCE mode: maximize performance subject to cost <= budget."""
        logger.info("\n" + "=" * 60)
        logger.info("MODE: MAX_PERFORMANCE (maximize perf, cost <= budget)")
        logger.info("=" * 60)

        logger.info(f"Cost budget: {cost_budget:.4f}")
        logger.info(f"Min possible cost: {cost_model.min_cost(num_layers):.4f}")
        logger.info(f"Max possible cost: {cost_model.max_cost(num_layers):.4f}")

        # Find allocation at budget boundary
        start_allocation = cost_model.allocation_at_cost_bound(
            num_layers, cost_budget, from_above=True
        )
        if start_allocation is None:
            raise ValueError(f"No feasible allocation at cost budget {cost_budget}")

        start_cost = cost_model.allocation_cost(start_allocation)
        logger.info(f"Starting allocation: {start_allocation}")
        logger.info(f"Starting cost: {start_cost:.4f}")

        # Track best overall
        best_entry: BeamEntry | None = None
        allocations_explored = 0

        # Iterate through allocations from expensive to cheap (decreasing cost)
        for alloc_cost, allocation in cost_model.allocations_by_cost(
            num_layers, direction="decreasing"
        ):
            # Skip allocations above budget
            if alloc_cost > cost_budget:
                continue

            allocations_explored += 1

            logger.info(f"\n--- Allocation {allocations_explored}: {allocation} ---")
            logger.info(f"Allocation cost: {alloc_cost:.4f}")

            # Search for best assignment within this allocation
            best_in_allocation = await self._search_within_allocation(
                problem=problem,
                cost_model=cost_model,
                allocation=allocation,
                num_layers=num_layers,
            )

            if best_in_allocation is None:
                logger.info("  No valid assignments found for this allocation")
                continue

            logger.info(
                f"  Best in allocation: score={best_in_allocation.score:.4f}"
            )

            # Update global best
            if best_entry is None or best_in_allocation.score > best_entry.score:
                best_entry = best_in_allocation
                logger.info(
                    f"  NEW BEST: score={best_entry.score:.4f}, cost={best_entry.cost:.4f}"
                )

            # Check for max allocations limit
            if allocations_explored >= self.config.max_allocations:
                logger.info(f"Reached max allocations limit: {self.config.max_allocations}")
                break

        if best_entry is None:
            # Fallback to minimum cost allocation
            min_alloc = cost_model.min_cost_allocation(num_layers)
            assignments = self._allocation_to_assignment(min_alloc, num_layers)
            placement = LayerPlacement(assignments=assignments)
            score = await problem.evaluate(placement)
            self.eval_count += 1
            best_entry = BeamEntry(
                assignments=assignments,
                allocation=min_alloc,
                cost=cost_model.allocation_cost(min_alloc),
                score=score,
            )

        return self._create_result(
            best_entry=best_entry,
            mode="max_performance",
            constraint_value=cost_budget,
            allocations_explored=allocations_explored,
        )

    async def _search_min_cost(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        num_mixers: int,
        performance_target: float,
    ) -> CostConstrainedSearchResult:
        """MIN_COST mode: minimize cost subject to performance >= target."""
        logger.info("\n" + "=" * 60)
        logger.info("MODE: MIN_COST (minimize cost, perf >= target)")
        logger.info("=" * 60)

        logger.info(f"Performance target: {performance_target:.4f}")

        # Start from most expensive allocation (maximum performance expected)
        start_allocation = cost_model.max_cost_allocation(num_layers)
        start_cost = cost_model.allocation_cost(start_allocation)
        logger.info(f"Starting allocation: {start_allocation}")
        logger.info(f"Starting cost: {start_cost:.4f}")

        # Track best (lowest cost) that meets target
        best_entry: BeamEntry | None = None
        allocations_explored = 0

        # Iterate through allocations from expensive to cheap
        for alloc_cost, allocation in cost_model.allocations_by_cost(
            num_layers, direction="decreasing"
        ):
            allocations_explored += 1

            logger.info(f"\n--- Allocation {allocations_explored}: {allocation} ---")
            logger.info(f"Allocation cost: {alloc_cost:.4f}")

            # Search for best assignment within this allocation
            best_in_allocation = await self._search_within_allocation(
                problem=problem,
                cost_model=cost_model,
                allocation=allocation,
                num_layers=num_layers,
            )

            if best_in_allocation is None:
                logger.info("  No valid assignments found for this allocation")
                continue

            logger.info(
                f"  Best in allocation: score={best_in_allocation.score:.4f}"
            )

            # Check if it meets the performance target
            if best_in_allocation.score >= performance_target:
                # This allocation meets the target - it's the best (lowest cost) so far
                best_entry = best_in_allocation
                logger.info(
                    f"  MEETS TARGET: score={best_entry.score:.4f}, cost={best_entry.cost:.4f}"
                )
            else:
                # Performance dropped below target - stop searching
                logger.info(
                    f"  Performance {best_in_allocation.score:.4f} < target {performance_target:.4f}"
                )
                logger.info("  Stopping search (allocations getting cheaper won't help)")
                break

            # Check for max allocations limit
            if allocations_explored >= self.config.max_allocations:
                logger.info(f"Reached max allocations limit: {self.config.max_allocations}")
                break

        if best_entry is None:
            # No allocation met the target - return the most expensive as best effort
            max_alloc = cost_model.max_cost_allocation(num_layers)
            assignments = self._allocation_to_assignment(max_alloc, num_layers)
            placement = LayerPlacement(assignments=assignments)
            score = await problem.evaluate(placement)
            self.eval_count += 1
            best_entry = BeamEntry(
                assignments=assignments,
                allocation=max_alloc,
                cost=cost_model.allocation_cost(max_alloc),
                score=score,
            )
            logger.warning(
                f"No allocation met performance target {performance_target}. "
                f"Returning best effort: score={best_entry.score:.4f}"
            )

        return self._create_result(
            best_entry=best_entry,
            mode="min_cost",
            constraint_value=performance_target,
            allocations_explored=allocations_explored,
        )

    @abstractmethod
    async def _search_within_allocation(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        allocation: Allocation,
        num_layers: int,
    ) -> BeamEntry | None:
        """
        Search for the best assignment within a fixed allocation.

        This is the inner loop that subclasses must implement.
        Different strategies (swap beam, hierarchical, SA) provide different implementations.

        Args:
            problem: The optimization problem (for evaluation).
            cost_model: Cost model for cost calculations.
            allocation: Fixed allocation (counts per mixer).
            num_layers: Total number of layers.

        Returns:
            Best BeamEntry found, or None if no valid assignments exist.
        """
        pass

    def _allocation_to_assignment(
        self,
        allocation: Allocation,
        num_layers: int,
    ) -> tuple[int, ...]:
        """
        Convert an allocation to a canonical assignment.

        Creates a deterministic assignment that achieves the given allocation.
        Layers are assigned to mixers in order (first n_0 layers to mixer 0,
        next n_1 layers to mixer 1, etc.).
        """
        assignments: list[int] = []
        for mixer_idx, count in enumerate(allocation):
            assignments.extend([mixer_idx] * count)

        assert len(assignments) == num_layers, (
            f"Allocation sum {sum(allocation)} != num_layers {num_layers}"
        )
        return tuple(assignments)

    def _assignment_to_allocation(
        self,
        assignments: tuple[int, ...],
        num_mixers: int,
    ) -> Allocation:
        """Convert an assignment to its allocation."""
        counts = [0] * num_mixers
        for mixer in assignments:
            counts[mixer] += 1
        return tuple(counts)

    def _create_result(
        self,
        best_entry: BeamEntry,
        mode: Literal["max_performance", "min_cost"],
        constraint_value: float,
        allocations_explored: int,
    ) -> CostConstrainedSearchResult:
        """Create the final search result."""
        logger.info("\n" + "=" * 60)
        logger.info("SEARCH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best score: {best_entry.score:.4f}")
        logger.info(f"Best cost: {best_entry.cost:.4f}")
        logger.info(f"Best allocation: {best_entry.allocation}")
        logger.info(f"Allocations explored: {allocations_explored}")
        logger.info(f"Total evaluations: {self.eval_count}")

        return CostConstrainedSearchResult(
            best_parameters=LayerPlacement(assignments=best_entry.assignments),
            best_score=best_entry.score,
            total_evaluations=self.eval_count,
            final_cost=best_entry.cost,
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
        logger.info(f"Max allocations: {self.config.max_allocations}")
