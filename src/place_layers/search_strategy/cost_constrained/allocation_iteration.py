"""Allocation-iteration search: iterates allocations in cost order.

This module provides the intermediate class that implements the outer loop
of iterating through allocations. Concrete strategies (hierarchical, swap-beam,
annealing) subclass this and implement _search_within_allocation().
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.search_strategy.cost_constrained.base import BaseCostConstrainedSearch
from place_layers.search_strategy.cost_constrained.types import BeamEntry

if TYPE_CHECKING:
    from place_layers.core.types import LayerPlacement
    from place_layers.search_strategy.cost_constrained.base_config import (
        AllocationIterationConfig,
    )


logger = logging.getLogger(__name__)


class AllocationIterationSearch(BaseCostConstrainedSearch):
    """Iterates allocations in cost order, calls _search_within_allocation() per allocation.

    This class implements the outer loop that navigates through allocations
    in cost order. Subclasses implement _search_within_allocation() to define
    how assignments are optimized within each fixed allocation.

    The key insight is that cost depends only on allocation (counts per mixer),
    not on which specific layers use which mixers. But performance depends on
    the full assignment. So we enumerate allocations in cost order and search
    within each allocation.
    """

    config: "AllocationIterationConfig"

    def _emit_search_started(
        self,
        mode: Literal["max_performance", "min_cost"],
        constraint_value: float,
        cost_model: BaseCostModel,
        num_layers: int,
        mixer_names: tuple[str, ...],
    ) -> None:
        """Override: add max_allocations, progress estimator bounds, AllocationPlanEvent."""
        if not self.observers:
            return

        from place_layers.search_strategy.progress import SearchStartedEvent

        # Get initial bounds from progress estimator if available
        eval_lower = None
        eval_upper = None
        eval_expected = None
        if self._progress_estimator is not None:
            estimate = self._progress_estimator.bounds_after(
                completed_allocations=0, completed_evals=0
            )
            eval_lower = estimate.lower
            eval_upper = estimate.upper
            eval_expected = estimate.expected

        event = SearchStartedEvent(
            mode=mode,
            constraint_value=constraint_value,
            min_cost=cost_model.min_cost(num_layers),
            max_cost=cost_model.max_cost(num_layers),
            num_layers=num_layers,
            mixer_names=mixer_names,
            max_allocations=self.config.max_allocations,
            total_allocations=cost_model.count_allocations(num_layers),
            eval_count_lower_bound=eval_lower,
            eval_count_upper_bound=eval_upper,
            eval_count_expected=eval_expected,
        )
        for observer in self.observers:
            observer.on_search_started(event)

        # Emit allocation plan if progress estimator is available
        if self._progress_estimator is not None:
            from place_layers.search_strategy.progress import AllocationPlanEvent

            plan_event = AllocationPlanEvent(
                allocation_info=self._progress_estimator.allocation_info,
                mixer_names=mixer_names,
            )
            for observer in self.observers:
                observer.on_allocation_plan(plan_event)

    async def _search_max_performance(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        cost_budget: float,
    ) -> "CostConstrainedSearchResult":
        """MAX_PERFORMANCE mode: maximize performance subject to cost <= budget."""
        from place_layers.core.types import LayerPlacement
        from place_layers.search_strategy.cost_constrained.types import (
            CostConstrainedSearchResult,
        )

        mixer_names = cost_model.phase_order

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

            # Get per-allocation bounds from progress estimator
            alloc_lower: int | None = None
            alloc_upper: int | None = None
            alloc_expected: int | None = None
            if self._progress_estimator is not None:
                alloc_info = self._progress_estimator.allocation_info[
                    allocations_explored - 1
                ]
                alloc_lower = alloc_info.estimated_lower
                alloc_upper = alloc_info.estimated_upper
                alloc_expected = int(alloc_info.expected)

            # Emit allocation started event
            if self.observers:
                from place_layers.search_strategy.progress import AllocationStartedEvent

                event = AllocationStartedEvent(
                    allocation_index=allocations_explored,
                    total_allocations=self.config.max_allocations,
                    allocation=allocation,
                    allocation_cost=alloc_cost,
                    mixer_names=tuple(mixer_names),
                    eval_lower_bound=alloc_lower,
                    eval_upper_bound=alloc_upper,
                    eval_expected=alloc_expected,
                )
                for observer in self.observers:
                    observer.on_allocation_started(event)

            # Track eval count before allocation for progress estimation
            evals_before = self.eval_count

            # Search for best assignment within this allocation
            beam_results = await self._search_within_allocation(
                problem=problem,
                cost_model=cost_model,
                allocation=allocation,
                num_layers=num_layers,
            )
            best_in_allocation = beam_results[0] if beam_results else None

            # Emit bounds update if progress estimator is present
            if self._progress_estimator is not None and self.observers:
                from place_layers.search_strategy.progress import BoundsUpdatedEvent

                previous = self._progress_estimator.bounds_after(
                    completed_allocations=allocations_explored - 1,
                    completed_evals=evals_before,
                )
                current = self._progress_estimator.bounds_after(
                    completed_allocations=allocations_explored,
                    completed_evals=self.eval_count,
                )

                if previous.lower != current.lower or previous.upper != current.upper:
                    event = BoundsUpdatedEvent(
                        previous_lower=previous.lower,
                        previous_upper=previous.upper,
                        new_lower=current.lower,
                        new_upper=current.upper,
                        completed=self.eval_count,
                        remaining_allocations=(
                            len(self._progress_estimator.allocation_info)
                            - allocations_explored
                        ),
                        reason="allocation_complete",
                    )
                    for observer in self.observers:
                        observer.on_bounds_updated(event)

            # Emit allocation completed event with final placement
            if self.observers:
                from place_layers.search_strategy.progress import (
                    AllocationCompletedEvent,
                )

                # Build top_placements from all beam entries with valid scores
                top_placements = tuple(
                    (entry.assignments, entry.score)
                    for entry in beam_results
                    if entry.score is not None
                )

                event = AllocationCompletedEvent(
                    allocation_index=allocations_explored,
                    allocation=allocation,
                    allocation_cost=alloc_cost,
                    mixer_names=tuple(mixer_names),
                    assignments=(
                        best_in_allocation.assignments
                        if best_in_allocation is not None
                        else None
                    ),
                    score=(
                        best_in_allocation.score
                        if best_in_allocation is not None
                        else None
                    ),
                    top_placements=top_placements,
                )
                for observer in self.observers:
                    observer.on_allocation_completed(event)

            if best_in_allocation is None:
                logger.info("  No valid assignments found for this allocation")
                continue

            # Entries returned from _search_within_allocation are always evaluated
            assert best_in_allocation.score is not None

            logger.info(f"  Best in allocation: score={best_in_allocation.score:.4f}")

            # Update global best
            if best_entry is None or best_in_allocation.score > (best_entry.score or float("-inf")):
                best_entry = best_in_allocation
                best_cost = cost_model.allocation_cost(best_entry.allocation)
                logger.info(
                    f"  NEW BEST: score={best_in_allocation.score:.4f}, cost={best_cost:.4f}"
                )

                # Emit best updated event
                if self.observers:
                    from place_layers.search_strategy.progress import BestUpdatedEvent

                    event = BestUpdatedEvent(
                        score=best_in_allocation.score,
                        cost=best_cost,
                        allocation=best_entry.allocation,
                        assignments=best_entry.assignments,
                    )
                    for observer in self.observers:
                        observer.on_best_updated(event)

            # Check for max allocations limit
            if allocations_explored >= self.config.max_allocations:
                logger.info(
                    f"Reached max allocations limit: {self.config.max_allocations}"
                )
                break

        if best_entry is None:
            # Fallback to minimum cost allocation
            min_alloc = cost_model.min_cost_allocation(num_layers)
            assignments = self._allocation_to_assignment(min_alloc, num_layers)
            placement = LayerPlacement(assignments=tuple(assignments))
            score = await problem.evaluate(placement)
            self.eval_count += 1
            best_entry = BeamEntry(
                assignments=tuple(assignments),
                allocation=min_alloc,
                score=score,
            )

        return self._create_result(
            best_entry=best_entry,
            cost_model=cost_model,
            mode="max_performance",
            constraint_value=cost_budget,
            allocations_explored=allocations_explored,
        )

    async def _search_min_cost(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        num_layers: int,
        performance_target: float,
    ) -> "CostConstrainedSearchResult":
        """MIN_COST mode: minimize cost subject to performance >= target."""
        from place_layers.core.types import LayerPlacement
        from place_layers.search_strategy.cost_constrained.types import (
            CostConstrainedSearchResult,
        )

        mixer_names = cost_model.phase_order

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

            # Get per-allocation bounds from progress estimator
            alloc_lower: int | None = None
            alloc_upper: int | None = None
            alloc_expected: int | None = None
            if self._progress_estimator is not None:
                alloc_info = self._progress_estimator.allocation_info[
                    allocations_explored - 1
                ]
                alloc_lower = alloc_info.estimated_lower
                alloc_upper = alloc_info.estimated_upper
                alloc_expected = int(alloc_info.expected)

            # Emit allocation started event
            if self.observers:
                from place_layers.search_strategy.progress import AllocationStartedEvent

                event = AllocationStartedEvent(
                    allocation_index=allocations_explored,
                    total_allocations=self.config.max_allocations,
                    allocation=allocation,
                    allocation_cost=alloc_cost,
                    mixer_names=tuple(mixer_names),
                    eval_lower_bound=alloc_lower,
                    eval_upper_bound=alloc_upper,
                    eval_expected=alloc_expected,
                )
                for observer in self.observers:
                    observer.on_allocation_started(event)

            # Track eval count before allocation for progress estimation
            evals_before = self.eval_count

            # Search for best assignment within this allocation
            beam_results = await self._search_within_allocation(
                problem=problem,
                cost_model=cost_model,
                allocation=allocation,
                num_layers=num_layers,
            )
            best_in_allocation = beam_results[0] if beam_results else None

            # Emit bounds update if progress estimator is present
            if self._progress_estimator is not None and self.observers:
                from place_layers.search_strategy.progress import BoundsUpdatedEvent

                previous = self._progress_estimator.bounds_after(
                    completed_allocations=allocations_explored - 1,
                    completed_evals=evals_before,
                )
                current = self._progress_estimator.bounds_after(
                    completed_allocations=allocations_explored,
                    completed_evals=self.eval_count,
                )

                if previous.lower != current.lower or previous.upper != current.upper:
                    event = BoundsUpdatedEvent(
                        previous_lower=previous.lower,
                        previous_upper=previous.upper,
                        new_lower=current.lower,
                        new_upper=current.upper,
                        completed=self.eval_count,
                        remaining_allocations=(
                            len(self._progress_estimator.allocation_info)
                            - allocations_explored
                        ),
                        reason="allocation_complete",
                    )
                    for observer in self.observers:
                        observer.on_bounds_updated(event)

            # Emit allocation completed event with final placement
            if self.observers:
                from place_layers.search_strategy.progress import (
                    AllocationCompletedEvent,
                )

                # Build top_placements from all beam entries with valid scores
                top_placements = tuple(
                    (entry.assignments, entry.score)
                    for entry in beam_results
                    if entry.score is not None
                )

                event = AllocationCompletedEvent(
                    allocation_index=allocations_explored,
                    allocation=allocation,
                    allocation_cost=alloc_cost,
                    mixer_names=tuple(mixer_names),
                    assignments=(
                        best_in_allocation.assignments
                        if best_in_allocation is not None
                        else None
                    ),
                    score=(
                        best_in_allocation.score
                        if best_in_allocation is not None
                        else None
                    ),
                    top_placements=top_placements,
                )
                for observer in self.observers:
                    observer.on_allocation_completed(event)

            if best_in_allocation is None:
                logger.info("  No valid assignments found for this allocation")
                continue

            # Entries returned from _search_within_allocation are always evaluated
            assert best_in_allocation.score is not None

            logger.info(f"  Best in allocation: score={best_in_allocation.score:.4f}")

            # Check if it meets the performance target
            if best_in_allocation.score >= performance_target:
                # This allocation meets the target - it's the best (lowest cost) so far
                best_entry = best_in_allocation
                best_cost = cost_model.allocation_cost(best_entry.allocation)
                logger.info(
                    f"  MEETS TARGET: score={best_in_allocation.score:.4f}, cost={best_cost:.4f}"
                )

                # Emit best updated event
                if self.observers:
                    from place_layers.search_strategy.progress import BestUpdatedEvent

                    event = BestUpdatedEvent(
                        score=best_in_allocation.score,
                        cost=best_cost,
                        allocation=best_entry.allocation,
                        assignments=best_entry.assignments,
                    )
                    for observer in self.observers:
                        observer.on_best_updated(event)
            else:
                # Performance dropped below target - stop searching
                logger.info(
                    f"  Performance {best_in_allocation.score:.4f} < target {performance_target:.4f}"
                )
                logger.info(
                    "  Stopping search (allocations getting cheaper won't help)"
                )
                break

            # Check for max allocations limit
            if allocations_explored >= self.config.max_allocations:
                logger.info(
                    f"Reached max allocations limit: {self.config.max_allocations}"
                )
                break

        if best_entry is None:
            # No allocation met the target - return the most expensive as best effort
            max_alloc = cost_model.max_cost_allocation(num_layers)
            assignments = self._allocation_to_assignment(max_alloc, num_layers)
            placement = LayerPlacement(assignments=tuple(assignments))
            score = await problem.evaluate(placement)
            self.eval_count += 1
            best_entry = BeamEntry(
                assignments=tuple(assignments),
                allocation=max_alloc,
                score=score,
            )
            logger.warning(
                f"No allocation met performance target {performance_target}. "
                f"Returning best effort: score={best_entry.score:.4f}"
            )

        return self._create_result(
            best_entry=best_entry,
            cost_model=cost_model,
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
    ) -> list[BeamEntry]:
        """Search for the best assignment within a fixed allocation.

        This is the inner loop that subclasses must implement.
        Different strategies (swap beam, hierarchical, SA) provide different implementations.

        Args:
            problem: The optimization problem (for evaluation).
            cost_model: Cost model for cost calculations.
            allocation: Fixed allocation (counts per mixer).
            num_layers: Total number of layers.

        Returns:
            List of BeamEntry results sorted by score descending, or empty list
            if no valid assignments exist.
        """
        pass

    def _log_config(self, config, cost_model, num_layers: int, num_mixers: int) -> None:
        """Override: add max_allocations to log."""
        super()._log_config(config, cost_model, num_layers, num_mixers)
        logger.info(f"Max allocations: {self.config.max_allocations}")
