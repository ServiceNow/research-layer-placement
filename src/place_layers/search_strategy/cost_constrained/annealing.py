"""Simulated annealing for cost-constrained optimization.

This implements an inner loop strategy that uses simulated annealing to explore
the assignment space within a fixed allocation.

Key features:
- Ruin-and-recreate move operator: shuffles k positions (generalizes k-cycles)
- Multi-start: multiple random restarts explore different basins
- Adaptive reheating: reheats when acceptance rate drops below threshold
- Geometric cooling schedule
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, ClassVar, Literal, Sequence

from pydantic import Field

from place_layers.annealing.sa_core import SAConfig, run_multi_start_sa
from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.search_strategy.cost_constrained.allocation_iteration import (
    AllocationIterationSearch,
)
from place_layers.search_strategy.cost_constrained.base_config import (
    AllocationIterationConfig,
)
from place_layers.search_strategy.cost_constrained.types import BeamEntry

if TYPE_CHECKING:
    from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
    from place_layers.search_strategy.progress import SearchObserver
    from place_layers.search_strategy.progress.estimator import SearchProgressEstimator


logger = logging.getLogger(__name__)


class AnnealingSearch(AllocationIterationSearch):
    """
    Cost-constrained search using simulated annealing for the inner loop.

    Within each allocation, uses simulated annealing with:
    1. Random initial assignments (multi-start)
    2. Ruin-and-recreate move operator (shuffles k positions)
    3. Metropolis acceptance criterion
    4. Adaptive reheating when acceptance rate drops
    5. Geometric cooling schedule

    The ruin-and-recreate operator generalizes k-cycles: at k=2 it's a swap,
    at higher k it can reach states that require multiple intermediate swaps.
    The perturbation size k is coupled to temperature via a geometric distribution.
    """

    def __init__(
        self,
        config: "AnnealingConfig",
        observers: "Sequence[SearchObserver]" = (),
        progress_estimator: "SearchProgressEstimator | None" = None,
    ) -> None:
        super().__init__(config, observers, progress_estimator)
        self.config: AnnealingConfig
        self._rng = random.Random(self.config.sa.seed)

    async def _search_within_allocation(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        allocation: Allocation,
        num_layers: int,
    ) -> list[BeamEntry]:
        """
        Search for the best assignment within a fixed allocation using simulated annealing.

        Uses multiple random restarts, each running a SA trajectory with
        ruin-and-recreate moves and adaptive reheating.

        Returns:
            List containing a single BeamEntry result (SA only tracks best),
            or empty list if no valid assignments exist.
        """
        from place_layers.annealing.events import AnnealingCompletedEvent

        # Check if any moves are possible (need >= 2 non-zero mixer counts)
        if not self._can_swap(allocation):
            # All layers on one mixer - just return canonical
            assignments = self._allocation_to_assignment(allocation, num_layers)
            placement = LayerPlacement(assignments=tuple(assignments))
            score = await problem.evaluate(placement)
            self.eval_count += 1

            # Notify observers
            for observer in self.observers:
                observer.on_eval_completed(self.eval_count)

            # Emit completed event with no_swaps reason
            if self.observers:
                event = AnnealingCompletedEvent(
                    steps_taken=0,
                    final_temperature=self.config.sa.initial_temperature,
                    termination_reason="no_swaps",
                    best_assignments=tuple(assignments),
                    best_score=score,
                    restarts_completed=0,
                    best_restart=0,
                    total_reheats=0,
                )
                for observer in self.observers:
                    observer.on_annealing_completed(event)

            return [
                BeamEntry(
                    assignments=tuple(assignments),
                    allocation=allocation,
                    score=score,
                )
            ]

        # Score function: evaluate + count + notify
        async def score_fn(assignment: tuple[str, ...]) -> float:
            placement = LayerPlacement(assignments=assignment)
            score = await problem.evaluate(placement)
            self.eval_count += 1
            for obs in self.observers:
                obs.on_eval_completed(self.eval_count)
            return score

        # Pass observers directly — SearchObserver satisfies SAObserver
        result = await run_multi_start_sa(
            score_fn=score_fn,
            allocation_cost=cost_model.allocation_cost,
            feasible_allocations=[allocation],
            num_layers=num_layers,
            mixer_names=cost_model.phase_order,
            cost_budget=None,
            config=self.config.sa,
            rng=self._rng,
            observers=self.observers,
        )

        # Emit completed event
        if self.observers and result.candidates:
            best_assignment, best_score = result.candidates[0]
            if result.any_early_termination:
                termination_reason = "min_temperature"
            else:
                termination_reason = "max_steps"
            event = AnnealingCompletedEvent(
                steps_taken=result.total_steps,
                final_temperature=result.final_temperature,
                termination_reason=termination_reason,
                best_assignments=best_assignment,
                best_score=best_score,
                restarts_completed=self.config.sa.num_restarts,
                best_restart=result.best_restart,
                total_reheats=result.total_reheats,
            )
            for observer in self.observers:
                observer.on_annealing_completed(event)

        if not result.candidates:
            return []

        best_assignment, best_score = result.candidates[0]
        return [
            BeamEntry(
                assignments=best_assignment,
                allocation=allocation,
                score=best_score,
            )
        ]

    @staticmethod
    def _can_swap(allocation: Allocation) -> bool:
        """Check if any allocation-preserving move is possible.

        Requires at least 2 mixer types with non-zero counts.
        """
        non_zero = sum(1 for count in allocation.values() if count > 0)
        return non_zero >= 2


class AnnealingConfig(AllocationIterationConfig):
    """Configuration for simulated annealing inner loop."""

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = AnnealingSearch

    strategy_type: Literal["cost_constrained_annealing"] = "cost_constrained_annealing"

    sa: SAConfig = Field(description="Simulated annealing configuration")
