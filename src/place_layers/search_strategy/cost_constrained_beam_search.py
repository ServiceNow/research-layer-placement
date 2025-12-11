"""Cost-constrained beam search strategy implementation.

This module implements a hierarchical beam search that operates on two levels:

1. **Allocation Level**: Navigate through allocations (counts per mixer) in cost order
   using the cost model's allocation navigation API.

2. **Assignment Level**: For each allocation, use beam search to find the best
   layer-to-mixer assignment that achieves that allocation.

The key insight is that cost depends only on allocation (counts), not on which
specific layers use which mixers. But performance depends on the full assignment.
So we enumerate allocations in cost order and search within each allocation.
"""

import logging
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from place_layers.core.types import LayerPlacement, SearchResult
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.problem.base_problem import BaseProblem
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    CostConstrainedProblemConfig,
    MaxPerformanceProblemConfig,
)
from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy
from place_layers.search_strategy.base_search_strategy_config import (
    BaseSearchStrategyConfig,
)


logger = logging.getLogger(__name__)


class BeamEntry(BaseModel):
    """An entry in the search beam."""

    model_config = {"frozen": True}

    assignments: tuple[int, ...]
    allocation: tuple[int, ...]  # Counts per mixer
    cost: float
    score: float | None = None  # None until evaluated


class CostConstrainedSearchResult(SearchResult[LayerPlacement]):
    """Search results with cost and constraint information."""

    final_cost: float
    mode: Literal["max_performance", "min_cost"]
    constraint_value: float  # cost_budget or performance_target
    allocations_explored: int


class CostConstrainedBeamSearch(BaseSearchStrategy[LayerPlacement]):
    """
    Cost-constrained beam search for layer placement optimization.

    This is a hierarchical algorithm that operates on two levels:

    1. **Outer loop (allocation level)**: Navigate through allocations (counts per mixer)
       in cost order using the cost model's allocation navigation API.

    2. **Inner loop (assignment level)**: For each allocation, use beam search to find
       the best layer-to-mixer assignment that achieves that allocation.

    Supports two optimization modes:

    1. MAX_PERFORMANCE mode (maximize performance, cost <= budget):
       - Start: Allocation at cost bound from above
       - Navigate: Through allocations at or below budget
       - Search: Find best assignment within each allocation
       - Stop: Exhausted allocations or no improvement

    2. MIN_COST mode (minimize cost, performance >= target):
       - Start: All layers on most expensive mixer
       - Navigate: Through allocations from expensive to cheap
       - Search: Find best assignment within each allocation
       - Stop: Best performance in allocation falls below target

    The key insight is that cost depends only on allocation (counts), not on which
    specific layers use which mixers. But performance depends on the full assignment.
    """

    def __init__(self, config: "CostConstrainedBeamSearchConfig") -> None:
        super().__init__(config)
        self.config: CostConstrainedBeamSearchConfig
        self.eval_count = 0

    async def search(
        self,
        problem: BaseProblem[LayerPlacement],
    ) -> CostConstrainedSearchResult:
        """
        Run cost-constrained beam search.

        Args:
            problem: CostConstrainedProblem with cost model and constraints.

        Returns:
            Search results with best configuration and metadata.
        """
        if not isinstance(problem, CostConstrainedProblem):
            raise TypeError(
                f"CostConstrainedBeamSearch requires CostConstrainedProblem, "
                f"got {type(problem)}"
            )

        self.eval_count = 0  # Reset for new search

        config = problem.config
        cost_model = problem.cost_model
        num_layers = problem.evaluator.config.num_layers
        num_mixers = problem.evaluator.config.num_mixers

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
        """
        MAX_PERFORMANCE mode: maximize performance subject to cost <= budget.

        Strategy:
        1. Find the allocation at the cost budget (from above)
        2. Iterate through allocations from expensive to cheap (within budget)
        3. For each allocation, search for the best assignment
        4. Track the globally best assignment found
        """
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
        """
        MIN_COST mode: minimize cost subject to performance >= target.

        Strategy:
        1. Start with the most expensive allocation
        2. Iterate through allocations from expensive to cheap
        3. For each allocation, search for the best assignment
        4. Stop when best performance in allocation drops below target
        """
        logger.info("\n" + "=" * 60)
        logger.info("MODE: MIN_COST (minimize cost, perf >= target)")
        logger.info("=" * 60)

        logger.info(f"Performance target: {performance_target:.4f}")
        logger.info(f"Min possible cost: {cost_model.min_cost(num_layers):.4f}")
        logger.info(f"Max possible cost: {cost_model.max_cost(num_layers):.4f}")

        # Start with most expensive allocation
        start_allocation = cost_model.max_cost_allocation(num_layers)
        start_cost = cost_model.allocation_cost(start_allocation)

        logger.info(f"Starting allocation: {start_allocation}")
        logger.info(f"Starting cost: {start_cost:.4f}")

        # Check if most expensive allocation meets target
        start_assignments = self._allocation_to_assignment(start_allocation, num_layers)
        start_placement = LayerPlacement(assignments=start_assignments)
        start_score = await problem.evaluate(start_placement)
        self.eval_count += 1

        logger.info(f"Initial performance: {start_score:.4f}")

        if start_score < performance_target:
            logger.warning(
                f"Initial configuration doesn't meet target! "
                f"Score {start_score:.4f} < target {performance_target:.4f}"
            )
            # Return infeasible result
            return self._create_result(
                best_entry=BeamEntry(
                    assignments=start_assignments,
                    allocation=start_allocation,
                    cost=start_cost,
                    score=start_score,
                ),
                mode="min_cost",
                constraint_value=performance_target,
                allocations_explored=1,
            )

        # Track best feasible (lowest cost that meets target)
        best_entry = BeamEntry(
            assignments=start_assignments,
            allocation=start_allocation,
            cost=start_cost,
            score=start_score,
        )
        allocations_explored = 0

        # Iterate through allocations from expensive to cheap (decreasing cost)
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

            # Check if this allocation meets the target
            if best_in_allocation.score >= performance_target:
                # This is a feasible solution - update best if lower cost
                if best_in_allocation.cost < best_entry.cost:
                    best_entry = best_in_allocation
                    logger.info(
                        f"  NEW BEST: cost={best_entry.cost:.4f}, "
                        f"score={best_entry.score:.4f}"
                    )
            else:
                # Best we can do at this allocation doesn't meet target
                # Since we're going from expensive to cheap, stop here
                logger.info(
                    f"  Best score {best_in_allocation.score:.4f} < target "
                    f"{performance_target:.4f}, stopping search"
                )
                break

            # Check for max allocations limit
            if allocations_explored >= self.config.max_allocations:
                logger.info(f"Reached max allocations limit: {self.config.max_allocations}")
                break

        return self._create_result(
            best_entry=best_entry,
            mode="min_cost",
            constraint_value=performance_target,
            allocations_explored=allocations_explored,
        )

    async def _search_within_allocation(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        allocation: Allocation,
        num_layers: int,
    ) -> BeamEntry | None:
        """
        Search for the best assignment within a fixed allocation.

        Uses beam search to explore different ways of assigning layers to mixers
        while maintaining the same allocation (counts per mixer).

        Args:
            problem: The optimization problem.
            cost_model: Cost model for cost calculations.
            allocation: Fixed allocation (counts per mixer).
            num_layers: Total number of layers.

        Returns:
            Best BeamEntry found, or None if no valid assignments exist.
        """
        alloc_cost = cost_model.allocation_cost(allocation)

        # Generate initial assignment (canonical assignment for this allocation)
        initial_assignments = self._allocation_to_assignment(allocation, num_layers)
        initial_placement = LayerPlacement(assignments=initial_assignments)
        initial_score = await problem.evaluate(initial_placement)
        self.eval_count += 1

        beam = [
            BeamEntry(
                assignments=initial_assignments,
                allocation=allocation,
                cost=alloc_cost,
                score=initial_score,
            )
        ]

        # Run beam search within this allocation
        for iteration in range(self.config.max_iterations_per_allocation):
            # Generate swap candidates (swaps that preserve allocation)
            all_candidates: list[BeamEntry] = []

            for entry in beam:
                candidates = self._generate_allocation_preserving_swaps(
                    entry=entry,
                    cost_model=cost_model,
                    num_layers=num_layers,
                )
                all_candidates.extend(candidates)

            if not all_candidates:
                break

            # Deduplicate
            unique_candidates = self._deduplicate_candidates(all_candidates)

            # Evaluate candidates
            evaluated: list[BeamEntry] = []
            for candidate in unique_candidates:
                if candidate.score is None:
                    placement = LayerPlacement(assignments=candidate.assignments)
                    score = await problem.evaluate(placement)
                    self.eval_count += 1
                    evaluated.append(
                        BeamEntry(
                            assignments=candidate.assignments,
                            allocation=candidate.allocation,
                            cost=candidate.cost,
                            score=score,
                        )
                    )
                else:
                    evaluated.append(candidate)

            # Sort by score (higher is better)
            evaluated.sort(key=lambda e: e.score, reverse=True)

            # Update beam
            new_beam = evaluated[: self.config.beam_width]

            # Check for convergence
            if beam[0].score >= new_beam[0].score:
                # No improvement
                break

            beam = new_beam

        # Return best entry found
        return max(beam, key=lambda e: e.score) if beam else None

    def _generate_allocation_preserving_swaps(
        self,
        entry: BeamEntry,
        cost_model: BaseCostModel,
        num_layers: int,
    ) -> list[BeamEntry]:
        """
        Generate candidate swaps that preserve the allocation.

        A swap exchanges the mixers of two layers, which preserves counts.
        We only generate swaps where the two layers have different mixers.
        """
        candidates = []
        num_mixers = cost_model.num_mixers

        # Group layers by their mixer
        layers_by_mixer: list[list[int]] = [[] for _ in range(num_mixers)]
        for layer_idx, mixer in enumerate(entry.assignments):
            layers_by_mixer[mixer].append(layer_idx)

        # Generate swaps between different mixers
        for mixer_a in range(num_mixers):
            for mixer_b in range(mixer_a + 1, num_mixers):
                for layer_a in layers_by_mixer[mixer_a]:
                    for layer_b in layers_by_mixer[mixer_b]:
                        # Swap layer_a and layer_b's mixers
                        new_assignments = list(entry.assignments)
                        new_assignments[layer_a] = mixer_b
                        new_assignments[layer_b] = mixer_a

                        candidates.append(
                            BeamEntry(
                                assignments=tuple(new_assignments),
                                allocation=entry.allocation,
                                cost=entry.cost,  # Cost unchanged
                                score=None,
                            )
                        )

        return candidates

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

        Args:
            allocation: Counts per mixer.
            num_layers: Total number of layers.

        Returns:
            Assignment tuple (length = num_layers).
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
        """
        Convert an assignment to its allocation.

        Args:
            assignments: Layer-to-mixer mapping.
            num_mixers: Number of mixer types.

        Returns:
            Allocation tuple (counts per mixer).
        """
        counts = [0] * num_mixers
        for mixer in assignments:
            counts[mixer] += 1
        return tuple(counts)

    def _deduplicate_candidates(
        self, candidates: list[BeamEntry]
    ) -> list[BeamEntry]:
        """Remove duplicate candidates by assignment tuple."""
        seen: set[tuple[int, ...]] = set()
        unique = []

        for candidate in candidates:
            if candidate.assignments not in seen:
                seen.add(candidate.assignments)
                unique.append(candidate)

        return unique

    def _log_config(
        self,
        config: CostConstrainedProblemConfig,
        cost_model: BaseCostModel,
        num_layers: int,
        num_mixers: int,
    ) -> None:
        """Log search configuration."""
        logger.info("\n" + "=" * 60)
        logger.info("Cost-Constrained Beam Search Configuration")
        logger.info("=" * 60)
        logger.info(f"Total layers: {num_layers}")
        logger.info(f"Number of mixer types: {num_mixers}")
        logger.info(f"Mixer costs: {cost_model.mixer_costs}")
        logger.info(f"Min possible cost: {cost_model.min_cost(num_layers):.4f}")
        logger.info(f"Max possible cost: {cost_model.max_cost(num_layers):.4f}")
        logger.info(f"Beam width: {self.config.beam_width}")
        logger.info(f"Max allocations: {self.config.max_allocations}")
        logger.info(f"Max iterations per allocation: {self.config.max_iterations_per_allocation}")
        logger.info(f"Optimization mode: {config.problem_type}")

    def _create_result(
        self,
        best_entry: BeamEntry,
        mode: Literal["max_performance", "min_cost"],
        constraint_value: float,
        allocations_explored: int,
    ) -> CostConstrainedSearchResult:
        """Create search result from best entry."""
        logger.info("\n" + "=" * 60)
        logger.info("SEARCH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best cost: {best_entry.cost:.4f}")
        logger.info(f"Best performance: {best_entry.score:.4f}")
        logger.info(f"Total evaluations: {self.eval_count}")
        logger.info(f"Allocations explored: {allocations_explored}")

        return CostConstrainedSearchResult(
            best_parameters=LayerPlacement(assignments=best_entry.assignments),
            best_score=best_entry.score,
            total_evaluations=self.eval_count,
            final_cost=best_entry.cost,
            mode=mode,
            constraint_value=constraint_value,
            allocations_explored=allocations_explored,
        )


class CostConstrainedBeamSearchConfig(BaseSearchStrategyConfig):
    """Configuration for cost-constrained beam search.

    This search operates on two levels:

    1. **Outer loop**: Navigate through allocations (counts per mixer) in cost order.
       Controlled by `max_allocations`.

    2. **Inner loop**: For each allocation, use beam search to find the best
       layer assignment. Controlled by `beam_width` and `max_iterations_per_allocation`.
    """

    _impl_class: ClassVar[type[BaseSearchStrategy]] = CostConstrainedBeamSearch

    strategy_type: Literal["cost_constrained_beam"] = "cost_constrained_beam"

    beam_width: int = Field(
        description="Number of top candidates to keep during within-allocation search",
        gt=0,
    )

    max_allocations: int = Field(
        description="Maximum number of allocations to explore",
        gt=0,
    )

    max_iterations_per_allocation: int = Field(
        description="Maximum beam search iterations per allocation",
        gt=0,
    )
