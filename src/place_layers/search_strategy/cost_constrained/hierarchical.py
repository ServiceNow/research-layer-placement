"""Hierarchical beam search for cost-constrained optimization.

This implements an inner loop strategy that uses the hierarchical beam search
approach: phase-by-phase assignment of layers to mixers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.search_strategy.cost_constrained.base import BaseCostConstrainedSearch
from place_layers.search_strategy.cost_constrained.base_config import (
    BaseCostConstrainedConfig,
)
from place_layers.search_strategy.cost_constrained.types import BeamEntry

if TYPE_CHECKING:
    from place_layers.search_strategy.base_search_strategy import BaseSearchStrategy


logger = logging.getLogger(__name__)


class HierarchicalSearch(BaseCostConstrainedSearch):
    """
    Cost-constrained search using hierarchical beam search for the inner loop.

    Within each allocation, uses the hierarchical approach:
    - Phase 1: Find best layers for mixer 0 (e.g., FA)
    - Phase 2: Given fixed mixer 0 positions, find best layers for mixer 1 (e.g., SWA)
    - Phase 3: Remaining layers get mixer 2 (e.g., GDN)

    This is much faster than swap-based search (~10x fewer evaluations) but
    has a different search bias - it cannot reconsider early phase decisions.

    Key difference from standalone HierarchicalBeamSearch:
    - This is an inner loop within cost-constrained search
    - Allocation is fixed (counts per mixer already determined)
    - Only assignment (which layers) needs to be found
    """

    def __init__(self, config: "HierarchicalConfig") -> None:
        super().__init__(config)
        self.config: HierarchicalConfig

    async def _search_within_allocation(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        allocation: Allocation,
        num_layers: int,
    ) -> BeamEntry | None:
        """
        Search for the best assignment within a fixed allocation using hierarchical beam search.

        Runs phases sequentially, each phase finding the best layers for one mixer type.
        """
        alloc_cost = cost_model.allocation_cost(allocation)
        num_mixers = len(allocation)

        # Track which layers are assigned and to what mixer
        fixed_assignments: dict[int, int] = {}
        unassigned = list(range(num_layers))

        # Extract budgets (counts per mixer, excluding the last one which gets remainder)
        budgets = list(allocation[:-1])  # e.g., [1, 17] from (1, 17, 30)

        # Run phases for each mixer type except the last
        for phase_idx, budget in enumerate(budgets):
            mixer_idx = phase_idx
            fallback_mixer = phase_idx + 1  # Unassigned go to next mixer during evaluation

            if budget == 0:
                # Skip phases with no layers to assign
                continue

            if not unassigned:
                # No more layers to assign
                break

            # Find best layers for this mixer
            best_layers = await self._run_phase(
                problem=problem,
                budget=budget,
                mixer_idx=mixer_idx,
                fallback_mixer=fallback_mixer,
                unassigned=unassigned,
                fixed_assignments=fixed_assignments,
                num_layers=num_layers,
            )

            # Lock in the assignments
            for layer_idx in best_layers:
                fixed_assignments[layer_idx] = mixer_idx
                unassigned.remove(layer_idx)

        # Assign remaining layers to the last mixer
        last_mixer = num_mixers - 1
        for layer_idx in unassigned:
            fixed_assignments[layer_idx] = last_mixer

        # Build final assignment tuple
        assignments = tuple(fixed_assignments[i] for i in range(num_layers))

        # Final evaluation
        placement = LayerPlacement(assignments=assignments)
        final_score = await problem.evaluate(placement)
        self.eval_count += 1

        return BeamEntry(
            assignments=assignments,
            allocation=allocation,
            cost=alloc_cost,
            score=final_score,
        )

    async def _run_phase(
        self,
        problem: CostConstrainedProblem,
        budget: int,
        mixer_idx: int,
        fallback_mixer: int,
        unassigned: list[int],
        fixed_assignments: dict[int, int],
        num_layers: int,
    ) -> list[int]:
        """
        Run one phase: find best 'budget' layers for mixer_idx.

        Returns the list of layer indices to assign to this mixer.
        """
        # Pre-score individual layers
        layer_scores = await self._prescore_layers(
            problem=problem,
            unassigned=unassigned,
            mixer_idx=mixer_idx,
            fallback_mixer=fallback_mixer,
            fixed_assignments=fixed_assignments,
            num_layers=num_layers,
        )

        if budget == 1:
            # Simple case: just take the best single layer
            return [layer_scores[0][0]]

        # Grow beam from best single layers to budget size
        beam = await self._grow_beam(
            problem=problem,
            layer_scores=layer_scores,
            budget=budget,
            mixer_idx=mixer_idx,
            fallback_mixer=fallback_mixer,
            unassigned=unassigned,
            fixed_assignments=fixed_assignments,
            num_layers=num_layers,
        )

        return beam[0]["layers"]

    async def _prescore_layers(
        self,
        problem: CostConstrainedProblem,
        unassigned: list[int],
        mixer_idx: int,
        fallback_mixer: int,
        fixed_assignments: dict[int, int],
        num_layers: int,
    ) -> list[tuple[int, float]]:
        """Pre-score individual layers to seed the beam."""
        layer_scores = []

        for layer_idx in unassigned:
            assignment = self._create_assignment(
                fixed_assignments=fixed_assignments,
                target_layers=[layer_idx],
                target_mixer_idx=mixer_idx,
                unassigned=unassigned,
                fallback_mixer_idx=fallback_mixer,
                num_layers=num_layers,
            )
            placement = LayerPlacement(assignments=assignment)
            score = await problem.evaluate(placement)
            self.eval_count += 1
            layer_scores.append((layer_idx, score))

        # Sort by score (higher is better)
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        return layer_scores

    async def _grow_beam(
        self,
        problem: CostConstrainedProblem,
        layer_scores: list[tuple[int, float]],
        budget: int,
        mixer_idx: int,
        fallback_mixer: int,
        unassigned: list[int],
        fixed_assignments: dict[int, int],
        num_layers: int,
    ) -> list[dict]:
        """Grow the beam from seed to budget size."""
        # Initialize beam with single-layer candidates
        beam = [
            {"layers": [layer_idx], "score": score}
            for layer_idx, score in layer_scores[: self.config.beam_width]
        ]

        best_score = beam[0]["score"]

        # Grow beam iteratively
        for growth_step in range(1, budget):
            # Generate candidates
            candidates = self._generate_candidates(beam, unassigned)

            if not candidates:
                break

            # Evaluate candidates
            await self._evaluate_candidates(
                problem=problem,
                candidates=candidates,
                mixer_idx=mixer_idx,
                fallback_mixer=fallback_mixer,
                unassigned=unassigned,
                fixed_assignments=fixed_assignments,
                num_layers=num_layers,
            )

            # Select top candidates for new beam
            candidates.sort(key=lambda x: x["score"], reverse=True)
            beam = candidates[: self.config.beam_width]

            # Check early stopping
            new_best_score = beam[0]["score"]
            if new_best_score <= best_score + self.config.early_stop_threshold:
                break
            best_score = new_best_score

        return beam

    def _generate_candidates(
        self, beam: list[dict], unassigned: list[int]
    ) -> list[dict]:
        """Generate new candidates by expanding each beam entry."""
        candidates = []
        seen: set[tuple[int, ...]] = set()

        for beam_entry in beam:
            existing = set(beam_entry["layers"])

            for layer_idx in unassigned:
                if layer_idx in existing:
                    continue

                new_layers = tuple(sorted(beam_entry["layers"] + [layer_idx]))

                if new_layers in seen:
                    continue
                seen.add(new_layers)

                candidates.append({"layers": list(new_layers), "score": None})

        return candidates

    async def _evaluate_candidates(
        self,
        problem: CostConstrainedProblem,
        candidates: list[dict],
        mixer_idx: int,
        fallback_mixer: int,
        unassigned: list[int],
        fixed_assignments: dict[int, int],
        num_layers: int,
    ) -> None:
        """Evaluate all candidates and store scores."""
        for candidate in candidates:
            assignment = self._create_assignment(
                fixed_assignments=fixed_assignments,
                target_layers=candidate["layers"],
                target_mixer_idx=mixer_idx,
                unassigned=unassigned,
                fallback_mixer_idx=fallback_mixer,
                num_layers=num_layers,
            )
            placement = LayerPlacement(assignments=assignment)
            candidate["score"] = await problem.evaluate(placement)
            self.eval_count += 1

    def _create_assignment(
        self,
        fixed_assignments: dict[int, int],
        target_layers: list[int],
        target_mixer_idx: int,
        unassigned: list[int],
        fallback_mixer_idx: int,
        num_layers: int,
    ) -> tuple[int, ...]:
        """Create a complete assignment for evaluation."""
        assignment = dict(fixed_assignments)

        # Assign target layers to target mixer
        for layer_idx in target_layers:
            assignment[layer_idx] = target_mixer_idx

        # Assign unassigned layers to fallback
        for layer_idx in unassigned:
            if layer_idx not in assignment:
                assignment[layer_idx] = fallback_mixer_idx

        # Convert dict to tuple
        return tuple(assignment[i] for i in range(num_layers))


class HierarchicalConfig(BaseCostConstrainedConfig):
    """Configuration for hierarchical beam search inner loop."""

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = HierarchicalSearch

    strategy_type: Literal["cost_constrained_hierarchical"] = "cost_constrained_hierarchical"

    beam_width: int = Field(
        description="Number of top candidates to keep at each growth step",
        gt=0,
    )

    early_stop_threshold: float = Field(
        description="Stop growing beam if best score improvement is below this threshold",
        ge=0.0,
    )
