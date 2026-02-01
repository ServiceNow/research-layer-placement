"""Swap-based beam search for cost-constrained optimization.

This implements the original inner loop strategy that uses allocation-preserving
swaps to explore the assignment space within a fixed allocation.
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
    from place_layers.search_strategy.progress import SearchObserver
    from place_layers.search_strategy.progress.estimator import SearchProgressEstimator


logger = logging.getLogger(__name__)


class SwapBeamSearch(BaseCostConstrainedSearch):
    """
    Cost-constrained search using swap-based beam search for the inner loop.

    Within each allocation, explores assignments using allocation-preserving swaps.
    A swap exchanges the mixers of two layers, preserving the overall counts.

    This is the original algorithm with thorough exploration but high evaluation cost.
    Evaluation complexity is O(beam_width × iterations × swaps_per_state).
    """

    def __init__(
        self,
        config: "SwapBeamConfig",
        observer: "SearchObserver | None" = None,
        progress_estimator: "SearchProgressEstimator | None" = None,
    ) -> None:
        super().__init__(config, observer, progress_estimator)
        self.config: SwapBeamConfig

        if self.observer is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support progress observers."
            )

    async def _search_within_allocation(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        allocation: Allocation,
        num_layers: int,
    ) -> BeamEntry | None:
        """
        Search for the best assignment within a fixed allocation using swap-based beam search.

        Uses beam search to explore different ways of assigning layers to mixers
        while maintaining the same allocation (counts per mixer).
        """
        alloc_cost = cost_model.allocation_cost(allocation)
        mixer_names = problem.evaluator.mixer_names

        # Generate initial assignment (canonical assignment for this allocation)
        initial_assignments = self._allocation_to_assignment(allocation, num_layers, mixer_names)
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
                    mixer_names=mixer_names,
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
        mixer_names: list[str],
    ) -> list[BeamEntry]:
        """
        Generate candidate swaps that preserve the allocation.

        A swap exchanges the mixers of two layers, which preserves counts.
        We only generate swaps where the two layers have different mixers.
        """
        candidates = []

        # Group layers by their mixer name
        layers_by_mixer: dict[str, list[int]] = {name: [] for name in mixer_names}
        for layer_idx, mixer_name in enumerate(entry.assignments):
            layers_by_mixer[mixer_name].append(layer_idx)

        # Generate swaps between different mixers
        mixer_list = list(mixer_names)
        for i, mixer_a in enumerate(mixer_list):
            for mixer_b in mixer_list[i + 1:]:
                for layer_a in layers_by_mixer[mixer_a]:
                    for layer_b in layers_by_mixer[mixer_b]:
                        # Swap layer_a and layer_b's mixers
                        new_assignments = list(entry.assignments)
                        new_assignments[layer_a] = mixer_b
                        new_assignments[layer_b] = mixer_a

                        candidates.append(
                            BeamEntry(
                                assignments=new_assignments,
                                allocation=entry.allocation,
                                cost=entry.cost,  # Cost unchanged
                                score=None,
                            )
                        )

        return candidates

    def _deduplicate_candidates(
        self, candidates: list[BeamEntry]
    ) -> list[BeamEntry]:
        """Remove duplicate candidates by assignment."""
        seen: set[tuple[str, ...]] = set()
        unique = []

        for candidate in candidates:
            key = tuple(candidate.assignments)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)

        return unique


class SwapBeamConfig(BaseCostConstrainedConfig):
    """Configuration for swap-based beam search."""

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = SwapBeamSearch

    strategy_type: Literal["cost_constrained_swap_beam"] = "cost_constrained_swap_beam"

    beam_width: int = Field(
        description="Number of top candidates to keep in beam",
        gt=0,
    )

    max_iterations_per_allocation: int = Field(
        description="Maximum iterations of swap-based search per allocation",
        gt=0,
    )
