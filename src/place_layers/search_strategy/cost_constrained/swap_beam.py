"""Swap-based beam search for cost-constrained optimization.

This implements the original inner loop strategy that uses allocation-preserving
swaps to explore the assignment space within a fixed allocation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Literal, Sequence

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
        observers: "Sequence[SearchObserver]" = (),
        progress_estimator: "SearchProgressEstimator | None" = None,
    ) -> None:
        super().__init__(config, observers, progress_estimator)
        self.config: SwapBeamConfig

    async def _search_within_allocation(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        allocation: Allocation,
        num_layers: int,
    ) -> list[BeamEntry]:
        """
        Search for the best assignment within a fixed allocation using swap-based beam search.

        Uses beam search to explore different ways of assigning layers to mixers
        while maintaining the same allocation (counts per mixer).

        Returns:
            List of BeamEntry results sorted by score descending, or empty list
            if no valid assignments exist.
        """
        from place_layers.search_strategy.progress.events import (
            SwapBeamIterationEvent,
            SwapBeamIterationStartedEvent,
            SwapBeamNodeId,
            SwapCandidate,
            SwapEvaluatedEvent,
        )

        alloc_cost = cost_model.allocation_cost(allocation)
        mixer_names = problem.evaluator.mixer_names
        max_iterations = self.config.max_iterations_per_allocation

        # Generate initial assignment (canonical assignment for this allocation)
        initial_assignments = self._allocation_to_assignment(allocation, num_layers, mixer_names)
        initial_placement = LayerPlacement(assignments=tuple(initial_assignments))
        initial_score = await problem.evaluate(initial_placement)
        self.eval_count += 1

        # Notify observers of eval completion
        for observer in self.observers:
            observer.on_eval_completed(self.eval_count)

        # Create initial beam entry and swap candidate
        initial_id = SwapBeamNodeId(allocation=allocation, iteration=0, seq=0)
        initial_candidate = SwapCandidate(
            id=initial_id,
            parent_id=None,
            swap=(-1, -1),  # No swap for initial
            assignments=tuple(initial_assignments),
            score=initial_score,
        )

        beam = [
            BeamEntry(
                assignments=tuple(initial_assignments),
                allocation=allocation,
                cost=alloc_cost,
                score=initial_score,
            )
        ]
        beam_candidates = [initial_candidate]

        # Emit iteration 0 started event (initial state)
        if self.observers:
            event = SwapBeamIterationStartedEvent(
                iteration=0,
                max_iterations=max_iterations,
                num_candidates=0,  # No candidates to evaluate in iteration 0
                beam=tuple(beam_candidates),
            )
            for observer in self.observers:
                observer.on_swap_beam_iteration_started(event)

        # Run beam search within this allocation
        for iteration in range(1, max_iterations + 1):
            # Generate swap candidates (swaps that preserve allocation)
            all_candidates: list[tuple[BeamEntry, int, tuple[int, int]]] = []  # (entry, parent_seq, swap)

            for parent_seq, entry in enumerate(beam):
                candidates = self._generate_allocation_preserving_swaps(
                    entry=entry,
                    mixer_names=mixer_names,
                )
                for candidate in candidates:
                    # Track parent and the swap that was made
                    swap = self._find_swap(entry.assignments, candidate.assignments)
                    all_candidates.append((candidate, parent_seq, swap))

            if not all_candidates:
                # No swaps possible - emit completion event and break
                if self.observers:
                    event = SwapBeamIterationEvent(
                        iteration=iteration,
                        max_iterations=max_iterations,
                        tried=(),
                        beam=tuple(beam_candidates),
                        best_candidate=beam_candidates[0] if beam_candidates else None,
                        improved=False,
                        converged=True,
                    )
                    for observer in self.observers:
                        observer.on_swap_beam_iteration(event)
                break

            # Deduplicate by assignments
            seen: set[tuple[str, ...]] = set()
            unique_candidates: list[tuple[BeamEntry, int, tuple[int, int]]] = []
            for candidate, parent_seq, swap in all_candidates:
                key = tuple(candidate.assignments)
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append((candidate, parent_seq, swap))

            # Emit iteration started event
            if self.observers:
                event = SwapBeamIterationStartedEvent(
                    iteration=iteration,
                    max_iterations=max_iterations,
                    num_candidates=len(unique_candidates),
                    beam=tuple(beam_candidates),
                )
                for observer in self.observers:
                    observer.on_swap_beam_iteration_started(event)

            # Evaluate candidates
            evaluated: list[BeamEntry] = []
            tried_candidates: list[SwapCandidate] = []
            total_evals = len(unique_candidates)

            for eval_idx, (candidate, parent_seq, swap) in enumerate(unique_candidates):
                if candidate.score is None:
                    placement = LayerPlacement(assignments=candidate.assignments)
                    score = await problem.evaluate(placement)
                    self.eval_count += 1

                    # Notify observers of eval completion
                    for observer in self.observers:
                        observer.on_eval_completed(self.eval_count)

                    evaluated_entry = BeamEntry(
                        assignments=candidate.assignments,
                        allocation=candidate.allocation,
                        cost=candidate.cost,
                        score=score,
                    )
                    evaluated.append(evaluated_entry)

                    # Create SwapCandidate for event
                    parent_id = SwapBeamNodeId(
                        allocation=allocation,
                        iteration=iteration - 1,
                        seq=parent_seq,
                    )
                    candidate_id = SwapBeamNodeId(
                        allocation=allocation,
                        iteration=iteration,
                        seq=len(tried_candidates),
                    )
                    swap_candidate = SwapCandidate(
                        id=candidate_id,
                        parent_id=parent_id,
                        swap=swap,
                        assignments=candidate.assignments,
                        score=score,
                    )
                    tried_candidates.append(swap_candidate)

                    # Emit swap evaluated event
                    if self.observers:
                        event = SwapEvaluatedEvent(
                            iteration=iteration,
                            id=candidate_id,
                            parent_id=parent_id,
                            swap=swap,
                            assignments=candidate.assignments,
                            score=score,
                            eval_index=eval_idx + 1,
                            total_evals=total_evals,
                        )
                        for observer in self.observers:
                            observer.on_swap_evaluated(event)
                else:
                    evaluated.append(candidate)

            # Sort by score (higher is better)
            evaluated.sort(key=lambda e: e.score if e.score is not None else float("-inf"), reverse=True)
            tried_candidates.sort(key=lambda c: c.score, reverse=True)

            # Update beam
            new_beam = evaluated[: self.config.beam_width]

            # Build new beam candidates
            new_beam_candidates: list[SwapCandidate] = []
            for seq, entry in enumerate(new_beam):
                # Find the corresponding tried candidate
                for tc in tried_candidates:
                    if tc.assignments == entry.assignments:
                        new_beam_candidates.append(SwapCandidate(
                            id=SwapBeamNodeId(allocation=allocation, iteration=iteration, seq=seq),
                            parent_id=tc.parent_id,
                            swap=tc.swap,
                            assignments=tc.assignments,
                            score=tc.score,
                        ))
                        break

            # Check for convergence
            new_best = new_beam[0].score if new_beam and new_beam[0].score is not None else float("-inf")
            old_best = beam[0].score if beam and beam[0].score is not None else float("-inf")
            improved = new_best > old_best
            converged = not improved

            # Emit iteration completed event
            if self.observers:
                best = tried_candidates[0] if tried_candidates else None
                event = SwapBeamIterationEvent(
                    iteration=iteration,
                    max_iterations=max_iterations,
                    tried=tuple(tried_candidates),
                    beam=tuple(new_beam_candidates),
                    best_candidate=best,
                    improved=improved,
                    converged=converged,
                )
                for observer in self.observers:
                    observer.on_swap_beam_iteration(event)

            if converged:
                break

            beam = new_beam
            beam_candidates = new_beam_candidates

        # Return all beam entries sorted by score descending
        if not beam:
            return []
        beam.sort(key=lambda e: e.score if e.score is not None else float("-inf"), reverse=True)
        return beam

    def _find_swap(
        self, old_assignments: tuple[str, ...], new_assignments: tuple[str, ...]
    ) -> tuple[int, int]:
        """Find the two layers that were swapped between old and new assignments."""
        changed = []
        for i, (old, new) in enumerate(zip(old_assignments, new_assignments)):
            if old != new:
                changed.append(i)
        if len(changed) == 2:
            return (changed[0], changed[1])
        return (-1, -1)  # No swap or unexpected change

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
                                assignments=tuple(new_assignments),
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
