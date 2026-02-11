"""Simulated annealing for cost-constrained optimization.

This implements an inner loop strategy that uses simulated annealing to explore
the assignment space within a fixed allocation.
"""

from __future__ import annotations

import logging
import math
import random
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


class AnnealingSearch(BaseCostConstrainedSearch):
    """
    Cost-constrained search using simulated annealing for the inner loop.

    Within each allocation, uses simulated annealing:
    1. Start from canonical assignment
    2. Propose random allocation-preserving swaps
    3. Accept improving moves always; accept worsening moves with probability exp(delta/T)
    4. Cool temperature over time

    This provides O(max_steps) evaluations per allocation regardless of the
    allocation size, making it much more efficient than swap-based beam search.

    Trade-off: SA explores randomly rather than systematically, which may miss
    good directions that beam search would find. However, it naturally explores
    multi-hop paths through accumulated accepted moves.
    """

    def __init__(
        self,
        config: "AnnealingConfig",
        observers: "Sequence[SearchObserver]" = (),
        progress_estimator: "SearchProgressEstimator | None" = None,
    ) -> None:
        super().__init__(config, observers, progress_estimator)
        self.config: AnnealingConfig

        # Seed RNG for reproducibility if specified
        if self.config.seed is not None:
            random.seed(self.config.seed)

    async def _search_within_allocation(
        self,
        problem: CostConstrainedProblem,
        cost_model: BaseCostModel,
        allocation: Allocation,
        num_layers: int,
    ) -> list[BeamEntry]:
        """
        Search for the best assignment within a fixed allocation using simulated annealing.

        Performs a random walk through assignment space, accepting moves based
        on the Metropolis criterion.

        Returns:
            List containing a single BeamEntry result (SA only tracks best),
            or empty list if no valid assignments exist.
        """
        from place_layers.search_strategy.progress.events import (
            AnnealingCompletedEvent,
            AnnealingStepEvent,
        )

        max_steps = self.config.max_steps

        # Group layers by mixer for efficient swap generation
        layers_by_mixer = self._group_layers_by_mixer(allocation)

        # Check if any swaps are possible
        if not self._can_swap(layers_by_mixer):
            # All layers on one mixer - just return canonical
            assignments = self._allocation_to_assignment(allocation, num_layers)
            placement = LayerPlacement(assignments=tuple(assignments))
            score, auxiliary = await self._evaluate_with_details(problem, placement)
            self.eval_count += 1

            # Notify observers
            for observer in self.observers:
                observer.on_eval_completed(self.eval_count)

            # Emit completed event with no_swaps reason
            if self.observers:
                event = AnnealingCompletedEvent(
                    steps_taken=0,
                    final_temperature=self.config.initial_temperature,
                    termination_reason="no_swaps",
                    best_assignments=tuple(assignments),
                    best_score=score,
                )
                for observer in self.observers:
                    observer.on_annealing_completed(event)

            return [
                BeamEntry(
                    assignments=tuple(assignments),
                    allocation=allocation,
                    score=score,
                    auxiliary_scores=auxiliary,
                )
            ]

        # Initialize with canonical assignment
        current = list(self._allocation_to_assignment(allocation, num_layers))
        placement = LayerPlacement(assignments=tuple(current))
        current_score, current_auxiliary = await self._evaluate_with_details(problem, placement)
        self.eval_count += 1

        # Notify observers
        for observer in self.observers:
            observer.on_eval_completed(self.eval_count)

        best = current.copy()
        best_score = current_score
        best_auxiliary = current_auxiliary

        # Initialize temperature
        temperature = self.config.initial_temperature
        termination_reason: str = "max_steps"
        steps_taken = 0

        # Main annealing loop
        for step in range(max_steps):
            # Propose a random swap
            layer_a, layer_b = self._random_swap(current, allocation)
            if layer_a is None or layer_b is None:
                # No valid swap possible (shouldn't happen if can_swap returned True)
                termination_reason = "no_swaps"
                break

            # Apply swap to get proposed state
            proposed = current.copy()
            proposed[layer_a], proposed[layer_b] = proposed[layer_b], proposed[layer_a]

            # Evaluate
            placement = LayerPlacement(assignments=tuple(proposed))
            proposed_score, proposed_auxiliary = await self._evaluate_with_details(problem, placement)
            self.eval_count += 1

            # Notify observers
            for observer in self.observers:
                observer.on_eval_completed(self.eval_count)

            # Metropolis criterion
            delta = proposed_score - current_score

            if delta > 0:
                # Better: always accept
                acceptance_prob = 1.0
                accepted = True
            elif temperature > 0:
                acceptance_prob = math.exp(delta / temperature)
                accepted = random.random() < acceptance_prob
            else:
                acceptance_prob = 0.0
                accepted = False

            if accepted:
                current = proposed
                current_score = proposed_score
                current_auxiliary = proposed_auxiliary
                if proposed_score > best_score:
                    best = current.copy()
                    best_score = proposed_score
                    best_auxiliary = proposed_auxiliary

            steps_taken = step + 1

            # Emit step event
            if self.observers:
                event = AnnealingStepEvent(
                    step=step,
                    max_steps=max_steps,
                    swap=(layer_a, layer_b),
                    proposed_assignments=tuple(proposed),
                    proposed_score=proposed_score,
                    temperature=temperature,
                    delta=delta,
                    acceptance_probability=acceptance_prob,
                    accepted=accepted,
                    current_assignments=tuple(current),
                    current_score=current_score,
                    best_assignments=tuple(best),
                    best_score=best_score,
                )
                for observer in self.observers:
                    observer.on_annealing_step(event)

            # Cool down
            temperature *= self.config.cooling_rate

            # Early termination if temperature is too low
            if temperature < self.config.min_temperature:
                termination_reason = "min_temperature"
                break

        # Emit completed event
        if self.observers:
            event = AnnealingCompletedEvent(
                steps_taken=steps_taken,
                final_temperature=temperature,
                termination_reason=termination_reason,  # type: ignore[arg-type]
                best_assignments=tuple(best),
                best_score=best_score,
            )
            for observer in self.observers:
                observer.on_annealing_completed(event)

        return [
            BeamEntry(
                assignments=tuple(best),
                allocation=allocation,
                score=best_score,
                auxiliary_scores=best_auxiliary,
            )
        ]

    def _group_layers_by_mixer(
        self, allocation: Allocation
    ) -> dict[str, list[int]]:
        """
        Create a canonical grouping of layers by mixer name.

        This is used for efficient random swap generation.
        """
        layers_by_mixer: dict[str, list[int]] = {}
        layer_idx = 0
        for mixer_name, count in allocation.items():
            layers_by_mixer[mixer_name] = list(range(layer_idx, layer_idx + count))
            layer_idx += count
        return layers_by_mixer

    def _can_swap(self, layers_by_mixer: dict[str, list[int]]) -> bool:
        """Check if any allocation-preserving swap is possible."""
        non_empty = [group for group in layers_by_mixer.values() if len(group) > 0]
        return len(non_empty) >= 2

    def _random_swap(
        self, assignments: list[str], allocation: Allocation
    ) -> tuple[int | None, int | None]:
        """
        Generate a random allocation-preserving swap.

        Returns (layer_a, layer_b) where the two layers have different mixers.
        Returns (None, None) if no swap is possible.
        """
        # Group current layers by their mixer name
        layers_by_mixer: dict[str, list[int]] = {name: [] for name in allocation}
        for layer_idx, mixer_name in enumerate(assignments):
            layers_by_mixer[mixer_name].append(layer_idx)

        # Find pairs of mixers with non-empty groups
        non_empty_mixers = [name for name, group in layers_by_mixer.items() if len(group) > 0]

        if len(non_empty_mixers) < 2:
            return None, None

        # Pick two random different mixers
        mixer_a, mixer_b = random.sample(non_empty_mixers, 2)

        # Pick random layers from each
        layer_a = random.choice(layers_by_mixer[mixer_a])
        layer_b = random.choice(layers_by_mixer[mixer_b])

        return layer_a, layer_b


class AnnealingConfig(BaseCostConstrainedConfig):
    """Configuration for simulated annealing inner loop."""

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = AnnealingSearch

    strategy_type: Literal["cost_constrained_annealing"] = "cost_constrained_annealing"

    max_steps: int = Field(
        description="Maximum number of annealing steps per allocation",
        gt=0,
    )

    initial_temperature: float = Field(
        description="Initial temperature for annealing",
        gt=0,
    )

    cooling_rate: float = Field(
        description="Temperature decay rate per step (T_new = T * cooling_rate)",
        gt=0,
        lt=1,
    )

    min_temperature: float = Field(
        description="Minimum temperature before early termination",
        ge=0,
    )

    seed: int | None = Field(
        description="Random seed for reproducibility. If None, uses system entropy.",
    )
