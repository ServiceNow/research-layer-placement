"""Simulated annealing for cost-constrained optimization.

This implements an inner loop strategy that uses simulated annealing to explore
the assignment space within a fixed allocation.
"""

from __future__ import annotations

import logging
import math
import random
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

    def __init__(self, config: "AnnealingConfig") -> None:
        super().__init__(config)
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
    ) -> BeamEntry | None:
        """
        Search for the best assignment within a fixed allocation using simulated annealing.

        Performs a random walk through assignment space, accepting moves based
        on the Metropolis criterion.
        """
        alloc_cost = cost_model.allocation_cost(allocation)
        mixer_names = problem.evaluator.mixer_names

        # Group layers by mixer for efficient swap generation
        layers_by_mixer = self._group_layers_by_mixer(allocation, mixer_names)

        # Check if any swaps are possible
        if not self._can_swap(layers_by_mixer):
            # All layers on one mixer - just return canonical
            assignments = self._allocation_to_assignment(allocation, num_layers, mixer_names)
            placement = LayerPlacement(assignments=assignments)
            score = await problem.evaluate(placement)
            self.eval_count += 1
            return BeamEntry(
                assignments=assignments,
                allocation=allocation,
                cost=alloc_cost,
                score=score,
            )

        # Initialize with canonical assignment
        current = list(self._allocation_to_assignment(allocation, num_layers, mixer_names))
        placement = LayerPlacement(assignments=current)
        current_score = await problem.evaluate(placement)
        self.eval_count += 1

        best = current.copy()
        best_score = current_score

        # Initialize temperature
        temperature = self.config.initial_temperature

        # Main annealing loop
        for step in range(self.config.max_steps):
            # Propose a random swap
            layer_a, layer_b = self._random_swap(current, mixer_names)
            if layer_a is None:
                # No valid swap possible (shouldn't happen if can_swap returned True)
                break

            # Apply swap
            current[layer_a], current[layer_b] = current[layer_b], current[layer_a]

            # Evaluate
            placement = LayerPlacement(assignments=current)
            new_score = await problem.evaluate(placement)
            self.eval_count += 1

            # Metropolis criterion
            delta = new_score - current_score

            if delta > 0:
                # Better: always accept
                current_score = new_score
                if new_score > best_score:
                    best = current.copy()
                    best_score = new_score
            elif temperature > 0 and random.random() < math.exp(delta / temperature):
                # Worse but accepted with probability
                current_score = new_score
            else:
                # Rejected: undo swap
                current[layer_a], current[layer_b] = current[layer_b], current[layer_a]

            # Cool down
            temperature *= self.config.cooling_rate

            # Early termination if temperature is too low
            if temperature < self.config.min_temperature:
                break

        return BeamEntry(
            assignments=best,
            allocation=allocation,
            cost=alloc_cost,
            score=best_score,
        )

    def _group_layers_by_mixer(
        self, allocation: Allocation, mixer_names: list[str]
    ) -> dict[str, list[int]]:
        """
        Create a canonical grouping of layers by mixer name.

        This is used for efficient random swap generation.
        """
        layers_by_mixer: dict[str, list[int]] = {}
        layer_idx = 0
        for i, count in enumerate(allocation):
            mixer_name = mixer_names[i]
            layers_by_mixer[mixer_name] = list(range(layer_idx, layer_idx + count))
            layer_idx += count
        return layers_by_mixer

    def _can_swap(self, layers_by_mixer: dict[str, list[int]]) -> bool:
        """Check if any allocation-preserving swap is possible."""
        non_empty = [group for group in layers_by_mixer.values() if len(group) > 0]
        return len(non_empty) >= 2

    def _random_swap(
        self, assignments: list[str], mixer_names: list[str]
    ) -> tuple[int | None, int | None]:
        """
        Generate a random allocation-preserving swap.

        Returns (layer_a, layer_b) where the two layers have different mixers.
        Returns (None, None) if no swap is possible.
        """
        # Group current layers by their mixer name
        layers_by_mixer: dict[str, list[int]] = {name: [] for name in mixer_names}
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
