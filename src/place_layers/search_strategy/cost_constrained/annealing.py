"""Simulated annealing for cost-constrained optimization.

This implements an inner loop strategy that uses simulated annealing to explore
the assignment space within a fixed allocation.

Key features:
- Ruin-and-recreate move operator: shuffles k positions (generalizes k-cycles)
- Multi-start: multiple random restarts explore different basins
- Adaptive reheating: reheats when acceptance rate drops below threshold
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from typing import TYPE_CHECKING, ClassVar, Literal, Sequence

from pydantic import Field

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.problem.layer_placement import CostConstrainedProblem
from place_layers.search_strategy.cost_constrained.base import BaseCostConstrainedSearch
from place_layers.search_strategy.cost_constrained.base_config import (
    BaseCostConstrainedConfig,
)
from place_layers.search_strategy.cost_constrained.moves import (
    contiguous_ruin,
    generate_move,
    random_assignment,
    ruin_and_recreate,
    sample_perturbation_size,
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
        self._rng = random.Random(self.config.seed)

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
        from place_layers.search_strategy.progress.events import (
            AnnealingCompletedEvent,
            AnnealingRestartEvent,
        )

        max_steps = self.config.max_steps

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
                    final_temperature=self.config.initial_temperature,
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

        # Multi-start SA
        num_restarts = self.config.num_restarts
        steps_per_restart = max_steps // num_restarts

        best_entry: BeamEntry | None = None
        best_restart = 0
        total_reheats = 0
        total_steps = 0

        for restart in range(num_restarts):
            initial = self._random_initial_assignment(allocation, num_layers)

            # Emit restart event
            if self.observers:
                restart_event = AnnealingRestartEvent(
                    restart_index=restart,
                    num_restarts=num_restarts,
                    steps_budget=steps_per_restart,
                    initial_assignments=tuple(initial),
                )
                for observer in self.observers:
                    observer.on_annealing_restart(restart_event)

            entry, reheats, steps_taken = await self._run_sa_trajectory(
                problem=problem,
                allocation=allocation,
                initial=initial,
                max_steps=steps_per_restart,
                num_layers=num_layers,
                restart_index=restart,
            )
            total_reheats += reheats
            total_steps += steps_taken

            if best_entry is None or entry.score > best_entry.score:  # type: ignore[operator]
                best_entry = entry
                best_restart = restart

        # Emit completed event
        if self.observers and best_entry is not None:
            event = AnnealingCompletedEvent(
                steps_taken=total_steps,
                final_temperature=self.config.min_temperature,
                termination_reason="max_steps",
                best_assignments=best_entry.assignments,
                best_score=best_entry.score,  # type: ignore[arg-type]
                restarts_completed=num_restarts,
                best_restart=best_restart,
                total_reheats=total_reheats,
            )
            for observer in self.observers:
                observer.on_annealing_completed(event)

        return [best_entry] if best_entry is not None else []

    async def _run_sa_trajectory(
        self,
        problem: CostConstrainedProblem,
        allocation: Allocation,
        initial: list[str],
        max_steps: int,
        num_layers: int,
        restart_index: int,
    ) -> tuple[BeamEntry, int, int]:
        """
        Run a single SA trajectory from an initial assignment.

        Returns:
            Tuple of (best_entry, reheats_count, steps_taken).
        """
        from place_layers.search_strategy.progress.events import (
            AnnealingReheatEvent,
            AnnealingStepEvent,
        )

        # Evaluate initial assignment
        current = initial.copy()
        placement = LayerPlacement(assignments=tuple(current))
        current_score = await problem.evaluate(placement)
        self.eval_count += 1

        for observer in self.observers:
            observer.on_eval_completed(self.eval_count)

        best = current.copy()
        best_score = current_score

        # Initialize temperature
        temperature = self.config.initial_temperature
        reheats = 0
        steps_taken = 0

        # Acceptance tracking for adaptive reheating
        accept_window: deque[bool] = deque(maxlen=self.config.reheat_window)

        for step in range(max_steps):
            # Sample perturbation size
            k = self._sample_perturbation_size(num_layers, temperature)

            # Generate move
            proposed, move_type = self._generate_move(current, k)

            # No-op detection: if proposed == current, skip evaluation
            if proposed == current:
                steps_taken = step + 1
                temperature *= self.config.cooling_rate
                if temperature < self.config.min_temperature:
                    break
                continue

            # Evaluate proposed
            placement = LayerPlacement(assignments=tuple(proposed))
            proposed_score = await problem.evaluate(placement)
            self.eval_count += 1

            for observer in self.observers:
                observer.on_eval_completed(self.eval_count)

            # Metropolis criterion
            delta = proposed_score - current_score

            if delta > 0:
                acceptance_prob = 1.0
                accepted = True
            elif temperature > 0:
                acceptance_prob = math.exp(delta / temperature)
                accepted = self._rng.random() < acceptance_prob
            else:
                acceptance_prob = 0.0
                accepted = False

            # Track acceptance for reheating
            accept_window.append(accepted)

            if accepted:
                current = proposed
                current_score = proposed_score
                if proposed_score > best_score:
                    best = current.copy()
                    best_score = proposed_score

            steps_taken = step + 1

            # Emit step event
            if self.observers:
                step_event = AnnealingStepEvent(
                    step=step,
                    max_steps=max_steps,
                    perturbation_size=k,
                    move_type=move_type,
                    restart_index=restart_index,
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
                    observer.on_annealing_step(step_event)

            # Cool down
            temperature *= self.config.cooling_rate

            # Adaptive reheating check
            if len(accept_window) == accept_window.maxlen:
                # Check acceptance rate over the window
                acceptance_rate = sum(accept_window) / len(accept_window)
                if acceptance_rate < self.config.reheat_threshold:
                    old_temp = temperature
                    temperature = self.config.initial_temperature * self.config.reheat_fraction
                    reheats += 1
                    accept_window.clear()

                    if self.observers:
                        reheat_event = AnnealingReheatEvent(
                            restart_index=restart_index,
                            step=step,
                            old_temperature=old_temp,
                            new_temperature=temperature,
                            acceptance_rate=acceptance_rate,
                        )
                        for observer in self.observers:
                            observer.on_annealing_reheat(reheat_event)

            # Early termination if temperature is too low
            if temperature < self.config.min_temperature:
                break

        return (
            BeamEntry(
                assignments=tuple(best),
                allocation=allocation,
                score=best_score,
            ),
            reheats,
            steps_taken,
        )

    @staticmethod
    def _can_swap(allocation: Allocation) -> bool:
        """Check if any allocation-preserving move is possible.

        Requires at least 2 mixer types with non-zero counts.
        """
        non_zero = sum(1 for count in allocation.values() if count > 0)
        return non_zero >= 2

    def _random_initial_assignment(
        self, allocation: Allocation, num_layers: int
    ) -> list[str]:
        """Build a random initial assignment from allocation counts."""
        return random_assignment(self._rng, allocation, num_layers)

    def _sample_perturbation_size(
        self,
        num_layers: int,
        temperature: float,
    ) -> int:
        """Sample perturbation size using instance config for max_ruin_fraction."""
        return sample_perturbation_size(
            self._rng,
            num_layers,
            temperature,
            self.config.initial_temperature,
            self.config.max_ruin_fraction,
        )

    def _generate_move(
        self, assignments: list[str], k: int
    ) -> tuple[list[str], Literal["random_ruin", "contiguous_ruin"]]:
        """Generate a ruin-and-recreate move."""
        return generate_move(
            self._rng, assignments, k, self.config.contiguous_ruin_probability
        )

    def _ruin_and_recreate(self, assignments: list[str], k: int) -> list[str]:
        """Ruin-and-recreate: sample k random positions, shuffle their types."""
        return ruin_and_recreate(self._rng, assignments, k)

    def _contiguous_ruin(self, assignments: list[str], k: int) -> list[str]:
        """Contiguous ruin: shuffle a contiguous block of k positions."""
        return contiguous_ruin(self._rng, assignments, k)


class AnnealingConfig(BaseCostConstrainedConfig):
    """Configuration for simulated annealing inner loop."""

    _impl_class: ClassVar[type["BaseSearchStrategy"]] = AnnealingSearch

    strategy_type: Literal["cost_constrained_annealing"] = "cost_constrained_annealing"

    # --- Core SA ---
    max_steps: int = Field(
        description="Maximum number of annealing steps per allocation (split among restarts)",
        gt=0,
    )

    num_restarts: int = Field(
        description="Number of random restarts. Total steps split evenly among restarts.",
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

    # --- Move operator ---
    max_ruin_fraction: float = Field(
        description="Max fraction of layers ruined at full temperature.",
        gt=0,
        le=1,
    )

    contiguous_ruin_probability: float = Field(
        description="Probability of contiguous block ruin vs random scattered ruin.",
        ge=0,
        le=1,
    )

    # --- Adaptive reheating ---
    reheat_fraction: float = Field(
        description="Reheat temperature = initial_temperature * reheat_fraction.",
        gt=0,
        lt=1,
    )

    reheat_threshold: float = Field(
        description="Acceptance rate below this triggers adaptive reheating.",
        gt=0,
        lt=1,
    )

    reheat_window: int = Field(
        description="Sliding window size for acceptance rate tracking.",
        gt=0,
    )

    # --- Reproducibility ---
    seed: int | None = Field(
        description="Random seed for reproducibility. If None, uses system entropy.",
    )
