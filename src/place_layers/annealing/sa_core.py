"""Core simulated annealing building blocks.

Reusable primitives shared by AnnealingSearch (real evaluator SA),
SurrogateSearch (surrogate-guided SA), and CalibrateSACommand.

The ``SAConfig`` Pydantic model is the single source of truth for SA
hyperparameters.  ``run_multi_start_sa`` is the single async SA loop
used by both strategies.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol, Sequence

from pydantic import BaseModel, Field

from place_layers.annealing.events import (
    AnnealingReheatEvent,
    AnnealingRestartEvent,
    AnnealingStepEvent,
)
from place_layers.annealing.moves import (
    generate_move,
    random_assignment,
    sample_perturbation_size,
    type_mutation,
)


# ============================================================================
# Pydantic SA configuration
# ============================================================================


class ReheatConfig(BaseModel):
    """Adaptive reheating configuration."""

    model_config = {"frozen": True}

    fraction: float = Field(gt=0, lt=1)
    threshold: float = Field(gt=0, lt=1)
    window: int = Field(gt=0)


class SAConfig(BaseModel):
    """Shared SA configuration — used by AnnealingSearch, SurrogateSearch, and CalibrateSA."""

    model_config = {"frozen": True}

    num_restarts: int = Field(gt=0)
    steps_per_restart: int = Field(gt=0)
    initial_temperature: float = Field(gt=0)
    cooling_rate: float = Field(gt=0, lt=1)
    min_temperature: float = Field(ge=0)
    mutation_prob: float = Field(ge=0, le=1)
    max_ruin_fraction: float = Field(gt=0, le=1)
    contiguous_ruin_prob: float = Field(ge=0, le=1)
    top_k: int = Field(gt=0)
    no_op_retries: int = Field(ge=0)
    reheat: ReheatConfig | None
    seed: int | None


# ============================================================================
# Observer protocol
# ============================================================================


class SAObserver(Protocol):
    """Observer for SA events.

    ``SearchObserver`` structurally satisfies this protocol — no adapter needed.
    """

    def on_annealing_step(self, event: AnnealingStepEvent) -> None: ...
    def on_annealing_restart(self, event: AnnealingRestartEvent) -> None: ...
    def on_annealing_reheat(self, event: AnnealingReheatEvent) -> None: ...


# ============================================================================
# Result
# ============================================================================


@dataclass(frozen=True)
class MultiStartSAResult:
    """Result of ``run_multi_start_sa``."""

    candidates: list[tuple[tuple[str, ...], float]]  # (assignment, score) sorted desc
    total_steps: int
    best_restart: int
    total_reheats: int
    final_temperature: float
    any_early_termination: bool


# ============================================================================
# Metropolis criterion
# ============================================================================


def metropolis_accept(
    delta: float, temperature: float, rng: random.Random
) -> bool:
    """Metropolis acceptance criterion for SA.

    Always accepts improving moves (delta > 0).
    Worsening moves accepted with probability exp(delta / T).
    """
    if delta > 0:
        return True
    if temperature <= 0:
        return False
    return rng.random() < math.exp(delta / temperature)


# ============================================================================
# Top-K tracker
# ============================================================================


class TopKTracker:
    """Bounded min-heap for tracking top-k distinct candidates by score."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._heap: list[tuple[float, tuple[str, ...]]] = []
        self._seen: set[tuple[str, ...]] = set()

    def add(self, candidate: tuple[str, ...], score: float) -> None:
        """Add candidate if novel and good enough."""
        if candidate in self._seen:
            return
        self._seen.add(candidate)
        if len(self._heap) < self._max_size:
            heapq.heappush(self._heap, (score, candidate))
        elif score > self._heap[0][0]:
            heapq.heapreplace(self._heap, (score, candidate))

    def top_candidates(self) -> list[tuple[tuple[str, ...], float]]:
        """Return (candidate, score) pairs sorted by score descending."""
        return [
            (candidate, score)
            for score, candidate in sorted(self._heap, reverse=True)
        ]

    def __len__(self) -> int:
        return len(self._heap)


# ============================================================================
# Multi-start SA
# ============================================================================


def pick_restart_allocation(
    feasible_allocations: list[dict[str, int]],
    allocation_cost: Callable[[dict[str, int]], float],
    cost_budget: float | None,
    rng: random.Random,
) -> dict[str, int]:
    """Pick a random allocation for SA restart.

    With a budget: target a random cost in [0.7 * budget, budget].
    Without a budget (min_cost mode): uniform random from all allocations.
    """
    if cost_budget is None or len(feasible_allocations) <= 1:
        return rng.choice(feasible_allocations)

    target_cost = rng.uniform(0.7, 1.0) * cost_budget

    best_alloc = feasible_allocations[0]
    best_diff = float("inf")
    for alloc in feasible_allocations:
        diff = abs(allocation_cost(alloc) - target_cost)
        if diff < best_diff:
            best_diff = diff
            best_alloc = alloc
    return best_alloc


async def run_multi_start_sa(
    score_fn: Callable[[tuple[str, ...]], Awaitable[float]],
    allocation_cost: Callable[[dict[str, int]], float],
    feasible_allocations: list[dict[str, int]],
    num_layers: int,
    mixer_names: list[str],
    cost_budget: float | None,
    config: SAConfig,
    rng: random.Random,
    observers: Sequence[SAObserver] = (),
) -> MultiStartSAResult:
    """Run multi-start SA with an async scoring function as objective.

    Returns a ``MultiStartSAResult`` with up to ``config.top_k`` distinct
    candidates sorted by score (desc).
    """
    tracker = TopKTracker(config.top_k)

    total_steps = 0
    total_reheats = 0
    best_restart = 0
    best_global_score = float("-inf")
    last_temperature = config.initial_temperature
    any_early = False

    for restart in range(config.num_restarts):
        alloc = pick_restart_allocation(
            feasible_allocations, allocation_cost, cost_budget, rng
        )
        current = random_assignment(rng, alloc, num_layers)
        current_score = await score_fn(tuple(current))

        tracker.add(tuple(current), current_score)

        if current_score > best_global_score:
            best_global_score = current_score
            best_restart = restart

        best = current.copy()
        best_score = current_score

        # Emit restart event
        if observers:
            restart_event = AnnealingRestartEvent(
                restart_index=restart,
                num_restarts=config.num_restarts,
                steps_budget=config.steps_per_restart,
                initial_assignments=tuple(current),
            )
            for obs in observers:
                obs.on_annealing_restart(restart_event)

        temperature = config.initial_temperature

        # Acceptance tracking for adaptive reheating
        accept_window: deque[bool] | None = None
        if config.reheat is not None:
            accept_window = deque(maxlen=config.reheat.window)

        steps_taken_this_restart = 0

        for step in range(config.steps_per_restart):
            # Sample move, with no-op retries
            proposed = None
            move_type: str = "random_ruin"
            k = 0

            for _attempt in range(1 + config.no_op_retries):
                if rng.random() < config.mutation_prob:
                    # Type mutation -- changes allocation, hard budget check
                    candidate = type_mutation(rng, current, mixer_names)
                    if cost_budget is not None:
                        proposed_alloc = dict(Counter(candidate))
                        if allocation_cost(proposed_alloc) > cost_budget + 1e-9:
                            continue
                    if candidate != current:
                        proposed = candidate
                        move_type = "type_mutation"
                        k = 1
                        break
                else:
                    # Allocation-preserving ruin-and-recreate
                    k = sample_perturbation_size(
                        rng,
                        num_layers,
                        temperature,
                        config.initial_temperature,
                        config.max_ruin_fraction,
                    )
                    candidate, move_type = generate_move(
                        rng, current, k, config.contiguous_ruin_prob
                    )
                    if candidate != current:
                        proposed = candidate
                        break

            if proposed is None:
                # All attempts were no-ops, skip this step
                steps_taken_this_restart = step + 1
                temperature *= config.cooling_rate
                if temperature < config.min_temperature:
                    break
                continue

            proposed_score = await score_fn(tuple(proposed))

            # Metropolis criterion
            delta = proposed_score - current_score
            accepted = metropolis_accept(delta, temperature, rng)
            acceptance_prob = 1.0 if delta > 0 else (
                math.exp(delta / temperature) if temperature > 0 else 0.0
            )

            # Track acceptance for reheating
            if accept_window is not None:
                accept_window.append(accepted)

            if accepted:
                current = proposed
                current_score = proposed_score
                if proposed_score > best_score:
                    best = current.copy()
                    best_score = proposed_score

            tracker.add(tuple(current), current_score)

            steps_taken_this_restart = step + 1

            # Emit step event
            if observers:
                step_event = AnnealingStepEvent(
                    step=step,
                    max_steps=config.steps_per_restart,
                    perturbation_size=k,
                    move_type=move_type,
                    restart_index=restart,
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
                for obs in observers:
                    obs.on_annealing_step(step_event)

            # Cool down
            temperature *= config.cooling_rate

            # Adaptive reheating check
            if (
                config.reheat is not None
                and accept_window is not None
                and len(accept_window) == accept_window.maxlen
            ):
                acceptance_rate = sum(accept_window) / len(accept_window)
                if acceptance_rate < config.reheat.threshold:
                    old_temp = temperature
                    temperature = config.initial_temperature * config.reheat.fraction
                    total_reheats += 1
                    accept_window.clear()

                    if observers:
                        reheat_event = AnnealingReheatEvent(
                            restart_index=restart,
                            step=step,
                            old_temperature=old_temp,
                            new_temperature=temperature,
                            acceptance_rate=acceptance_rate,
                        )
                        for obs in observers:
                            obs.on_annealing_reheat(reheat_event)

            # Early termination if temperature is too low
            if temperature < config.min_temperature:
                break

        total_steps += steps_taken_this_restart
        last_temperature = temperature

        if steps_taken_this_restart < config.steps_per_restart:
            any_early = True

        if best_score > best_global_score:
            best_global_score = best_score
            best_restart = restart

    return MultiStartSAResult(
        candidates=tracker.top_candidates(),
        total_steps=total_steps,
        best_restart=best_restart,
        total_reheats=total_reheats,
        final_temperature=last_temperature,
        any_early_termination=any_early,
    )
