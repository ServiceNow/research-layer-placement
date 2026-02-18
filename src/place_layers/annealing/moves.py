"""Pure move operators for simulated annealing.

Free functions that implement allocation-preserving moves on assignment vectors.
Used by ``AnnealingSearch`` (runtime optimisation) and ``SurrogateSearch``
(surrogate-guided refinement).
"""

from __future__ import annotations

import random
from typing import Literal

from place_layers.cost_model.base_cost_model import Allocation


def ruin_and_recreate(rng: random.Random, assignments: list[str], k: int) -> list[str]:
    """Ruin-and-recreate: sample k random positions, shuffle their types.

    Allocation-preserving by construction: the same types are placed
    back at the same positions, just in a different order.
    At k=2 this is equivalent to a swap.
    """
    n = len(assignments)
    k = min(k, n)
    positions = rng.sample(range(n), k)
    types = [assignments[pos] for pos in positions]
    rng.shuffle(types)
    proposed = assignments.copy()
    for pos, typ in zip(positions, types):
        proposed[pos] = typ
    return proposed


def contiguous_ruin(rng: random.Random, assignments: list[str], k: int) -> list[str]:
    """Contiguous ruin: shuffle a contiguous block of k positions.

    Allocation-preserving by construction.
    """
    n = len(assignments)
    k = min(k, n)
    start = rng.randint(0, n - k)
    proposed = assignments.copy()
    block = proposed[start : start + k]
    rng.shuffle(block)
    proposed[start : start + k] = block
    return proposed


def generate_move(
    rng: random.Random,
    assignments: list[str],
    k: int,
    contiguous_p: float,
) -> tuple[list[str], Literal["random_ruin", "contiguous_ruin"]]:
    """Generate a ruin-and-recreate move.

    With probability *contiguous_p*, uses contiguous ruin.
    Otherwise, uses random (scattered) ruin.

    Returns ``(proposed_assignments, move_type)``.
    """
    if rng.random() < contiguous_p:
        return contiguous_ruin(rng, assignments, k), "contiguous_ruin"
    else:
        return ruin_and_recreate(rng, assignments, k), "random_ruin"


def sample_perturbation_size(
    rng: random.Random,
    num_layers: int,
    temperature: float,
    initial_temperature: float,
    max_ruin_fraction: float,
) -> int:
    """Sample perturbation size k from a geometric distribution.

    The expected value of k is coupled to the temperature ratio so that
    higher temperatures produce larger perturbations.
    """
    t_ratio = temperature / initial_temperature if initial_temperature > 0 else 0.0
    t_ratio = min(t_ratio, 1.0)

    max_k_at_full_temp = num_layers * max_ruin_fraction
    expected = 2.0 + t_ratio * (max_k_at_full_temp - 2.0)

    if expected > 2.0:
        p = (expected - 2.0) / (expected - 1.0)
    else:
        p = 0.0

    max_k = max(2, int(num_layers * (0.1 + 0.5 * t_ratio)))

    k = 2
    while rng.random() < p and k < max_k:
        k += 1

    return min(k, num_layers)


def type_mutation(
    rng: random.Random,
    assignments: list[str],
    mixer_names: list[str],
) -> list[str]:
    """Change one random position to a different mixer type.

    Unlike ruin_and_recreate, this changes the allocation.
    Caller must check cost feasibility of the new allocation.
    """
    pos = rng.randrange(len(assignments))
    current_type = assignments[pos]
    candidates = [m for m in mixer_names if m != current_type]
    if not candidates:
        return assignments.copy()
    proposed = assignments.copy()
    proposed[pos] = rng.choice(candidates)
    return proposed


def random_assignment(
    rng: random.Random,
    allocation: Allocation,
    num_layers: int,
) -> list[str]:
    """Build a random initial assignment from allocation counts.

    Creates a flat list from allocation counts, then shuffles randomly.
    """
    assignments: list[str] = []
    for mixer_name, count in allocation.items():
        assignments.extend([mixer_name] * count)
    assert len(assignments) == num_layers
    rng.shuffle(assignments)
    return assignments
