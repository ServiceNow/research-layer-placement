"""Exact MRF solver using dynamic programming (Viterbi algorithm).

This module provides exact optimization for chain MRFs, computing the
globally optimal layer placement. Two modes are supported:

1. Unconstrained: Find best placement with no allocation constraints
   - Complexity: O(N * K^(m+1)) where m = model.memory

2. Constrained: Find best placement with exact allocation constraints
   - Complexity: O(N^K * K^(m+1)) via augmented Viterbi with allocation tracking

where N = num_layers and K = num_mixers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation

if TYPE_CHECKING:
    from place_layers.mrf.model import ChainMRFModel


@dataclass(frozen=True)
class ExactSolution:
    """Result from exact MRF optimization."""

    placement: LayerPlacement
    true_score: float


class ExactMRFSolver:
    """Viterbi-style DP solver for chain MRF optimization.

    This solver finds the globally optimal layer placement by exploiting
    the chain structure of the MRF. The key insight is that the score
    decomposes into unary (per-layer) and pairwise (adjacent-layer) terms,
    enabling efficient dynamic programming.

    For unconstrained optimization:
        V[i, state] = best score for layers 0..i with recent history = state

    For constrained optimization (with allocation tracking):
        V[i, state, c] = best score for layers 0..i with:
            - recent history = state
            - exactly c[k] layers assigned to mixer k overall
    """

    def __init__(self, model: ChainMRFModel) -> None:
        """Initialize solver with MRF model.

        Args:
            model: Chain MRF model to optimize.
        """
        self.model = model

    def _transition_score(self, layer: int, state: tuple[int, ...], new_type: int) -> float:
        """Compute all interaction terms for assigning new_type at layer given state.

        Args:
            layer: Current layer index (the layer being assigned new_type).
            state: Tuple of preceding type indices (most recent last).
            new_type: Mixer index being assigned to this layer.

        Returns:
            Sum of all interaction potentials involving this assignment.
        """
        model = self.model
        names = model.mixer_names
        score = 0.0

        # Range-1 pairwise (adjacent): needs at least 1 preceding layer
        if len(state) >= 1:
            prev_type = state[-1]
            score += model.pairwise_potential(layer - 1, names[prev_type], names[new_type])

        # Skip-pairwise: for each range r, need state[-r] (which is r layers back)
        for r in range(2, model.pairwise_max_range + 1):
            if len(state) >= r and layer >= r:
                back_type = state[-r]
                score += model.skip_pairwise_potential(
                    layer - r, r, names[back_type], names[new_type]
                )

        # Triplets: (layer-2, layer-1, layer) — need 2 preceding layers
        if model.triplet_scale > 0 and len(state) >= 2 and layer >= 2:
            score += model.triplet_potential(
                layer - 2, names[state[-2]], names[state[-1]], names[new_type]
            )

        return score

    def solve_unconstrained(self) -> ExactSolution:
        """Find optimal placement with no allocation constraints.

        Uses generalized Viterbi algorithm. State is a tuple of recent
        mixer indices of length min(i+1, m) where m = model.memory.

        Returns:
            ExactSolution with optimal placement and true score.
        """
        N = self.model.num_layers
        K = self.model.num_mixers
        names = self.model.mixer_names
        m = self.model.memory

        # V[i]: dict mapping state_tuple -> best score
        # backtrack[i]: dict mapping state_tuple -> previous state_tuple
        V: list[dict[tuple[int, ...], float]] = [{} for _ in range(N)]
        backtrack: list[dict[tuple[int, ...], tuple[int, ...]]] = [{} for _ in range(N)]

        # Base case: first layer
        for j in range(K):
            state = (j,)
            V[0][state] = self.model.unary_potential(0, names[j])

        # Forward pass
        for i in range(1, N):
            for prev_state, prev_score in V[i - 1].items():
                for j in range(K):
                    unary = self.model.unary_potential(i, names[j])
                    trans = self._transition_score(i, prev_state, j)
                    score = prev_score + unary + trans

                    new_state = (*prev_state, j)[-m:]

                    if new_state not in V[i] or score > V[i][new_state]:
                        V[i][new_state] = score
                        backtrack[i][new_state] = prev_state

        # Find best final state
        best_score = float("-inf")
        best_state: tuple[int, ...] | None = None
        for state, score in V[N - 1].items():
            if score > best_score:
                best_score = score
                best_state = state

        assert best_state is not None

        # Backtrack to find optimal assignment
        assignment_indices: list[int] = [0] * N
        current_state = best_state
        assignment_indices[N - 1] = current_state[-1]

        for i in range(N - 2, -1, -1):
            prev_state = backtrack[i + 1][current_state]
            assignment_indices[i] = prev_state[-1]
            current_state = prev_state

        # Convert to LayerPlacement
        assignments = tuple(names[j] for j in assignment_indices)
        placement = LayerPlacement(assignments=assignments)

        return ExactSolution(
            placement=placement,
            true_score=best_score,
        )

    def solve_constrained(self, allocation: Allocation) -> ExactSolution | None:
        """Find optimal placement with exact allocation constraints.

        Uses augmented Viterbi with allocation tracking.

        Args:
            allocation: Required counts per mixer, e.g., {"mixer_0": 5, "mixer_1": 15}.
                Sum must equal num_layers.

        Returns:
            ExactSolution if a valid placement exists, None otherwise.
        """
        N = self.model.num_layers
        K = self.model.num_mixers
        names = self.model.mixer_names
        m = self.model.memory

        # Convert allocation to target counts tuple (in mixer_names order)
        target_counts = tuple(
            allocation.get(name, 0) for name in names
        )

        if sum(target_counts) != N:
            raise ValueError(
                f"Allocation sum ({sum(target_counts)}) must equal num_layers ({N})"
            )

        # State: (history_tuple, counts_tuple)
        # V[layer][(history, counts)] = best score
        V: list[dict[tuple[tuple[int, ...], tuple[int, ...]], float]] = [{} for _ in range(N)]
        backtrack: list[
            dict[
                tuple[tuple[int, ...], tuple[int, ...]],
                tuple[tuple[int, ...], tuple[int, ...]],
            ]
        ] = [{} for _ in range(N)]

        # Base case: first layer
        for j in range(K):
            counts = tuple(1 if k == j else 0 for k in range(K))
            if all(counts[k] <= target_counts[k] for k in range(K)):
                history = (j,)
                state = (history, counts)
                V[0][state] = self.model.unary_potential(0, names[j])

        # Forward pass
        for i in range(1, N):
            for (prev_history, prev_counts), prev_score in V[i - 1].items():
                for j in range(K):
                    new_counts = tuple(
                        prev_counts[k] + (1 if k == j else 0) for k in range(K)
                    )

                    # Prune if any count exceeds target
                    if any(new_counts[k] > target_counts[k] for k in range(K)):
                        continue

                    unary = self.model.unary_potential(i, names[j])
                    trans = self._transition_score(i, prev_history, j)
                    score = prev_score + unary + trans

                    new_history = (*prev_history, j)[-m:]
                    new_state = (new_history, new_counts)

                    if new_state not in V[i] or score > V[i][new_state]:
                        V[i][new_state] = score
                        backtrack[i][new_state] = (prev_history, prev_counts)

        # Find best final state with exact target counts
        best_score = float("-inf")
        best_state: tuple[tuple[int, ...], tuple[int, ...]] | None = None

        for (history, counts), score in V[N - 1].items():
            if counts == target_counts and score > best_score:
                best_score = score
                best_state = (history, counts)

        if best_state is None:
            return None

        # Backtrack to reconstruct assignment
        assignment_indices: list[int] = [0] * N
        current_state = best_state
        assignment_indices[N - 1] = current_state[0][-1]

        for i in range(N - 2, -1, -1):
            prev_history, prev_counts = backtrack[i + 1][current_state]
            assignment_indices[i] = prev_history[-1]
            current_state = (prev_history, prev_counts)

        # Convert to LayerPlacement
        assignments = tuple(names[j] for j in assignment_indices)
        placement = LayerPlacement(assignments=assignments)

        return ExactSolution(
            placement=placement,
            true_score=best_score,
        )

    def enumerate_all(self) -> list[tuple[LayerPlacement, float]]:
        """Enumerate all possible placements with their true scores.

        This is for testing/verification only - complexity is O(K^N).

        Returns:
            List of (LayerPlacement, true_score) pairs, sorted by score descending.
        """
        N = self.model.num_layers
        K = self.model.num_mixers
        names = self.model.mixer_names

        results: list[tuple[LayerPlacement, float]] = []

        for assignment in product(range(K), repeat=N):
            placement = LayerPlacement(assignments=tuple(names[j] for j in assignment))
            score = self.model.true_score(placement)
            results.append((placement, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def enumerate_constrained(
        self, allocation: Allocation
    ) -> list[tuple[LayerPlacement, float]]:
        """Enumerate all placements matching an allocation constraint.

        This is for testing/verification only.

        Args:
            allocation: Required counts per mixer.

        Returns:
            List of (LayerPlacement, true_score) pairs, sorted by score descending.
        """
        N = self.model.num_layers
        K = self.model.num_mixers
        names = self.model.mixer_names

        # Convert allocation to target counts tuple
        target_counts = tuple(
            allocation.get(name, 0) for name in names
        )

        if sum(target_counts) != N:
            raise ValueError(
                f"Allocation sum ({sum(target_counts)}) must equal num_layers ({N})"
            )

        results: list[tuple[LayerPlacement, float]] = []

        for assignment in product(range(K), repeat=N):
            # Check if assignment matches allocation
            counts = tuple(assignment.count(k) for k in range(K))
            if counts != target_counts:
                continue

            placement = LayerPlacement(assignments=tuple(names[j] for j in assignment))
            score = self.model.true_score(placement)
            results.append((placement, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
