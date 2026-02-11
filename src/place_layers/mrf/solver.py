"""Exact MRF solver using dynamic programming (Viterbi algorithm).

This module provides exact optimization for chain MRFs, computing the
globally optimal layer placement. Two modes are supported:

1. Unconstrained: Find best placement with no allocation constraints
   - Complexity: O(N * K^2) via standard Viterbi

2. Constrained: Find best placement with exact allocation constraints
   - Complexity: O(N^K * K) via augmented Viterbi with allocation tracking

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
        V[i, j] = best score for layers 0..i-1 with layer i-1 on mixer j

    For constrained optimization (with allocation tracking):
        V[i, j, c] = best score for layers 0..i-1 with:
            - layer i-1 on mixer j
            - exactly c[k] layers assigned to mixer k overall
    """

    def __init__(self, model: ChainMRFModel) -> None:
        """Initialize solver with MRF model.

        Args:
            model: Chain MRF model to optimize.
        """
        self.model = model

    def solve_unconstrained(self) -> ExactSolution:
        """Find optimal placement with no allocation constraints.

        Uses standard Viterbi algorithm with complexity O(N * K^2).

        Returns:
            ExactSolution with optimal placement and true score.
        """
        N = self.model.num_layers
        K = self.model.num_mixers
        names = self.model.mixer_names

        # V[i, j] = best score for layers 0..i with layer i on mixer j
        # backtrack[i, j] = best mixer for layer i-1 given layer i is on mixer j
        V: list[list[float]] = [[float("-inf")] * K for _ in range(N)]
        backtrack: list[list[int]] = [[-1] * K for _ in range(N)]

        # Base case: first layer
        for j in range(K):
            V[0][j] = self.model.unary_potential(0, names[j])

        # Forward pass: fill DP table
        for i in range(1, N):
            for j in range(K):  # current layer's mixer
                unary = self.model.unary_potential(i, names[j])
                best_score = float("-inf")
                best_prev = -1

                for prev_j in range(K):  # previous layer's mixer
                    pairwise = self.model.pairwise_potential(i - 1, names[prev_j], names[j])
                    score = V[i - 1][prev_j] + pairwise + unary

                    if score > best_score:
                        best_score = score
                        best_prev = prev_j

                V[i][j] = best_score
                backtrack[i][j] = best_prev

        # Find best final state
        best_final_score = float("-inf")
        best_final_mixer = -1
        for j in range(K):
            if V[N - 1][j] > best_final_score:
                best_final_score = V[N - 1][j]
                best_final_mixer = j

        # Backtrack to find optimal assignment
        assignment_indices: list[int] = [0] * N
        assignment_indices[N - 1] = best_final_mixer

        for i in range(N - 2, -1, -1):
            assignment_indices[i] = backtrack[i + 1][assignment_indices[i + 1]]

        # Convert to LayerPlacement
        assignments = tuple(names[j] for j in assignment_indices)
        placement = LayerPlacement(assignments=assignments)

        return ExactSolution(
            placement=placement,
            true_score=best_final_score,
        )

    def solve_constrained(self, allocation: Allocation) -> ExactSolution | None:
        """Find optimal placement with exact allocation constraints.

        Uses augmented Viterbi with allocation tracking. Complexity is
        O(N^K * K) where the N^K factor comes from tracking all possible
        partial allocations.

        For small K (2-4 mixers), this is tractable:
        - K=2, N=50: ~5,000 states
        - K=3, N=50: ~190,000 states
        - K=4, N=50: ~15M states

        Args:
            allocation: Required counts per mixer, e.g., {"mixer_0": 5, "mixer_1": 15}.
                Sum must equal num_layers.

        Returns:
            ExactSolution if a valid placement exists, None otherwise.
        """
        N = self.model.num_layers
        K = self.model.num_mixers
        names = self.model.mixer_names

        # Convert allocation to target counts tuple (in mixer_names order)
        target_counts = tuple(
            allocation.get(name, 0) for name in names
        )

        # Validate allocation sums to num_layers
        if sum(target_counts) != N:
            raise ValueError(
                f"Allocation sum ({sum(target_counts)}) must equal num_layers ({N})"
            )

        # State: (layer, last_mixer, partial_counts)
        # partial_counts is a tuple of K integers representing counts so far

        # V[layer][last_mixer][partial_counts] = best score
        # backtrack stores (prev_mixer, prev_counts) for reconstruction

        # We use dicts to handle sparse state space
        V: list[dict[tuple[int, tuple[int, ...]], float]] = [{} for _ in range(N)]
        backtrack: list[dict[tuple[int, tuple[int, ...]], tuple[int, tuple[int, ...]]]] = (
            [{} for _ in range(N)]
        )

        # Base case: first layer
        for j in range(K):
            counts = tuple(1 if k == j else 0 for k in range(K))
            # Check if counts don't exceed target (early pruning)
            if all(counts[k] <= target_counts[k] for k in range(K)):
                V[0][(j, counts)] = self.model.unary_potential(0, names[j])

        # Forward pass
        for i in range(1, N):
            for (prev_j, prev_counts), prev_score in V[i - 1].items():
                for j in range(K):
                    # Update counts
                    new_counts = tuple(
                        prev_counts[k] + (1 if k == j else 0) for k in range(K)
                    )

                    # Prune if any count exceeds target
                    if any(new_counts[k] > target_counts[k] for k in range(K)):
                        continue

                    # Compute score
                    unary = self.model.unary_potential(i, names[j])
                    pairwise = self.model.pairwise_potential(i - 1, names[prev_j], names[j])
                    score = prev_score + unary + pairwise

                    state = (j, new_counts)
                    if state not in V[i] or score > V[i][state]:
                        V[i][state] = score
                        backtrack[i][state] = (prev_j, prev_counts)

        # Find best final state with exact target counts
        best_score = float("-inf")
        best_state: tuple[int, tuple[int, ...]] | None = None

        for (j, counts), score in V[N - 1].items():
            if counts == target_counts and score > best_score:
                best_score = score
                best_state = (j, counts)

        if best_state is None:
            return None  # No valid placement exists

        # Backtrack to reconstruct assignment
        assignment_indices: list[int] = [0] * N
        current_state = best_state
        assignment_indices[N - 1] = current_state[0]

        for i in range(N - 2, -1, -1):
            prev_j, prev_counts = backtrack[i + 1][current_state]
            assignment_indices[i] = prev_j
            current_state = (prev_j, prev_counts)

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
