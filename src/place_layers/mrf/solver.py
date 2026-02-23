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

import heapq
import logging
from dataclasses import dataclass
from itertools import product
from math import comb
from typing import TYPE_CHECKING

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation

if TYPE_CHECKING:
    from place_layers.mrf.model import MRFModelProtocol

logger = logging.getLogger(__name__)

# Type aliases for DP state representation
_FullState = tuple[tuple[int, ...], tuple[int, ...]]  # (history, counts)


@dataclass(frozen=True)
class ExactSolution:
    """Result from exact MRF optimization."""

    placement: LayerPlacement
    true_score: float


@dataclass
class _ConstrainedDPTables:
    """Internal DP tables from constrained forward pass."""

    V: list[dict[_FullState, float]]
    backtrack: list[dict[_FullState, _FullState]]
    feasible_set: set[tuple[int, ...]]
    # Side index: terminal states grouped by counts (only feasible counts)
    terminal_by_counts: dict[tuple[int, ...], dict[tuple[int, ...], float]]
    N: int
    names: tuple[str, ...]


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

    def __init__(self, model: MRFModelProtocol) -> None:
        """Initialize solver with MRF model.

        Args:
            model: Any object implementing MRFModelProtocol.
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

    # ------------------------------------------------------------------
    # Internal: shared DP build + backtrack helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_peak_states(K: int, m: int, N: int, max_counts: list[int]) -> int:
        """Upper-bound the peak number of DP states across all layers.

        Uses inclusion-exclusion over ``max_counts`` caps for a tight,
        cap-aware estimate.  For each layer *i* the number of distinct
        ``(history, counts)`` states is bounded by::

            K^min(i+1, m)  ×  #{tuples summing to i+1 with per-component caps}

        The composition count with caps is computed via inclusion-exclusion::

            count(n) = Σ_{S ⊆ [K]} (-1)^|S|
                       × C(n - Σ_{k∈S}(cap[k]+1) + K-1, K-1)

        Cost: O(N × 2^K) ``comb()`` calls — negligible for K ≤ 6.

        Args:
            K: Number of mixer types.
            m: Model memory (history length).
            N: Number of layers.
            max_counts: Per-mixer upper bounds (length K).

        Returns:
            Upper bound on the maximum number of states at any single layer.
        """
        peak = 0
        for i in range(N):
            n = i + 1  # layers 0..i use counts summing to i+1
            # Inclusion-exclusion over subsets of [K]
            compositions = 0
            for mask in range(1 << K):
                excess = 0
                bits = 0
                for k in range(K):
                    if mask & (1 << k):
                        excess += max_counts[k] + 1
                        bits += 1
                remainder = n - excess + K - 1
                if remainder < K - 1:
                    continue  # C(neg, ...) = 0
                term = comb(remainder, K - 1)
                if bits % 2 == 0:
                    compositions += term
                else:
                    compositions -= term
            history_states = K ** min(i + 1, m)
            layer_states = history_states * max(compositions, 0)
            if layer_states > peak:
                peak = layer_states
        return peak

    def _build_constrained_tables(
        self,
        feasible_allocations: list[Allocation],
        *,
        max_states_per_layer: int | None = None,
    ) -> _ConstrainedDPTables | None:
        """Build constrained DP tables for all feasible allocations simultaneously.

        Args:
            feasible_allocations: List of allocations to consider.
            max_states_per_layer: If set and any layer exceeds this many states,
                abort early and return None (signals caller to fall back).

        Returns:
            _ConstrainedDPTables if successful, None if feasible_set is empty
            or early abort triggered.
        """
        N = self.model.num_layers
        K = self.model.num_mixers
        names = self.model.mixer_names
        m = self.model.memory

        # Convert allocations to target count tuples and build feasible set
        feasible_set: set[tuple[int, ...]] = set()
        for alloc in feasible_allocations:
            target = tuple(alloc.get(name, 0) for name in names)
            if sum(target) == N:
                feasible_set.add(target)

        if not feasible_set:
            return None

        # Max counts per mixer for pruning
        max_counts = [0] * K
        for target in feasible_set:
            for k in range(K):
                if target[k] > max_counts[k]:
                    max_counts[k] = target[k]

        # Pre-check: estimate peak states before allocating DP tables
        if max_states_per_layer is not None:
            estimated_peak = self._estimate_peak_states(K, m, N, max_counts)
            if estimated_peak > max_states_per_layer:
                logger.info(
                    "DP pre-check: estimated peak states %d exceeds threshold %d "
                    "(K=%d, m=%d, N=%d, max_counts=%s). Skipping DP build.",
                    estimated_peak, max_states_per_layer, K, m, N, max_counts,
                )
                return None

        # State: (history_tuple, counts_tuple)
        V: list[dict[_FullState, float]] = [{} for _ in range(N)]
        backtrack: list[dict[_FullState, _FullState]] = [{} for _ in range(N)]
        terminal_by_counts: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}

        # Base case: first layer
        for j in range(K):
            if 1 > max_counts[j]:
                continue
            counts = tuple(1 if k == j else 0 for k in range(K))
            history = (j,)
            state: _FullState = (history, counts)
            V[0][state] = self.model.unary_potential(0, names[j])

        # Forward pass
        for i in range(1, N):
            is_final = i == N - 1
            for (prev_history, prev_counts), prev_score in V[i - 1].items():
                for j in range(K):
                    new_count_j = prev_counts[j] + 1
                    if new_count_j > max_counts[j]:
                        continue

                    new_counts = tuple(
                        prev_counts[k] + (1 if k == j else 0) for k in range(K)
                    )

                    unary = self.model.unary_potential(i, names[j])
                    trans = self._transition_score(i, prev_history, j)
                    score = prev_score + unary + trans

                    new_history = (*prev_history, j)[-m:]
                    new_state: _FullState = (new_history, new_counts)

                    if new_state not in V[i] or score > V[i][new_state]:
                        V[i][new_state] = score
                        backtrack[i][new_state] = (prev_history, prev_counts)

                        # Side index for final layer: record feasible terminal states
                        if is_final and new_counts in feasible_set:
                            by_counts = terminal_by_counts.get(new_counts)
                            if by_counts is None:
                                by_counts = {}
                                terminal_by_counts[new_counts] = by_counts
                            # Key by history (new_history) — the counts are the group key
                            if new_history not in by_counts or score > by_counts[new_history]:
                                by_counts[new_history] = score

            # Early-abort guard
            if max_states_per_layer is not None and len(V[i]) > max_states_per_layer:
                logger.warning(
                    "DP state space blowup: layer %d has %d states "
                    "(max_counts=%s, threshold=%d). Aborting DP build.",
                    i,
                    len(V[i]),
                    max_counts,
                    max_states_per_layer,
                )
                return None

        return _ConstrainedDPTables(
            V=V,
            backtrack=backtrack,
            feasible_set=feasible_set,
            terminal_by_counts=terminal_by_counts,
            N=N,
            names=tuple(names),
        )

    def _backtrack_one(
        self,
        tables: _ConstrainedDPTables,
        score: float,
        state: _FullState,
    ) -> ExactSolution:
        """Reconstruct a single placement from a final-layer state.

        Args:
            tables: DP tables from _build_constrained_tables.
            score: The score of this final state.
            state: (history, counts) tuple at the final layer.

        Returns:
            ExactSolution with reconstructed placement.
        """
        N = tables.N
        names = tables.names
        backtrack = tables.backtrack

        assignment_indices: list[int] = [0] * N
        current_state = state
        assignment_indices[N - 1] = current_state[0][-1]

        for i in range(N - 2, -1, -1):
            prev_history, prev_counts = backtrack[i + 1][current_state]
            assignment_indices[i] = prev_history[-1]
            current_state = (prev_history, prev_counts)

        assignments = tuple(names[j] for j in assignment_indices)
        placement = LayerPlacement(assignments=assignments)
        return ExactSolution(placement=placement, true_score=score)

    # ------------------------------------------------------------------
    # Public API: top-n methods
    # ------------------------------------------------------------------

    def solve_top_n(
        self,
        feasible_allocations: list[Allocation],
        n: int,
    ) -> list[ExactSolution]:
        """Find the globally top-n placements across all feasible allocations.

        Single DP pass with allocation tracking (state = (history, counts)),
        tracking ALL feasible count tuples simultaneously.  At the final layer,
        collects ALL entries whose counts match a feasible allocation, sorts
        globally by score, and backtracks the top n.

        Multiple solutions MAY share the same allocation but differ in
        arrangement (different history states).

        Args:
            feasible_allocations: List of allocations to consider.
            n: Number of top solutions to return.

        Returns:
            List of ExactSolution sorted by score descending, length <= n.
        """
        tables = self._build_constrained_tables(feasible_allocations)
        if tables is None:
            return []

        N = tables.N
        V = tables.V
        feasible_set = tables.feasible_set

        # Collect all final-layer entries whose counts are feasible
        candidates: list[tuple[float, _FullState]] = []
        for (history, counts), score in V[N - 1].items():
            if counts in feasible_set:
                candidates.append((score, (history, counts)))

        # Sort globally by score descending, take top n
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:n]

        # Backtrack each to reconstruct placement
        return [
            self._backtrack_one(tables, score, state)
            for score, state in top_candidates
        ]

    def solve_top_n_per_allocation(
        self,
        feasible_allocations: list[Allocation],
        n_per_allocation: int,
        *,
        max_states_per_layer: int = 2_000_000,
    ) -> dict[tuple[int, ...], list[ExactSolution]]:
        """Find top-n placements per allocation using a single DP pass.

        Single forward pass builds tables for ALL allocations simultaneously.
        Then for each allocation, filters terminal states matching that allocation
        and backtracks top-n.

        Args:
            feasible_allocations: List of allocations to consider.
            n_per_allocation: Maximum solutions per allocation.
            max_states_per_layer: If any layer exceeds this many states during
                the forward pass, fall back to the per-allocation loop.

        Returns:
            dict mapping counts tuple (in mixer_names order) to list of
            ExactSolution, sorted by score descending. Callers can flatten
            with itertools.chain.from_iterable(result.values()) for a pool.
        """
        tables = self._build_constrained_tables(
            feasible_allocations, max_states_per_layer=max_states_per_layer
        )
        if tables is None:
            return self._solve_top_n_bisect(
                feasible_allocations, n_per_allocation,
                max_states_per_layer=max_states_per_layer,
            )

        return self._extract_per_allocation_results(tables, n_per_allocation)

    def _extract_per_allocation_results(
        self,
        tables: _ConstrainedDPTables,
        n_per_allocation: int,
    ) -> dict[tuple[int, ...], list[ExactSolution]]:
        """Extract top-n solutions per allocation from completed DP tables.

        Uses a streaming bounded min-heap per counts tuple, then backtracks
        each kept state.

        Args:
            tables: DP tables from _build_constrained_tables.
            n_per_allocation: Maximum solutions per allocation.

        Returns:
            dict mapping counts tuple to list of ExactSolution, sorted by
            score descending.
        """
        # Streaming bounded min-heap per counts tuple
        heaps: dict[tuple[int, ...], list[tuple[float, tuple[int, ...]]]] = {}
        for counts_tuple, entries in tables.terminal_by_counts.items():
            for history, score in entries.items():
                h = heaps.get(counts_tuple)
                if h is None:
                    heaps[counts_tuple] = [(score, history)]
                elif len(h) < n_per_allocation:
                    heapq.heappush(h, (score, history))
                elif score > h[0][0]:
                    heapq.heapreplace(h, (score, history))

        # Backtrack each kept state
        results: dict[tuple[int, ...], list[ExactSolution]] = {}
        for counts_tuple, heap in heaps.items():
            results[counts_tuple] = [
                self._backtrack_one(
                    tables, score, (history, counts_tuple)
                )
                for score, history in sorted(heap, key=lambda x: x[0], reverse=True)
            ]

        return results

    def _solve_top_n_bisect(
        self,
        feasible_allocations: list[Allocation],
        n_per_allocation: int,
        *,
        max_states_per_layer: int,
    ) -> dict[tuple[int, ...], list[ExactSolution]]:
        """Fallback: recursive bisection over allocations.

        Deduplicates allocations by counts tuple, sorts lexicographically
        (to cluster similar allocations), then recursively bisects until
        each batch fits within max_states_per_layer.

        Single-allocation base case builds with no threshold (always succeeds).
        """
        names = self.model.mixer_names

        # Deduplicate by counts tuple, preserving one Allocation per unique tuple
        seen: dict[tuple[int, ...], Allocation] = {}
        for alloc in feasible_allocations:
            counts = tuple(alloc.get(name, 0) for name in names)
            if counts not in seen:
                seen[counts] = alloc

        # Sort lexicographically to cluster similar allocations
        sorted_items = sorted(seen.items(), key=lambda x: x[0])
        sorted_allocs = [alloc for _, alloc in sorted_items]

        results: dict[tuple[int, ...], list[ExactSolution]] = {}
        self._bisect_solve(sorted_allocs, n_per_allocation, max_states_per_layer, results)
        return results

    def _bisect_solve(
        self,
        allocs: list[Allocation],
        n_per_allocation: int,
        max_states_per_layer: int,
        results: dict[tuple[int, ...], list[ExactSolution]],
    ) -> None:
        """Recursively bisect allocation list until DP build succeeds.

        Args:
            allocs: Batch of allocations to solve.
            n_per_allocation: Maximum solutions per allocation.
            max_states_per_layer: Threshold for DP state space.
            results: Mutable accumulator for results.
        """
        if not allocs:
            return

        # Single allocation: build with no threshold (always succeeds)
        if len(allocs) == 1:
            tables = self._build_constrained_tables(allocs)
            if tables is not None:
                results.update(self._extract_per_allocation_results(tables, n_per_allocation))
            return

        # Try building tables for the whole batch
        tables = self._build_constrained_tables(allocs, max_states_per_layer=max_states_per_layer)
        if tables is not None:
            results.update(self._extract_per_allocation_results(tables, n_per_allocation))
            return

        # Split at midpoint and recurse each half
        mid = len(allocs) // 2
        self._bisect_solve(allocs[:mid], n_per_allocation, max_states_per_layer, results)
        self._bisect_solve(allocs[mid:], n_per_allocation, max_states_per_layer, results)

    # ------------------------------------------------------------------
    # Testing / enumeration helpers
    # ------------------------------------------------------------------

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
            score = self.model.true_score(placement)  # type: ignore[attr-defined]
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
            score = self.model.true_score(placement)  # type: ignore[attr-defined]
            results.append((placement, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
