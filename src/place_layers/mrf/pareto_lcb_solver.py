"""Pareto LCB solver using Viterbi DP with (mean, variance) frontier tracking.

Each DP state maintains a Pareto frontier of (m, v) pairs. At the final layer,
solutions are scored as m - β·sqrt(c·(1+v)). Since this objective is concave
in v for fixed c, the optimal point for any β ≥ 0 lies on the upper concave
envelope of the (v, m) Pareto frontier — enabling principled pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation
from place_layers.mrf.solver import ExactSolution

if TYPE_CHECKING:
    from place_layers.mrf.model import MRFModelProtocol

import math


@dataclass
class ParetoEntry:
    """Single point on a Pareto frontier with backtracking info."""

    m: float  # accumulated mean
    v: float  # accumulated variance
    layer: int  # current layer index
    prev_key: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    prev_idx: int = -1  # index into previous state's frontier
    choice: int = -1  # mixer index chosen at this layer


class ParetoLCBSolver:
    """Viterbi DP with (m, v) Pareto frontier tracking.

    Each state tracks a Pareto frontier of (mean_sum, var_sum) pairs.
    Pruning uses upper concave envelope: since m - β·sqrt(c·(1+v))
    is concave in v, only hull points can be optimal for some β ≥ 0.
    """

    def __init__(
        self,
        mean_model: "MRFModelProtocol",
        var_model: "MRFModelProtocol",
        noise_scale_sq: float,
    ) -> None:
        self.mean_model = mean_model
        self.var_model = var_model
        self.noise_scale_sq = noise_scale_sq

    def _mean_transition_score(
        self, layer: int, state: tuple[int, ...], new_type: int
    ) -> float:
        """Compute mean model interaction terms (mirrors ExactMRFSolver._transition_score)."""
        model = self.mean_model
        names = model.mixer_names
        score = 0.0

        if len(state) >= 1:
            prev_type = state[-1]
            score += model.pairwise_potential(
                layer - 1, names[prev_type], names[new_type]
            )

        for r in range(2, model.pairwise_max_range + 1):
            if len(state) >= r and layer >= r:
                back_type = state[-r]
                score += model.skip_pairwise_potential(
                    layer - r, r, names[back_type], names[new_type]
                )

        if model.triplet_scale > 0 and len(state) >= 2 and layer >= 2:
            score += model.triplet_potential(
                layer - 2, names[state[-2]], names[state[-1]], names[new_type]
            )

        return score

    def _var_transition_score(
        self, layer: int, state: tuple[int, ...], new_type: int
    ) -> float:
        """Compute var model interaction terms."""
        model = self.var_model
        names = model.mixer_names
        score = 0.0

        if len(state) >= 1:
            prev_type = state[-1]
            score += model.pairwise_potential(
                layer - 1, names[prev_type], names[new_type]
            )

        for r in range(2, model.pairwise_max_range + 1):
            if len(state) >= r and layer >= r:
                back_type = state[-r]
                score += model.skip_pairwise_potential(
                    layer - r, r, names[back_type], names[new_type]
                )

        if model.triplet_scale > 0 and len(state) >= 2 and layer >= 2:
            score += model.triplet_potential(
                layer - 2, names[state[-2]], names[state[-1]], names[new_type]
            )

        return score

    def solve_top_n_lcb(
        self,
        feasible_allocations: list[Allocation],
        n: int,
        beta: float,
        max_frontier_size: int = 500,
    ) -> list[ExactSolution]:
        """Pareto DP with upper concave hull pruning.

        Structure mirrors ExactMRFSolver.solve_top_n:
        - State: (history, counts) → list[ParetoEntry]
        - Transition: add mean + var potentials from both models
        - Pruning: Pareto dominance → upper concave hull → cap
        - Final: score = m - β·sqrt(c·(1+v)), top-n across feasible states
        - Backtrack each to reconstruct placement

        Args:
            feasible_allocations: List of allocations to consider.
            n: Number of top solutions to return.
            beta: LCB risk parameter. 0 = pure exploitation.
            max_frontier_size: Safety cap after hull pruning.

        Returns:
            List of ExactSolution sorted by LCB score descending, length <= n.
        """
        N = self.mean_model.num_layers
        K = self.mean_model.num_mixers
        names = self.mean_model.mixer_names
        m_mem = self.mean_model.memory
        v_mem = self.var_model.memory
        memory = max(m_mem, v_mem)

        c = self.noise_scale_sq

        # Convert allocations to target count tuples and build feasible set
        feasible_set: set[tuple[int, ...]] = set()
        for alloc in feasible_allocations:
            target = tuple(alloc.get(name, 0) for name in names)
            if sum(target) == N:
                feasible_set.add(target)

        if not feasible_set:
            return []

        # Max counts per mixer for pruning
        max_counts = [0] * K
        for target in feasible_set:
            for k in range(K):
                if target[k] > max_counts[k]:
                    max_counts[k] = target[k]

        # DP tables: layer -> {(history, counts) -> list[ParetoEntry]}
        dp: list[dict[tuple[tuple[int, ...], tuple[int, ...]], list[ParetoEntry]]] = [{} for _ in range(N)]

        # Base case: first layer
        for j in range(K):
            if 1 > max_counts[j]:
                continue
            counts = tuple(1 if k == j else 0 for k in range(K))
            history = (j,)
            key = (history, counts)

            m_val = self.mean_model.unary_potential(0, names[j])
            v_val = self.var_model.unary_potential(0, names[j])

            entry = ParetoEntry(
                m=m_val, v=v_val, layer=0, prev_key=None, prev_idx=-1, choice=j
            )
            dp[0][key] = [entry]

        # Forward pass
        for i in range(1, N):
            new_states: dict[tuple[tuple[int, ...], tuple[int, ...]], list[ParetoEntry]] = {}

            for (prev_history, prev_counts), prev_entries in dp[i - 1].items():
                for j in range(K):
                    new_count_j = prev_counts[j] + 1
                    if new_count_j > max_counts[j]:
                        continue

                    new_counts = tuple(
                        prev_counts[k] + (1 if k == j else 0) for k in range(K)
                    )

                    m_unary = self.mean_model.unary_potential(i, names[j])
                    v_unary = self.var_model.unary_potential(i, names[j])
                    m_trans = self._mean_transition_score(i, prev_history, j)
                    v_trans = self._var_transition_score(i, prev_history, j)

                    new_history = (*prev_history, j)[-memory:]
                    new_key = (new_history, new_counts)

                    prev_key_ref = (prev_history, prev_counts)

                    if new_key not in new_states:
                        new_states[new_key] = []

                    frontier = new_states[new_key]
                    for idx, pe in enumerate(prev_entries):
                        new_m = pe.m + m_unary + m_trans
                        new_v = pe.v + v_unary + v_trans

                        frontier.append(
                            ParetoEntry(
                                m=new_m,
                                v=new_v,
                                layer=i,
                                prev_key=prev_key_ref,
                                prev_idx=idx,
                                choice=j,
                            )
                        )

            # Prune each state's frontier
            for key in new_states:
                new_states[key] = _hull_prune(
                    new_states[key], max_frontier_size
                )

            dp[i] = new_states

        # Collect final-layer entries with feasible counts
        scored: list[tuple[float, tuple[tuple[int, ...], tuple[int, ...]], int]] = []
        for (history, counts), entries in dp[N - 1].items():
            if counts in feasible_set:
                for idx, entry in enumerate(entries):
                    lcb = entry.m - beta * math.sqrt(c * (1.0 + entry.v))
                    scored.append((lcb, (history, counts), idx))

        # Sort by LCB descending, take top n
        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored[:n]

        # Backtrack each to reconstruct placement
        results: list[ExactSolution] = []
        for lcb_score, final_key, final_idx in top_scored:
            assignment_indices: list[int] = [0] * N

            current_key = final_key
            current_idx = final_idx
            for layer in range(N - 1, -1, -1):
                entry = dp[layer][current_key][current_idx]
                assignment_indices[layer] = entry.choice
                if entry.prev_key is not None:
                    current_key = entry.prev_key
                    current_idx = entry.prev_idx

            assignments = tuple(names[j] for j in assignment_indices)
            placement = LayerPlacement(assignments=assignments)
            results.append(ExactSolution(placement=placement, true_score=lcb_score))

        return results


def _hull_prune(
    entries: list[ParetoEntry], max_size: int
) -> list[ParetoEntry]:
    """Keep upper concave envelope in (v, m) space.

    1. Remove dominated: (m1,v1) dominates (m2,v2) if m1≥m2 and v1≤v2
    2. Sort remaining by v ascending
    3. Build upper concave hull (non-increasing slope criterion)
    4. If still > max_size: uniformly downsample along v

    This preserves exactly the points that could be optimal for some β ≥ 0
    when the objective is m - β·sqrt(c·(1+v)) (concave in v).
    """
    if len(entries) <= 1:
        return entries

    # Step 1: Pareto dominance filter
    # Sort by m descending, then v ascending for sweep
    entries.sort(key=lambda e: (-e.m, e.v))
    pareto: list[ParetoEntry] = []
    min_v = float("inf")
    for e in entries:
        if e.v < min_v:
            pareto.append(e)
            min_v = e.v

    if len(pareto) <= 2:
        return pareto

    # Step 2: Sort by v ascending
    pareto.sort(key=lambda e: e.v)

    # Step 3: Upper concave hull via monotone chain
    # We want the upper envelope of points (v_i, m_i).
    # A point is on the upper hull if the slopes are non-increasing.
    hull: list[ParetoEntry] = []
    for p in pareto:
        while len(hull) >= 2:
            # Check if the last point is below the line from hull[-2] to p
            a, b = hull[-2], hull[-1]
            # Cross product: (b.v-a.v)*(p.m-a.m) - (b.m-a.m)*(p.v-a.v)
            # For upper hull: remove b if cross >= 0 (b is below or on the line)
            cross = (b.v - a.v) * (p.m - a.m) - (b.m - a.m) * (p.v - a.v)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(p)

    # Step 4: Safety cap via uniform downsampling along v
    if len(hull) > max_size:
        # Keep first and last, uniformly sample in between
        v_min = hull[0].v
        v_max = hull[-1].v
        if v_max > v_min:
            step = (v_max - v_min) / (max_size - 1)
            sampled: list[ParetoEntry] = [hull[0]]
            next_v = v_min + step
            for h in hull[1:]:
                if h.v >= next_v or h is hull[-1]:
                    sampled.append(h)
                    next_v = h.v + step
            hull = sampled[:max_size]

    return hull
