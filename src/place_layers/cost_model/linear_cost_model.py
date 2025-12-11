"""Linear cost model for layer placement optimization."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from place_layers.core.types import LayerPlacement
from place_layers.cost_model.base_cost_model import Allocation, BaseCostModel
from place_layers.cost_model.base_cost_model_config import BaseCostModelConfig

# Floating point tolerance for cost comparisons.
# This handles accumulated rounding errors in floating point arithmetic
# (e.g., 8.4 - 6.0 = 2.3999999999999986 instead of 2.4).
_COST_TOLERANCE = 1e-9


class LinearCostModel(BaseCostModel):
    """
    Linear cost model based on mixer time contributions.

    For autoregressive decoding, time-per-token is additive across layers.
    If mixer type i takes time α_i per layer, then:

        cost(config) = Σ_i α[mixer_idx[i]]

    This is equivalent to:

        cost = Σ_j (count_j * α_j)

    where count_j is the number of layers using mixer j.

    Key insight from throughput modeling:
    - Throughput T = 1 / (Σ α_i * p_i) where p_i = n_i / N (fraction on mixer i)
    - So 1/T (time per token) is linear in the mixer fractions
    - This is a weighted harmonic mean of per-mixer throughputs

    Properties:
    - Cost is decomposable: swapping a layer changes cost by (α_new - α_old)
    - Cost depends only on counts, not which specific layers
    - But performance depends on WHICH layers, creating the optimization problem

    Allocation Navigation:
    - All allocation operations are exact (no discretization, no floating point hacks)
    - allocation_at_cost_bound: O(N^{K-2} * K) via recursive enumeration with closed-form base case
    - allocation_neighbors: O(K²) with exact ordering
    """

    def __init__(self, config: "LinearCostModelConfig") -> None:
        self.config = config
        self._mixer_costs = tuple(config.mixer_costs)
        # Precompute sorted mixer indices for efficient operations
        self._mixers_by_cost_asc = tuple(
            sorted(range(len(self._mixer_costs)), key=lambda i: self._mixer_costs[i])
        )
        self._mixers_by_cost_desc = tuple(reversed(self._mixers_by_cost_asc))

    # =========================================================================
    # Assignment-level operations
    # =========================================================================

    def cost(self, placement: LayerPlacement) -> float:
        """Compute total cost as sum of per-layer mixer costs."""
        return sum(self._mixer_costs[mixer_idx] for mixer_idx in placement.assignments)

    def cost_delta(self, layer_idx: int, old_mixer: int, new_mixer: int) -> float:
        """Compute cost change from swapping a layer's mixer assignment."""
        return self._mixer_costs[new_mixer] - self._mixer_costs[old_mixer]

    def min_cost(self, num_layers: int) -> float:
        """Cost when all layers use the cheapest mixer."""
        return num_layers * self._mixer_costs[self._mixers_by_cost_asc[0]]

    def max_cost(self, num_layers: int) -> float:
        """Cost when all layers use the most expensive mixer."""
        return num_layers * self._mixer_costs[self._mixers_by_cost_desc[0]]

    # =========================================================================
    # Allocation-level operations
    # =========================================================================

    def allocation_cost(self, allocation: Allocation) -> float:
        """
        Compute cost of an allocation. O(K).

        Args:
            allocation: Counts per mixer.

        Returns:
            Total cost = Σ (count_i * α_i)
        """
        return sum(n * c for n, c in zip(allocation, self._mixer_costs, strict=True))

    def min_cost_allocation(self, num_layers: int) -> Allocation:
        """
        Return allocation with minimum cost. O(K).

        All layers go to the cheapest mixer.
        """
        allocation = [0] * self.num_mixers
        allocation[self._mixers_by_cost_asc[0]] = num_layers
        return tuple(allocation)

    def max_cost_allocation(self, num_layers: int) -> Allocation:
        """
        Return allocation with maximum cost. O(K).

        All layers go to the most expensive mixer.
        """
        allocation = [0] * self.num_mixers
        allocation[self._mixers_by_cost_desc[0]] = num_layers
        return tuple(allocation)

    def allocation_at_cost_bound(
        self,
        num_layers: int,
        cost_bound: float,
        from_above: bool = True,
    ) -> Allocation | None:
        """
        Find allocation closest to cost bound.

        This solves a constrained integer optimization problem on the simplex:
            {(n₀, ..., n_{K-1}) : Σnᵢ = N, nᵢ ≥ 0 integers}

        The algorithm uses recursive enumeration with a closed-form base case
        for K=2 mixers. For each outer mixer, we enumerate possible counts and
        solve the remaining subproblem recursively.

        Complexity: O(N^{K-2} * K) with effective pruning for typical cases.
        For K ≤ 5 and reasonable N, this is fast and exact.

        Args:
            num_layers: Total number of layers (N).
            cost_bound: Target cost value (C).
            from_above: If True, find largest cost ≤ bound (maximize).
                       If False, find smallest cost ≥ bound (minimize).

        Returns:
            Allocation closest to the bound, or None if infeasible.
        """
        return self._solve_allocation_bound(
            num_layers=num_layers,
            cost_bound=cost_bound,
            maximize=from_above,
        )

    def _solve_allocation_bound(
        self,
        num_layers: int,
        cost_bound: float,
        maximize: bool,
    ) -> Allocation | None:
        """
        Unified solver for allocation bound problems.

        Handles both directions:
        - maximize=True:  max Σαᵢnᵢ s.t. Σαᵢnᵢ ≤ C (approach bound from below)
        - maximize=False: min Σαᵢnᵢ s.t. Σαᵢnᵢ ≥ C (approach bound from above)

        The algorithm sorts mixers by cost (descending for maximize, ascending
        for minimize), then recursively enumerates counts on the "extreme" mixer
        first. This greedy ordering enables effective pruning.
        """
        # Check feasibility
        if maximize:
            if cost_bound < self.min_cost(num_layers) - _COST_TOLERANCE:
                return None
        else:
            if cost_bound > self.max_cost(num_layers) + _COST_TOLERANCE:
                return None

        num_mixers = self.num_mixers
        mixer_costs = self._mixer_costs

        # Sort mixers: descending for maximize (try expensive first),
        # ascending for minimize (try cheap first)
        sorted_indices = sorted(
            range(num_mixers),
            key=lambda i: mixer_costs[i],
            reverse=maximize,
        )
        sorted_costs = tuple(mixer_costs[i] for i in sorted_indices)

        # Sentinel values for infeasible solutions
        infeasible_cost = -float("inf") if maximize else float("inf")

        def is_feasible(cost: float) -> bool:
            if maximize:
                return cost > -float("inf")
            else:
                return cost < float("inf")

        def is_better(new_cost: float, old_cost: float) -> bool:
            return new_cost > old_cost if maximize else new_cost < old_cost

        def satisfies_bound(cost: float, bound: float) -> bool:
            if maximize:
                return cost <= bound + _COST_TOLERANCE
            else:
                return cost >= bound - _COST_TOLERANCE

        def solve_recursive(
            remaining_n: int,
            remaining_bound: float,
            mixer_start: int,
        ) -> tuple[float, tuple[int, ...]]:
            """Recursive solver returning (cost, allocation_in_sorted_order)."""
            remaining_mixers = num_mixers - mixer_start

            # Base case: no mixers left
            if remaining_mixers == 0:
                if remaining_n == 0:
                    return (0.0, ())
                return (infeasible_cost, ())

            # Base case: single mixer
            if remaining_mixers == 1:
                cost = remaining_n * sorted_costs[mixer_start]
                if satisfies_bound(cost, remaining_bound):
                    return (cost, (remaining_n,))
                return (infeasible_cost, ())

            # Base case: two mixers - closed form solution
            if remaining_mixers == 2:
                return self._solve_two_mixer_base_case(
                    remaining_n=remaining_n,
                    remaining_bound=remaining_bound,
                    alpha_first=sorted_costs[mixer_start],
                    alpha_second=sorted_costs[mixer_start + 1],
                    maximize=maximize,
                )

            # General case: enumerate counts on first mixer, recurse
            alpha_first = sorted_costs[mixer_start]
            best_cost = infeasible_cost
            best_alloc: tuple[int, ...] = ()

            # Determine iteration bounds based on direction
            if maximize:
                # Try high counts first (more on expensive = higher cost)
                max_n = min(
                    remaining_n,
                    int(remaining_bound / alpha_first) if alpha_first > 0 else remaining_n,
                )
                n_range = range(max_n, -1, -1)
            else:
                # Try high counts first (more on cheap = lower cost)
                n_range = range(remaining_n, -1, -1)

            for n_first in n_range:
                cost_first = n_first * alpha_first
                sub_n = remaining_n - n_first

                if maximize:
                    sub_bound = remaining_bound - cost_first
                    if sub_bound < -_COST_TOLERANCE:
                        continue
                else:
                    sub_bound = remaining_bound - cost_first
                    # Pruning: check if remaining mixers can meet the requirement
                    max_remaining = sub_n * sorted_costs[-1] if sub_n > 0 else 0.0
                    if cost_first + max_remaining < remaining_bound - _COST_TOLERANCE:
                        continue

                sub_cost, sub_alloc = solve_recursive(sub_n, sub_bound, mixer_start + 1)

                if is_feasible(sub_cost):
                    total_cost = cost_first + sub_cost
                    if satisfies_bound(total_cost, remaining_bound) and is_better(
                        total_cost, best_cost
                    ):
                        best_cost = total_cost
                        best_alloc = (n_first,) + sub_alloc

                        # Early termination for maximize: can't do better than exact match
                        if maximize and abs(total_cost - remaining_bound) < _COST_TOLERANCE:
                            break

            return (best_cost, best_alloc)

        best_cost, sorted_alloc = solve_recursive(num_layers, cost_bound, 0)

        if not is_feasible(best_cost):
            return None

        # Convert from sorted order to original mixer order
        alloc = [0] * num_mixers
        for i, count in enumerate(sorted_alloc):
            alloc[sorted_indices[i]] = count

        return tuple(alloc)

    def _solve_two_mixer_base_case(
        self,
        remaining_n: int,
        remaining_bound: float,
        alpha_first: float,
        alpha_second: float,
        maximize: bool,
    ) -> tuple[float, tuple[int, int]]:
        """
        Closed-form solution for the two-mixer allocation problem.

        For maximize (costs sorted descending): alpha_first > alpha_second
            cost = n₁·α₁ + n₂·α₂ = n·α₂ + n₁·(α₁ - α₂)
            Maximize n₁ subject to cost ≤ bound

        For minimize (costs sorted ascending): alpha_first < alpha_second
            cost = n₁·α₁ + n₂·α₂ = n·α₂ - n₁·(α₂ - α₁)
            Maximize n₁ (more on cheap) subject to cost ≥ bound
        """
        infeasible: tuple[float, tuple[int, int]] = (
            (-float("inf"), (0, 0)) if maximize else (float("inf"), (0, 0))
        )

        if maximize:
            # Base: all on cheaper mixer (alpha_second)
            base_cost = remaining_n * alpha_second
            if base_cost > remaining_bound + _COST_TOLERANCE:
                return infeasible

            delta = alpha_first - alpha_second
            if delta <= 0:
                return (base_cost, (0, remaining_n))

            # How many can we upgrade to expensive?
            budget_for_upgrades = remaining_bound - base_cost
            n_first_float = budget_for_upgrades / delta
            max_n_first = min(remaining_n, int(n_first_float))

            # Try candidates, handling floating point edge cases
            for candidate in (max_n_first + 1, max_n_first):
                if 0 <= candidate <= remaining_n:
                    cost = candidate * alpha_first + (remaining_n - candidate) * alpha_second
                    if cost <= remaining_bound + _COST_TOLERANCE:
                        return (cost, (candidate, remaining_n - candidate))

            return (base_cost, (0, remaining_n))

        else:
            # Base: all on expensive mixer (alpha_second, since sorted ascending)
            max_cost = remaining_n * alpha_second
            if max_cost < remaining_bound - _COST_TOLERANCE:
                return infeasible

            delta = alpha_second - alpha_first
            if delta <= 0:
                return (max_cost, (0, remaining_n))

            # How many can we downgrade to cheap while meeting bound?
            budget_for_downgrades = max_cost - remaining_bound
            n_first_float = budget_for_downgrades / delta
            max_n_first = min(remaining_n, int(n_first_float))

            # Try candidates (prefer more on cheap = lower cost)
            for candidate in (max_n_first + 1, max_n_first):
                if 0 <= candidate <= remaining_n:
                    cost = candidate * alpha_first + (remaining_n - candidate) * alpha_second
                    if cost >= remaining_bound - _COST_TOLERANCE:
                        return (cost, (candidate, remaining_n - candidate))

            return (max_cost, (0, remaining_n))

    def allocation_neighbors(
        self,
        allocation: Allocation,
        direction: Literal["increasing", "decreasing"],
    ) -> Iterator[tuple[Allocation, float]]:
        """
        Yield neighboring allocations in cost order. O(K² log K).

        For linear costs, the ordering is exact.

        Args:
            allocation: Current allocation.
            direction: "increasing" for more expensive neighbors,
                      "decreasing" for cheaper neighbors.

        Yields:
            (neighbor, cost_delta) pairs sorted by |delta|.
        """
        moves: list[tuple[float, int, int]] = []

        for from_mixer in range(self.num_mixers):
            if allocation[from_mixer] == 0:
                continue

            for to_mixer in range(self.num_mixers):
                if from_mixer == to_mixer:
                    continue

                delta = self._mixer_costs[to_mixer] - self._mixer_costs[from_mixer]

                # Filter by direction
                if direction == "increasing" and delta <= 0:
                    continue
                if direction == "decreasing" and delta >= 0:
                    continue

                moves.append((abs(delta), from_mixer, to_mixer))

        # Sort by |delta| (smallest change first)
        moves.sort(key=lambda x: x[0])

        for abs_delta, from_mixer, to_mixer in moves:
            new_allocation = list(allocation)
            new_allocation[from_mixer] -= 1
            new_allocation[to_mixer] += 1

            delta = self._mixer_costs[to_mixer] - self._mixer_costs[from_mixer]
            yield tuple(new_allocation), delta

    # =========================================================================
    # Convenience properties
    # =========================================================================

    @property
    def num_mixers(self) -> int:
        """Number of mixer types."""
        return len(self._mixer_costs)

    @property
    def mixer_costs(self) -> tuple[float, ...]:
        """Per-mixer cost coefficients (α values)."""
        return self._mixer_costs

    @property
    def cheapest_mixer(self) -> int:
        """Index of the cheapest (fastest) mixer."""
        return self._mixers_by_cost_asc[0]

    @property
    def most_expensive_mixer(self) -> int:
        """Index of the most expensive (slowest) mixer."""
        return self._mixers_by_cost_desc[0]


class LinearCostModelConfig(BaseCostModelConfig):
    """
    Configuration for linear cost model.

    The mixer_costs list should be ordered from most expensive (powerful)
    to cheapest (lightweight). For example, with 3 mixers:
    - mixer_costs[0] = 1.0  (full attention, slowest)
    - mixer_costs[1] = 0.5  (sliding window, medium)
    - mixer_costs[2] = 0.2  (linear/SSM, fastest)

    These values can be obtained by profiling or fitting to throughput
    measurements. The key relationship is:

        time_per_token ∝ Σ (mixer_costs[i] * fraction_of_layers_i)

    Or equivalently:

        throughput = 1 / (N * Σ (mixer_costs[i] * p_i))

    where N is total layers and p_i is fraction using mixer i.
    """

    _impl_class: ClassVar[type[BaseCostModel]] = LinearCostModel

    cost_model_type: Literal["linear"] = "linear"

    mixer_costs: list[float] = Field(
        description="Cost coefficient (α) for each mixer type. "
        "Index 0 is typically the most expensive (e.g., full attention), "
        "higher indices are cheaper (e.g., SSM). "
        "Values represent relative time contribution per layer.",
    )

    @model_validator(mode="after")
    def validate_mixer_costs(self) -> Self:
        """Validate mixer cost configuration."""
        if len(self.mixer_costs) < 2:
            raise ValueError("Must have at least 2 mixer types")

        if any(c <= 0 for c in self.mixer_costs):
            raise ValueError("All mixer costs must be positive")

        return self
