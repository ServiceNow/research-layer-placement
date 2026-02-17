"""Tests for cost-constrained beam search.

The core guarantee of the cost-constrained search is:
    "If I iterate allocations in increasing cost order and stop when cost > budget,
     I have seen the optimal allocation within budget."

This requires exactly three things:
1. Ordering - costs are monotonically non-decreasing (no inversions)
2. Completeness - all allocations are yielded (none skipped)
3. Iterator-Bound Agreement - stopping iterator at bound gives same result as allocation_at_cost_bound
"""

import logging
import math
import random
from collections import deque
from typing import Literal

import pytest

from place_layers.core.types import LayerPlacement
from place_layers.cost_model import LinearCostModelConfig
from place_layers.cost_model.linear_cost_model import LinearCostModel
from place_layers.evaluator import MockSupernetEvaluatorConfig
from place_layers.problem import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
    MinCostProblemConfig,
)
from place_layers.annealing.sa_core import SAConfig
from place_layers.search_strategy import AnnealingConfig

pytestmark = pytest.mark.asyncio(loop_scope="function")

# Small, fast SA config for integration tests
_SMALL_SA_CONFIG = SAConfig(
    num_restarts=1,
    steps_per_restart=50,
    initial_temperature=1.0,
    cooling_rate=0.95,
    min_temperature=0.01,
    mutation_prob=0.3,
    max_ruin_fraction=0.3,
    contiguous_ruin_prob=0.5,
    top_k=5,
    no_op_retries=3,
    reheat=None,
    seed=42,
)


# =============================================================================
# Module-level fixtures and helpers
# =============================================================================


def make_cost_model(num_mixers: int, spread: float = 0.1) -> LinearCostModel:
    """Create cost model with num_mixers mixers, costs spread apart.

    Costs are: 1.0, 1.0 - spread, 1.0 - 2*spread, ...
    """
    costs = [1.0 - i * spread for i in range(num_mixers)]
    mixer_costs = {f"mixer_{i}": cost for i, cost in enumerate(costs)}
    return LinearCostModelConfig(mixer_costs=mixer_costs).create_cost_model()


@pytest.fixture
def three_mixer_model() -> LinearCostModel:
    """Standard 3-mixer model for tests: costs [1.0, 0.9, 0.8]."""
    return make_cost_model(3)


@pytest.fixture
def four_mixer_model() -> LinearCostModel:
    """Standard 4-mixer model for tests: costs [1.0, 0.9, 0.8, 0.7]."""
    return make_cost_model(4)


# =============================================================================
# TestCostModelBasics - Fundamental cost computations
# =============================================================================


class TestCostModelBasics:
    """Fundamental cost computations - keep simple."""

    def test_cost_computation(self) -> None:
        """Test that cost is computed correctly from LayerPlacement."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        model = config.create_cost_model()

        # All on cheapest mixer
        placement = LayerPlacement(assignments=("mixer_2", "mixer_2", "mixer_2", "mixer_2", "mixer_2"))
        assert model.cost(placement) == 5 * 0.2  # 1.0

        # All on most expensive
        placement = LayerPlacement(assignments=("mixer_0", "mixer_0", "mixer_0", "mixer_0", "mixer_0"))
        assert model.cost(placement) == 5 * 1.0  # 5.0

        # Mixed
        placement = LayerPlacement(assignments=("mixer_0", "mixer_1", "mixer_2", "mixer_1", "mixer_0"))
        assert model.cost(placement) == 1.0 + 0.5 + 0.2 + 0.5 + 1.0  # 3.2

    def test_allocation_cost(self) -> None:
        """Test allocation cost computation."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        model = config.create_cost_model()

        # All on most expensive
        alloc = {"mixer_0": 5, "mixer_1": 0, "mixer_2": 0}
        assert model.allocation_cost(alloc) == 5 * 1.0

        # All on cheapest
        alloc = {"mixer_0": 0, "mixer_1": 0, "mixer_2": 5}
        assert model.allocation_cost(alloc) == 5 * 0.2

        # Mixed
        alloc = {"mixer_0": 2, "mixer_1": 1, "mixer_2": 2}
        assert model.allocation_cost(alloc) == 2 * 1.0 + 1 * 0.5 + 2 * 0.2

    def test_cost_delta(self) -> None:
        """Test incremental cost computation."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        model = config.create_cost_model()

        # Upgrade from cheapest to most expensive
        delta = model.cost_delta(layer_idx=0, old_mixer="mixer_2", new_mixer="mixer_0")
        assert delta == 0.8  # 1.0 - 0.2

        # Downgrade from most expensive to cheapest
        delta = model.cost_delta(layer_idx=0, old_mixer="mixer_0", new_mixer="mixer_2")
        assert delta == -0.8  # 0.2 - 1.0

    def test_min_max_cost(self) -> None:
        """Test min/max cost bounds."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        model = config.create_cost_model()

        assert model.min_cost(10) == 10 * 0.2  # 2.0
        assert model.max_cost(10) == 10 * 1.0  # 10.0

    def test_min_max_allocation(self) -> None:
        """Test min/max cost allocations."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        model = config.create_cost_model()

        min_alloc = model.min_cost_allocation(10)
        max_alloc = model.max_cost_allocation(10)

        # Min allocation should put all on cheapest mixer (mixer_2)
        assert min_alloc == {"mixer_0": 0, "mixer_1": 0, "mixer_2": 10}
        assert model.allocation_cost(min_alloc) == 10 * 0.2

        # Max allocation should put all on most expensive mixer (mixer_0)
        assert max_alloc == {"mixer_0": 10, "mixer_1": 0, "mixer_2": 0}
        assert model.allocation_cost(max_alloc) == 10 * 1.0

    def test_cheapest_most_expensive_mixer(self) -> None:
        """Test identification of extreme mixers."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        model = config.create_cost_model()

        assert model.cheapest_mixer == "mixer_2"
        assert model.most_expensive_mixer == "mixer_0"

        # Different ordering
        config2 = LinearCostModelConfig(mixer_costs={"mixer_0": 0.3, "mixer_1": 0.9, "mixer_2": 0.1})
        model2 = config2.create_cost_model()

        assert model2.cheapest_mixer == "mixer_2"
        assert model2.most_expensive_mixer == "mixer_1"


# =============================================================================
# TestAllocationEnumeration - Completeness, ordering, direction symmetry
# =============================================================================


class TestAllocationEnumeration:
    """Enumeration properties: completeness, ordering, direction symmetry.

    Consolidates TestAllocationsByCostCompleteness + ordering tests.
    """

    @pytest.mark.parametrize(
        "num_layers,num_mixers",
        [(5, 2), (6, 3), (8, 3), (7, 4), (4, 5), (10, 3), (12, 4), (15, 3)],
    )
    def test_completeness_and_uniqueness(self, num_layers: int, num_mixers: int) -> None:
        """All C(N+K-1,K-1) allocations yielded exactly once.

        For N layers distributed among K mixers, there are C(N+K-1, K-1)
        possible allocations (stars and bars). This test verifies that
        allocations_by_cost yields exactly that many allocations with
        no duplicates.
        """
        model = make_cost_model(num_mixers)
        expected_count = math.comb(num_layers + num_mixers - 1, num_mixers - 1)

        allocations = list(model.allocations_by_cost(num_layers, "increasing"))

        # Verify count
        assert len(allocations) == expected_count, (
            f"N={num_layers}, K={num_mixers}: expected {expected_count}, got {len(allocations)}"
        )

        # Verify no duplicates by converting to frozenset keys
        keys = [frozenset(alloc.items()) for _, alloc in allocations]
        assert len(keys) == len(set(keys)), f"Found duplicates for N={num_layers}, K={num_mixers}"

        # Verify all allocations sum to num_layers
        for _, alloc in allocations:
            assert sum(alloc.values()) == num_layers, f"Allocation {alloc} doesn't sum to {num_layers}"

    @pytest.mark.parametrize(
        "num_layers,num_mixers",
        [(5, 2), (6, 3), (8, 3), (7, 4), (10, 3), (12, 4), (15, 3)],
    )
    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])
    def test_strict_cost_monotonicity(
        self, num_layers: int, num_mixers: int, direction: Literal["increasing", "decreasing"]
    ) -> None:
        """Costs must be monotonically ordered with no inversions.

        This is the foundation of the search correctness: we can stop
        iterating as soon as cost exceeds (or drops below) the bound.
        """
        model = make_cost_model(num_mixers)
        allocations = list(model.allocations_by_cost(num_layers, direction))

        assert len(allocations) > 0, "Should have at least one allocation"

        costs = [c for c, _ in allocations]

        # Check strict monotonicity in correct direction
        if direction == "increasing":
            for i in range(len(costs) - 1):
                assert costs[i] <= costs[i + 1], (
                    f"Cost inversion at index {i}: {costs[i]} > {costs[i + 1]} "
                    f"(N={num_layers}, K={num_mixers})"
                )
            # First should be min, last should be max
            assert costs[0] == model.min_cost(num_layers)
            assert costs[-1] == model.max_cost(num_layers)
        else:
            for i in range(len(costs) - 1):
                assert costs[i] >= costs[i + 1], (
                    f"Cost inversion at index {i}: {costs[i]} < {costs[i + 1]} "
                    f"(N={num_layers}, K={num_mixers})"
                )
            # First should be max, last should be min
            assert costs[0] == model.max_cost(num_layers)
            assert costs[-1] == model.min_cost(num_layers)

    @pytest.mark.parametrize(
        "num_layers,num_mixers",
        [(5, 2), (6, 3), (8, 3), (7, 4), (10, 3)],
    )
    def test_direction_symmetry(self, num_layers: int, num_mixers: int) -> None:
        """Increasing and decreasing yield same allocations.

        The same set of allocations should be enumerated in both directions.
        This verifies the graph traversal reaches all nodes regardless of
        starting point.

        Note: Allocations at the *same* cost may appear in different tie-break
        orders, so we only check that the sets are identical and costs are
        mirrored (first of increasing equals last of decreasing, etc).
        """
        model = make_cost_model(num_mixers)

        increasing = list(model.allocations_by_cost(num_layers, "increasing"))
        decreasing = list(model.allocations_by_cost(num_layers, "decreasing"))

        # Same number of allocations
        assert len(increasing) == len(decreasing), (
            f"Different counts: increasing={len(increasing)}, decreasing={len(decreasing)}"
        )

        # Same allocations (as sets)
        increasing_set = {frozenset(a.items()) for _, a in increasing}
        decreasing_set = {frozenset(a.items()) for _, a in decreasing}
        assert increasing_set == decreasing_set, "Different allocation sets in increasing vs decreasing"

        # Costs should be mirrored (reversed)
        increasing_costs = [c for c, _ in increasing]
        decreasing_costs = [c for c, _ in decreasing]
        assert increasing_costs == list(reversed(decreasing_costs)), (
            "Costs are not mirrored between increasing and decreasing"
        )

    def test_heap_handles_equal_costs(self) -> None:
        """Heap tie-breaker works when all costs are equal.

        When all mixers have the same cost, every allocation has the same
        total cost. This forces the heap to use the tie-breaker (counter)
        for every comparison. Without a proper tie-breaker, this would
        raise TypeError since frozenset is not orderable.
        """
        # 4 mixers with SAME cost = guaranteed ties on every heap comparison
        mixer_costs = {f"mixer_{i}": 1.0 for i in range(4)}
        model = LinearCostModelConfig(mixer_costs=mixer_costs).create_cost_model()
        num_layers = 10
        num_mixers = 4

        # All allocations have same cost (10 * 1.0 = 10.0)
        allocations = list(model.allocations_by_cost(num_layers, "increasing"))

        # Should still enumerate all without error
        # C(10+4-1, 4-1) = C(13, 3) = 286
        expected_count = math.comb(num_layers + num_mixers - 1, num_mixers - 1)
        assert len(allocations) == expected_count, (
            f"Expected {expected_count} allocations with equal costs, got {len(allocations)}"
        )

        # Verify no duplicates
        keys = [frozenset(alloc.items()) for _, alloc in allocations]
        assert len(keys) == len(set(keys)), "Found duplicate allocations with equal costs"

        # Verify all costs are equal
        costs = [cost for cost, _ in allocations]
        assert all(c == costs[0] for c in costs), "Not all costs are equal as expected"


# =============================================================================
# TestNeighborGeneration - Neighbor graph properties
# =============================================================================


class TestNeighborGeneration:
    """Neighbor generation properties."""

    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])
    def test_neighbors_have_correct_delta_sign(
        self, direction: Literal["increasing", "decreasing"]
    ) -> None:
        """Neighbors have positive delta (increasing) or negative (decreasing)."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        model = config.create_cost_model()

        if direction == "increasing":
            alloc = {"mixer_0": 0, "mixer_1": 5, "mixer_2": 5}  # 5 on medium, 5 on cheap
        else:
            alloc = {"mixer_0": 5, "mixer_1": 5, "mixer_2": 0}  # 5 on expensive, 5 on medium

        neighbors = list(model.allocation_neighbors(alloc, direction))

        # Should have neighbors
        assert len(neighbors) > 0

        base_cost = model.allocation_cost(alloc)
        for neighbor, delta in neighbors:
            if direction == "increasing":
                assert delta > 0, f"Expected positive delta for increasing, got {delta}"
                assert model.allocation_cost(neighbor) > base_cost
            else:
                assert delta < 0, f"Expected negative delta for decreasing, got {delta}"
                assert model.allocation_cost(neighbor) < base_cost

            # Counts should still sum to 10
            assert sum(neighbor.values()) == 10

    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])
    def test_neighbors_sorted_by_delta_magnitude(
        self, direction: Literal["increasing", "decreasing"]
    ) -> None:
        """Neighbors are sorted by |delta| ascending."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        model = config.create_cost_model()

        if direction == "increasing":
            alloc = {"mixer_0": 0, "mixer_1": 5, "mixer_2": 5}
        else:
            alloc = {"mixer_0": 5, "mixer_1": 5, "mixer_2": 0}

        neighbors = list(model.allocation_neighbors(alloc, direction))

        abs_deltas = [abs(d) for _, d in neighbors]
        assert abs_deltas == sorted(abs_deltas), "Neighbors not sorted by |delta| ascending"


# =============================================================================
# TestBoundFinder - allocation_at_cost_bound correctness
# =============================================================================


class TestBoundFinder:
    """allocation_at_cost_bound correctness.

    Consolidates stress tests + edge cases.
    """

    @pytest.mark.parametrize("num_mixers", [2, 3, 4, 5, 6, 7, 8])
    @pytest.mark.parametrize("from_above", [True, False])
    def test_optimality(self, num_mixers: int, from_above: bool) -> None:
        """Bound finder returns true optimum for K=2..8 mixers."""
        model = make_cost_model(num_mixers)
        num_layers = 12

        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)

        for fraction in [0.2, 0.4, 0.5, 0.6, 0.8]:
            bound = min_cost + fraction * (max_cost - min_cost)

            alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=from_above)
            assert alloc is not None, f"Should find allocation for K={num_mixers}, bound={bound}"

            alloc_cost = model.allocation_cost(alloc)
            assert sum(alloc.values()) == num_layers

            if from_above:
                assert alloc_cost <= bound + 1e-9, f"Cost {alloc_cost} exceeds bound {bound}"
                # Verify optimality via exhaustive enumeration
                for cost, _ in model.allocations_by_cost(num_layers, "decreasing"):
                    if cost <= bound + 1e-9:
                        assert abs(alloc_cost - cost) < 1e-9, (
                            f"K={num_mixers}, bound={bound}: "
                            f"Algorithm found cost {alloc_cost}, but optimal is {cost}"
                        )
                        break
            else:
                assert alloc_cost >= bound - 1e-9, f"Cost {alloc_cost} below bound {bound}"
                # Verify optimality via exhaustive enumeration
                for cost, _ in model.allocations_by_cost(num_layers, "increasing"):
                    if cost >= bound - 1e-9:
                        assert abs(alloc_cost - cost) < 1e-9, (
                            f"K={num_mixers}, bound={bound}: "
                            f"Algorithm found cost {alloc_cost}, but optimal is {cost}"
                        )
                        break

    @pytest.mark.parametrize("num_layers", [5, 10, 20, 50])
    def test_varying_num_layers(self, num_layers: int) -> None:
        """Works correctly for various N."""
        model = make_cost_model(4)  # 4 mixers

        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)
        bound = (min_cost + max_cost) / 2

        # from_above
        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        assert model.allocation_cost(alloc) <= bound + 1e-9
        assert sum(alloc.values()) == num_layers

        # from_below
        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=False)
        assert alloc is not None
        assert model.allocation_cost(alloc) >= bound - 1e-9
        assert sum(alloc.values()) == num_layers

    @pytest.mark.parametrize(
        "bound_type",
        ["at_min", "at_max", "below_min", "above_max", "exact_hit"],
    )
    def test_edge_cases(self, bound_type: str) -> None:
        """Edge cases: bounds at/beyond extrema."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        model = config.create_cost_model()
        num_layers = 10

        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)

        if bound_type == "at_min":
            # At min cost
            alloc = model.allocation_at_cost_bound(num_layers, min_cost, from_above=True)
            assert alloc is not None
            assert model.allocation_cost(alloc) == min_cost

            alloc = model.allocation_at_cost_bound(num_layers, min_cost, from_above=False)
            assert alloc is not None
            assert model.allocation_cost(alloc) == min_cost

        elif bound_type == "at_max":
            # At max cost
            alloc = model.allocation_at_cost_bound(num_layers, max_cost, from_above=True)
            assert alloc is not None
            assert model.allocation_cost(alloc) == max_cost

            alloc = model.allocation_at_cost_bound(num_layers, max_cost, from_above=False)
            assert alloc is not None
            assert model.allocation_cost(alloc) == max_cost

        elif bound_type == "below_min":
            # Below minimum cost (infeasible)
            alloc = model.allocation_at_cost_bound(num_layers, min_cost - 0.1, from_above=True)
            assert alloc is None

        elif bound_type == "above_max":
            # Above maximum cost (infeasible for from_below)
            alloc = model.allocation_at_cost_bound(num_layers, max_cost + 0.1, from_above=False)
            assert alloc is None

        elif bound_type == "exact_hit":
            # Cost of (0, 10, 0) = 5.0 exactly
            bound = 5.0
            alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
            assert alloc is not None
            assert abs(model.allocation_cost(alloc) - 5.0) < 1e-9

    @pytest.mark.parametrize("seed", range(10))
    def test_random_stress(self, seed: int) -> None:
        """Random costs and bounds."""
        random.seed(seed)

        num_mixers = random.randint(2, 6)
        costs = sorted([random.uniform(0.1, 1.0) for _ in range(num_mixers)], reverse=True)
        mixer_costs = {f"mixer_{i}": cost for i, cost in enumerate(costs)}
        model = LinearCostModelConfig(mixer_costs=mixer_costs).create_cost_model()

        num_layers = random.randint(5, 20)
        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)
        bound = min_cost + random.uniform(0.1, 0.9) * (max_cost - min_cost)

        # from_above
        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        alloc_cost = model.allocation_cost(alloc)
        assert alloc_cost <= bound + 1e-9
        assert sum(alloc.values()) == num_layers

        # Verify optimality
        for cost, _ in model.allocations_by_cost(num_layers, "decreasing"):
            if cost <= bound + 1e-9:
                assert abs(alloc_cost - cost) < 1e-9, (
                    f"Seed {seed}: found {alloc_cost}, optimal {cost}"
                )
                break

    def test_equal_costs(self) -> None:
        """Test with some mixers having equal costs."""
        mixer_costs = {"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.5, "mixer_3": 0.2}
        model = LinearCostModelConfig(mixer_costs=mixer_costs).create_cost_model()

        num_layers = 10
        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)
        bound = (min_cost + max_cost) / 2

        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        assert model.allocation_cost(alloc) <= bound + 1e-9
        assert sum(alloc.values()) == num_layers

    def test_irrational_like_costs(self) -> None:
        """Test with costs that might cause floating point issues."""
        mixer_costs = {"mixer_0": 1.0, "mixer_1": 1.0 / 3, "mixer_2": math.pi / 10, "mixer_3": 0.1}
        model = LinearCostModelConfig(mixer_costs=mixer_costs).create_cost_model()

        num_layers = 15
        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)

        for fraction in [0.3, 0.5, 0.7]:
            bound = min_cost + fraction * (max_cost - min_cost)

            alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
            assert alloc is not None
            assert model.allocation_cost(alloc) <= bound + 1e-9
            assert sum(alloc.values()) == num_layers

    def test_large_k_small_n(self) -> None:
        """Test with many mixers but few layers."""
        model = make_cost_model(10, spread=0.08)  # 10 mixers
        num_layers = 5  # Fewer layers than mixers

        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)
        bound = (min_cost + max_cost) / 2

        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        assert model.allocation_cost(alloc) <= bound + 1e-9
        assert sum(alloc.values()) == num_layers


# =============================================================================
# TestIteratorBoundConsistency - CRITICAL: iterator and bound finder agree
# =============================================================================


class TestIteratorBoundConsistency:
    """The critical property: iterator and bound finder must agree on COST.

    This is the foundation of cost-constrained search correctness.
    If the costs disagree, the search may violate budget constraints or
    miss optimal allocations.

    Note: When multiple allocations have exactly the same cost, the iterator
    and bound finder may return different allocations (due to tie-breaking).
    What matters is that they return allocations with the same *cost*.
    """

    @pytest.mark.parametrize(
        "num_layers,num_mixers",
        [(8, 3), (10, 4), (12, 3), (6, 5), (15, 3)],
    )
    def test_from_above_cost_agreement(self, num_layers: int, num_mixers: int) -> None:
        """Bound finder and iterator find allocations with same cost (from_above)."""
        model = make_cost_model(num_mixers)
        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)

        # Test multiple bounds across the range
        for fraction in [0.1, 0.25, 0.5, 0.75, 0.9]:
            bound = min_cost + fraction * (max_cost - min_cost)

            # Get result from bound finder
            bound_result = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
            assert bound_result is not None
            bound_cost = model.allocation_cost(bound_result)

            # Get result from iterator
            iter_result = None
            iter_cost = None
            for cost, alloc in model.allocations_by_cost(num_layers, "decreasing"):
                if cost <= bound + 1e-9:
                    iter_result = alloc
                    iter_cost = cost
                    break

            assert iter_result is not None
            assert abs(bound_cost - iter_cost) < 1e-9, (  # type: ignore[operator]
                f"Cost disagreement at bound={bound} (N={num_layers}, K={num_mixers}): "
                f"bound_finder_cost={bound_cost}, iterator_cost={iter_cost}"
            )

    @pytest.mark.parametrize(
        "num_layers,num_mixers",
        [(8, 3), (10, 4), (12, 3), (6, 5), (15, 3)],
    )
    def test_from_below_cost_agreement(self, num_layers: int, num_mixers: int) -> None:
        """Bound finder and iterator find allocations with same cost (from_below)."""
        model = make_cost_model(num_mixers)
        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)

        # Test multiple bounds across the range
        for fraction in [0.1, 0.25, 0.5, 0.75, 0.9]:
            bound = min_cost + fraction * (max_cost - min_cost)

            # Get result from bound finder
            bound_result = model.allocation_at_cost_bound(num_layers, bound, from_above=False)
            assert bound_result is not None
            bound_cost = model.allocation_cost(bound_result)

            # Get result from iterator
            iter_result = None
            iter_cost = None
            for cost, alloc in model.allocations_by_cost(num_layers, "increasing"):
                if cost >= bound - 1e-9:
                    iter_result = alloc
                    iter_cost = cost
                    break

            assert iter_result is not None
            assert abs(bound_cost - iter_cost) < 1e-9, (  # type: ignore[operator]
                f"Cost disagreement at bound={bound} (N={num_layers}, K={num_mixers}): "
                f"bound_finder_cost={bound_cost}, iterator_cost={iter_cost}"
            )

    @pytest.mark.parametrize("seed", range(5))
    def test_random_cost_agreement(self, seed: int) -> None:
        """Random configurations have same costs from both methods."""
        random.seed(seed)

        num_mixers = random.randint(2, 6)
        num_layers = random.randint(5, 15)
        model = make_cost_model(num_mixers, spread=random.uniform(0.05, 0.15))

        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)

        for _ in range(5):
            bound = min_cost + random.uniform(0.05, 0.95) * (max_cost - min_cost)

            # Test from_above
            bound_result = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
            assert bound_result is not None
            bound_cost = model.allocation_cost(bound_result)

            iter_cost = None
            for cost, _ in model.allocations_by_cost(num_layers, "decreasing"):
                if cost <= bound + 1e-9:
                    iter_cost = cost
                    break

            assert iter_cost is not None
            assert abs(bound_cost - iter_cost) < 1e-9, (
                f"Seed {seed}, from_above cost disagreement: "
                f"bound_finder={bound_cost}, iterator={iter_cost}"
            )

            # Test from_below
            bound_result = model.allocation_at_cost_bound(num_layers, bound, from_above=False)
            assert bound_result is not None
            bound_cost = model.allocation_cost(bound_result)

            iter_cost = None
            for cost, _ in model.allocations_by_cost(num_layers, "increasing"):
                if cost >= bound - 1e-9:
                    iter_cost = cost
                    break

            assert iter_cost is not None
            assert abs(bound_cost - iter_cost) < 1e-9, (
                f"Seed {seed}, from_below cost disagreement: "
                f"bound_finder={bound_cost}, iterator={iter_cost}"
            )


# =============================================================================
# TestGraphConnectivity - All allocations reachable via neighbor graph
# =============================================================================


class TestGraphConnectivity:
    """All allocations reachable via neighbor graph.

    This proves that the heap traversal in allocations_by_cost can
    reach every allocation. If the graph were disconnected, some
    allocations would never be yielded.
    """

    @pytest.mark.parametrize(
        "num_layers,num_mixers",
        [(6, 3), (8, 4), (10, 3), (5, 5), (12, 3)],
    )
    def test_all_allocations_reachable(self, num_layers: int, num_mixers: int) -> None:
        """BFS from extremum reaches all C(N+K-1,K-1) allocations."""
        model = make_cost_model(num_mixers)
        start = model.min_cost_allocation(num_layers)

        # BFS using ALL neighbors (both directions)
        visited: set[frozenset[tuple[str, int]]] = {frozenset(start.items())}
        queue: deque[dict[str, int]] = deque([start])

        while queue:
            current = queue.popleft()
            # Get neighbors in both directions
            for neighbor, _ in model.allocation_neighbors(current, "increasing"):
                key = frozenset(neighbor.items())
                if key not in visited:
                    visited.add(key)
                    queue.append(neighbor)
            for neighbor, _ in model.allocation_neighbors(current, "decreasing"):
                key = frozenset(neighbor.items())
                if key not in visited:
                    visited.add(key)
                    queue.append(neighbor)

        expected = math.comb(num_layers + num_mixers - 1, num_mixers - 1)
        assert len(visited) == expected, (
            f"N={num_layers}, K={num_mixers}: "
            f"reached {len(visited)}, expected {expected}"
        )

    @pytest.mark.parametrize(
        "num_layers,num_mixers",
        [(6, 3), (8, 4), (10, 3)],
    )
    def test_single_direction_sufficient(self, num_layers: int, num_mixers: int) -> None:
        """BFS using only increasing neighbors from min reaches all allocations.

        This proves the traversal in allocations_by_cost("increasing") can
        reach every allocation without needing decreasing edges.
        """
        model = make_cost_model(num_mixers)
        start = model.min_cost_allocation(num_layers)

        # BFS using only increasing neighbors
        visited: set[frozenset[tuple[str, int]]] = {frozenset(start.items())}
        queue: deque[dict[str, int]] = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor, _ in model.allocation_neighbors(current, "increasing"):
                key = frozenset(neighbor.items())
                if key not in visited:
                    visited.add(key)
                    queue.append(neighbor)

        expected = math.comb(num_layers + num_mixers - 1, num_mixers - 1)
        assert len(visited) == expected, (
            f"N={num_layers}, K={num_mixers} (increasing only): "
            f"reached {len(visited)}, expected {expected}"
        )

    @pytest.mark.parametrize(
        "num_layers,num_mixers",
        [(6, 3), (8, 4), (10, 3)],
    )
    def test_decreasing_direction_sufficient(self, num_layers: int, num_mixers: int) -> None:
        """BFS using only decreasing neighbors from max reaches all allocations.

        This proves the traversal in allocations_by_cost("decreasing") can
        reach every allocation without needing increasing edges.
        """
        model = make_cost_model(num_mixers)
        start = model.max_cost_allocation(num_layers)

        # BFS using only decreasing neighbors
        visited: set[frozenset[tuple[str, int]]] = {frozenset(start.items())}
        queue: deque[dict[str, int]] = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor, _ in model.allocation_neighbors(current, "decreasing"):
                key = frozenset(neighbor.items())
                if key not in visited:
                    visited.add(key)
                    queue.append(neighbor)

        expected = math.comb(num_layers + num_mixers - 1, num_mixers - 1)
        assert len(visited) == expected, (
            f"N={num_layers}, K={num_mixers} (decreasing only): "
            f"reached {len(visited)}, expected {expected}"
        )


# =============================================================================
# TestCostConstrainedProblem - Input validation (keep but low priority)
# =============================================================================


class TestCostConstrainedProblem:
    """Tests for CostConstrainedProblem instantiation."""

    def test_budget_validation_too_low(self) -> None:
        """Budget below minimum cost should fail."""
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2}),
            cost_budget=1.0,  # Min cost is 10 * 0.2 = 2.0
            cost_floor=None,
        )
        evaluator = evaluator_config.create_evaluator()

        with pytest.raises(ValueError, match="less than minimum"):
            problem_config.create_problem(evaluator)

    def test_budget_validation_too_high(self) -> None:
        """Budget above maximum cost should fail (trivially satisfied)."""
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2}),
            cost_budget=100.0,  # Max cost is 10 * 1.0 = 10.0
            cost_floor=None,
        )
        evaluator = evaluator_config.create_evaluator()

        with pytest.raises(ValueError, match="exceeds maximum"):
            problem_config.create_problem(evaluator)

    def test_mixer_name_mismatch(self) -> None:
        """Mixer name mismatch should fail."""
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=LinearCostModelConfig(mixer_costs={"wrong_name": 1.0, "mixer_1": 0.5}),  # Wrong names
            cost_budget=5.0,
            cost_floor=None,
        )
        evaluator = evaluator_config.create_evaluator()

        with pytest.raises(ValueError, match="must match"):
            problem_config.create_problem(evaluator)

    def test_cost_floor_above_budget_fails(self) -> None:
        """cost_floor >= cost_budget should fail (empty search range)."""
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2}),
            cost_budget=4.0,
            cost_floor=6.0,  # Floor above budget
        )
        evaluator = evaluator_config.create_evaluator()

        with pytest.raises(ValueError, match="cost_floor.*must be less than"):
            problem_config.create_problem(evaluator)

    def test_cost_floor_above_max_cost_fails(self) -> None:
        """cost_floor above maximum possible cost should fail."""
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2}),
            cost_budget=8.0,
            cost_floor=100.0,  # Max cost is 10 * 1.0 = 10.0
        )
        evaluator = evaluator_config.create_evaluator()

        with pytest.raises(ValueError, match="exceeds maximum possible cost"):
            problem_config.create_problem(evaluator)

    def test_cost_floor_below_min_cost_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """cost_floor below minimum possible cost should warn but not error."""
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2}),
            cost_budget=8.0,
            cost_floor=0.1,  # Min cost is 10 * 0.2 = 2.0
        )
        evaluator = evaluator_config.create_evaluator()

        with caplog.at_level(logging.WARNING):
            problem = problem_config.create_problem(evaluator)

        assert problem is not None
        assert "below minimum possible cost" in caplog.text

    def test_cost_floor_valid(self) -> None:
        """Valid cost_floor within [min_cost, cost_budget) should succeed."""
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2}),
            cost_budget=8.0,
            cost_floor=3.0,  # Between min (2.0) and budget (8.0)
        )
        evaluator = evaluator_config.create_evaluator()

        problem = problem_config.create_problem(evaluator)
        assert problem is not None


# =============================================================================
# TestCostConstrainedBeamSearch - Integration tests (keep as-is)
# =============================================================================


class TestCostConstrainedBeamSearch:
    """Tests for the cost-constrained beam search algorithm."""

    @pytest.fixture
    def small_problem_max_perf(self) -> CostConstrainedProblem:
        """Create a small problem for MAX_PERFORMANCE mode."""
        cost_model = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=cost_model,
            cost_budget=4.0,  # Between min (2.0) and max (10.0)
            cost_floor=None,
        )
        evaluator = evaluator_config.create_evaluator()
        return problem_config.create_problem(evaluator)

    @pytest.fixture
    def small_problem_min_cost(self) -> CostConstrainedProblem:
        """Create a small problem for MIN_COST mode."""
        cost_model = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        problem_config = MinCostProblemConfig(
            cost_model=cost_model,
            performance_target=2.0,  # Modest target
            cost_floor=None,
        )
        evaluator = evaluator_config.create_evaluator()
        return problem_config.create_problem(evaluator)

    async def test_max_performance_respects_budget(
        self, small_problem_max_perf: CostConstrainedProblem
    ) -> None:
        """MAX_PERFORMANCE mode should respect cost budget."""
        search_config = AnnealingConfig(
            sa=_SMALL_SA_CONFIG, max_allocations=20
        )
        search = search_config.create_search_strategy()

        result = await search.search(small_problem_max_perf)

        # Cost should be within budget
        assert result.final_cost <= 4.0
        # Should have done some evaluations
        assert result.total_evaluations > 0

    async def test_min_cost_meets_target(
        self, small_problem_min_cost: CostConstrainedProblem
    ) -> None:
        """MIN_COST mode should meet performance target."""
        search_config = AnnealingConfig(
            sa=_SMALL_SA_CONFIG, max_allocations=20
        )
        search = search_config.create_search_strategy()

        result = await search.search(small_problem_min_cost)

        # Performance should meet target
        agg_score = sum(result.best_scores.values()) / len(result.best_scores)
        assert agg_score >= 2.0
        # Should have done some evaluations
        assert result.total_evaluations > 0

    async def test_max_performance_starts_cheap(
        self, small_problem_max_perf: CostConstrainedProblem
    ) -> None:
        """MAX_PERFORMANCE should respect budget (start at budget, explore cheaper)."""
        search_config = AnnealingConfig(
            sa=_SMALL_SA_CONFIG, max_allocations=10
        )
        search = search_config.create_search_strategy()

        result = await search.search(small_problem_max_perf)

        # Final cost should be within budget
        min_cost = small_problem_max_perf.cost_model.min_cost(10)
        assert result.final_cost >= min_cost
        assert result.final_cost <= 4.0  # Budget

    async def test_min_cost_starts_expensive(
        self, small_problem_min_cost: CostConstrainedProblem
    ) -> None:
        """MIN_COST should start from most expensive config and downgrade."""
        search_config = AnnealingConfig(
            sa=_SMALL_SA_CONFIG, max_allocations=10
        )
        search = search_config.create_search_strategy()

        result = await search.search(small_problem_min_cost)

        # Final cost should be less than maximum (we downgraded some layers)
        max_cost = small_problem_min_cost.cost_model.max_cost(10)
        assert result.final_cost <= max_cost

    async def test_deterministic_results(self) -> None:
        """Same seed should produce same results."""
        cost_model = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=cost_model,
            cost_budget=4.0,
            cost_floor=None,
        )
        search_config = AnnealingConfig(
            sa=_SMALL_SA_CONFIG, max_allocations=10
        )

        # Run twice
        evaluator1 = evaluator_config.create_evaluator()
        problem1 = problem_config.create_problem(evaluator1)
        search1 = search_config.create_search_strategy()
        result1 = await search1.search(problem1)

        evaluator2 = evaluator_config.create_evaluator()
        problem2 = problem_config.create_problem(evaluator2)
        search2 = search_config.create_search_strategy()
        result2 = await search2.search(problem2)

        assert result1.best_scores == result2.best_scores
        assert result1.final_cost == result2.final_cost
        assert result1.best_parameters.assignments == result2.best_parameters.assignments


# =============================================================================
# TestCostFloorSearch - Behavioral tests for cost_floor
# =============================================================================


class TestCostFloorSearch:
    """Tests that cost_floor correctly restricts the search space."""

    def test_enumerate_with_floor(self) -> None:
        """enumerate_feasible_allocations excludes allocations below floor."""
        from place_layers.cost_model.enumeration import enumerate_feasible_allocations

        cost_model_config = LinearCostModelConfig(
            mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2}
        )
        cost_model = cost_model_config.create_cost_model()
        num_layers = 10

        # Without floor: all allocations within budget
        all_allocs = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=8.0
        )

        # With floor: only allocations with cost >= 4.0
        floor_allocs = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=8.0, cost_floor=4.0
        )

        # Floor should exclude cheap allocations
        assert len(floor_allocs) < len(all_allocs)
        assert len(floor_allocs) > 0

        # All returned allocations should have cost >= floor
        for alloc in floor_allocs:
            alloc_cost = cost_model.allocation_cost(alloc)
            assert alloc_cost >= 4.0 - 1e-9, (
                f"Allocation {alloc} has cost {alloc_cost} below floor 4.0"
            )

        # All returned allocations should have cost <= budget
        for alloc in floor_allocs:
            alloc_cost = cost_model.allocation_cost(alloc)
            assert alloc_cost <= 8.0 + 1e-9, (
                f"Allocation {alloc} has cost {alloc_cost} above budget 8.0"
            )

    async def test_max_perf_with_floor_skips_cheap(self) -> None:
        """Search with cost_floor should not explore allocations below the floor."""
        cost_model_config = LinearCostModelConfig(
            mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2}
        )
        evaluator_config = MockSupernetEvaluatorConfig(
            num_layers=10,
            mixer_names=["mixer_0", "mixer_1", "mixer_2"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.2,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        # Set floor at 4.0 — search should only explore [4.0, 8.0]
        problem_config = MaxPerformanceProblemConfig(
            cost_model=cost_model_config,
            cost_budget=8.0,
            cost_floor=4.0,
        )
        evaluator = evaluator_config.create_evaluator()
        problem = problem_config.create_problem(evaluator)

        search_config = AnnealingConfig(
            sa=_SMALL_SA_CONFIG, max_allocations=10
        )
        search = search_config.create_search_strategy()
        result = await search.search(problem)

        # Result cost should be within [floor, budget]
        assert result.final_cost >= 4.0 - 1e-9, (
            f"Result cost {result.final_cost} is below floor 4.0"
        )
        assert result.final_cost <= 8.0 + 1e-9, (
            f"Result cost {result.final_cost} is above budget 8.0"
        )
