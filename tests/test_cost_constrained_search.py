"""Tests for cost-constrained beam search."""

import pytest

from place_layers.core.types import LayerPlacement

pytestmark = pytest.mark.asyncio(loop_scope="function")
from place_layers.cost_model import LinearCostModelConfig
from place_layers.cost_model.linear_cost_model import LinearCostModel
from place_layers.evaluator import MockSupernetEvaluatorConfig
from place_layers.problem import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
    MinCostProblemConfig,
)
from place_layers.search_strategy import SwapBeamConfig


class TestAllocationNavigation:
    """Tests for allocation navigation in the linear cost model."""

    @pytest.fixture
    def model(self) -> LinearCostModel:
        """Create a linear cost model for testing."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        return config.create_cost_model()

    def test_allocation_cost(self, model: LinearCostModel) -> None:
        """Test allocation cost computation."""
        # All on most expensive
        alloc = (5, 0, 0)
        assert model.allocation_cost(alloc) == 5 * 1.0

        # All on cheapest
        alloc = (0, 0, 5)
        assert model.allocation_cost(alloc) == 5 * 0.2

        # Mixed
        alloc = (2, 1, 2)
        assert model.allocation_cost(alloc) == 2 * 1.0 + 1 * 0.5 + 2 * 0.2

    def test_min_max_allocation(self, model: LinearCostModel) -> None:
        """Test min/max cost allocations."""
        min_alloc = model.min_cost_allocation(10)
        max_alloc = model.max_cost_allocation(10)

        # Min allocation should put all on cheapest mixer (index 2)
        assert min_alloc == (0, 0, 10)
        assert model.allocation_cost(min_alloc) == 10 * 0.2

        # Max allocation should put all on most expensive mixer (index 0)
        assert max_alloc == (10, 0, 0)
        assert model.allocation_cost(max_alloc) == 10 * 1.0

    def test_allocation_at_cost_bound_from_above(self, model: LinearCostModel) -> None:
        """Test finding allocation at cost bound from above.

        'from_above' means we find the largest cost that is <= bound.
        """
        num_layers = 10

        # At max cost, should return max allocation
        alloc = model.allocation_at_cost_bound(num_layers, 10.0, from_above=True)
        assert alloc == (10, 0, 0)
        assert model.allocation_cost(alloc) == 10.0

        # At min cost exactly, should return min allocation
        # Note: min_cost = 10 * 0.2 = 2.0
        alloc = model.allocation_at_cost_bound(num_layers, 2.0, from_above=True)
        assert alloc is not None
        assert model.allocation_cost(alloc) <= 2.0
        assert sum(alloc) == num_layers

        # At intermediate cost (e.g., 5.0)
        # mixer_costs = [1.0, 0.5, 0.2]
        # (0, 10, 0) has cost 5.0 exactly - this should be the optimal!
        alloc = model.allocation_at_cost_bound(num_layers, 5.0, from_above=True)
        assert alloc is not None
        assert model.allocation_cost(alloc) == 5.0  # Should hit exactly
        assert sum(alloc) == num_layers

        # Below min should return None
        alloc = model.allocation_at_cost_bound(num_layers, 1.0, from_above=True)
        assert alloc is None

    def test_allocation_at_cost_bound_optimality(self, model: LinearCostModel) -> None:
        """Test that allocation_at_cost_bound finds the true optimum.

        For various bounds, verify that no other allocation achieves a
        better cost while satisfying the constraint.
        """
        num_layers = 10

        # Test from_above: should find largest cost <= bound
        for bound in [3.0, 4.5, 6.0, 7.5, 9.0]:
            alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
            assert alloc is not None
            alloc_cost = model.allocation_cost(alloc)
            assert alloc_cost <= bound + 1e-9, f"Cost {alloc_cost} > bound {bound}"

            # Verify optimality by checking all single and two-mixer allocations
            for cost, other_alloc in model.allocations_by_cost(num_layers, "decreasing"):
                if cost <= bound:
                    assert cost <= alloc_cost + 1e-9, (
                        f"Found better allocation {other_alloc} with cost {cost} "
                        f"vs selected {alloc} with cost {alloc_cost}"
                    )
                    break  # First one <= bound should match

        # Test from_below: should find smallest cost >= bound
        for bound in [3.0, 4.5, 6.0, 7.5, 9.0]:
            alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=False)
            assert alloc is not None
            alloc_cost = model.allocation_cost(alloc)
            assert alloc_cost >= bound - 1e-9, f"Cost {alloc_cost} < bound {bound}"

            # Verify optimality by checking all allocations
            for cost, other_alloc in model.allocations_by_cost(num_layers, "increasing"):
                if cost >= bound:
                    assert cost >= alloc_cost - 1e-9, (
                        f"Found better allocation {other_alloc} with cost {cost} "
                        f"vs selected {alloc} with cost {alloc_cost}"
                    )
                    break  # First one >= bound should match

    def test_allocation_at_cost_bound_from_below(self, model: LinearCostModel) -> None:
        """Test finding allocation at cost bound from below.

        'from_below' means we find the smallest cost that is >= bound.
        The greedy algorithm starts from min and upgrades.
        """
        num_layers = 10

        # At min cost, should return min allocation
        alloc = model.allocation_at_cost_bound(num_layers, 2.0, from_above=False)
        assert alloc == (0, 0, 10)
        assert model.allocation_cost(alloc) == 2.0

        # At max cost, should return allocation with cost >= max_cost
        alloc = model.allocation_at_cost_bound(num_layers, 10.0, from_above=False)
        assert alloc is not None
        assert model.allocation_cost(alloc) >= 10.0
        assert sum(alloc) == num_layers

        # At intermediate cost (e.g., 3.0)
        alloc = model.allocation_at_cost_bound(num_layers, 3.0, from_above=False)
        assert alloc is not None
        assert model.allocation_cost(alloc) >= 3.0
        assert sum(alloc) == num_layers

        # Above max should return None
        alloc = model.allocation_at_cost_bound(num_layers, 15.0, from_above=False)
        assert alloc is None

    def test_allocation_neighbors_increasing(self, model: LinearCostModel) -> None:
        """Test generating increasing cost neighbors."""
        alloc = (0, 5, 5)  # 5 on medium, 5 on cheap

        neighbors = list(model.allocation_neighbors(alloc, "increasing"))

        # Should have neighbors
        assert len(neighbors) > 0

        # All neighbors should have higher cost
        base_cost = model.allocation_cost(alloc)
        for neighbor, delta in neighbors:
            assert delta > 0
            assert model.allocation_cost(neighbor) > base_cost
            # Counts should still sum to 10
            assert sum(neighbor) == 10

        # Neighbors should be in order of increasing delta
        deltas = [d for _, d in neighbors]
        assert deltas == sorted(deltas)

    def test_allocation_neighbors_decreasing(self, model: LinearCostModel) -> None:
        """Test generating decreasing cost neighbors."""
        alloc = (5, 5, 0)  # 5 on expensive, 5 on medium

        neighbors = list(model.allocation_neighbors(alloc, "decreasing"))

        # Should have neighbors
        assert len(neighbors) > 0

        # All neighbors should have lower cost
        base_cost = model.allocation_cost(alloc)
        for neighbor, delta in neighbors:
            assert delta < 0
            assert model.allocation_cost(neighbor) < base_cost
            # Counts should still sum to 10
            assert sum(neighbor) == 10

        # Neighbors should be in order of smallest |delta| first
        abs_deltas = [abs(d) for _, d in neighbors]
        assert abs_deltas == sorted(abs_deltas)

    def test_allocations_by_cost_increasing(self, model: LinearCostModel) -> None:
        """Test iterating allocations in increasing cost order."""
        num_layers = 5
        allocations = list(model.allocations_by_cost(num_layers, "increasing"))

        # Should have allocations
        assert len(allocations) > 0

        # First should be min cost, last should be max cost
        first_cost, first_alloc = allocations[0]
        last_cost, last_alloc = allocations[-1]
        assert first_cost == model.min_cost(num_layers)
        assert last_cost == model.max_cost(num_layers)

        # Should be in strictly non-decreasing cost order
        costs = [c for c, _ in allocations]
        assert costs == sorted(costs)

        # All allocations should sum to num_layers
        for _, alloc in allocations:
            assert sum(alloc) == num_layers

    def test_allocations_by_cost_decreasing(self, model: LinearCostModel) -> None:
        """Test iterating allocations in decreasing cost order."""
        num_layers = 5
        allocations = list(model.allocations_by_cost(num_layers, "decreasing"))

        # Should have allocations
        assert len(allocations) > 0

        # First should be max cost, last should be min cost
        first_cost, first_alloc = allocations[0]
        last_cost, last_alloc = allocations[-1]
        assert first_cost == model.max_cost(num_layers)
        assert last_cost == model.min_cost(num_layers)

        # Should be in strictly non-increasing cost order
        costs = [c for c, _ in allocations]
        assert costs == sorted(costs, reverse=True)


class TestLinearCostModel:
    """Tests for the linear cost model."""

    def test_cost_computation(self) -> None:
        """Test that cost is computed correctly."""
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

    def test_cheapest_most_expensive_mixer(self) -> None:
        """Test identification of extreme mixers."""
        config = LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2})
        model = config.create_cost_model()

        assert model.cheapest_mixer == 2
        assert model.most_expensive_mixer == 0

        # Different ordering
        config2 = LinearCostModelConfig(mixer_costs={"mixer_0": 0.3, "mixer_1": 0.9, "mixer_2": 0.1})
        model2 = config2.create_cost_model()

        assert model2.cheapest_mixer == 2
        assert model2.most_expensive_mixer == 1


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
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2}),
            cost_budget=1.0,  # Min cost is 10 * 0.2 = 2.0
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
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=LinearCostModelConfig(mixer_costs={"mixer_0": 1.0, "mixer_1": 0.5, "mixer_2": 0.2}),
            cost_budget=100.0,  # Max cost is 10 * 1.0 = 10.0
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
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=LinearCostModelConfig(mixer_costs={"wrong_name": 1.0, "mixer_1": 0.5}),  # Wrong names
            cost_budget=5.0,
        )
        evaluator = evaluator_config.create_evaluator()

        with pytest.raises(ValueError, match="must match"):
            problem_config.create_problem(evaluator)


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
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=cost_model,
            cost_budget=4.0,  # Between min (2.0) and max (10.0)
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
        )
        problem_config = MinCostProblemConfig(
            cost_model=cost_model,
            performance_target=2.0,  # Modest target
        )
        evaluator = evaluator_config.create_evaluator()
        return problem_config.create_problem(evaluator)

    async def test_max_performance_respects_budget(
        self, small_problem_max_perf: CostConstrainedProblem
    ) -> None:
        """MAX_PERFORMANCE mode should respect cost budget."""
        search_config = SwapBeamConfig(
            beam_width=5, max_allocations=20, max_iterations_per_allocation=10
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
        search_config = SwapBeamConfig(
            beam_width=5, max_allocations=20, max_iterations_per_allocation=10
        )
        search = search_config.create_search_strategy()

        result = await search.search(small_problem_min_cost)

        # Performance should meet target
        assert result.best_score >= 2.0
        # Should have done some evaluations
        assert result.total_evaluations > 0

    async def test_max_performance_starts_cheap(
        self, small_problem_max_perf: CostConstrainedProblem
    ) -> None:
        """MAX_PERFORMANCE should respect budget (start at budget, explore cheaper)."""
        search_config = SwapBeamConfig(
            beam_width=3, max_allocations=10, max_iterations_per_allocation=5
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
        search_config = SwapBeamConfig(
            beam_width=3, max_allocations=10, max_iterations_per_allocation=5
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
        )
        problem_config = MaxPerformanceProblemConfig(
            cost_model=cost_model,
            cost_budget=4.0,
        )
        search_config = SwapBeamConfig(
            beam_width=3, max_allocations=10, max_iterations_per_allocation=5
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

        assert result1.best_score == result2.best_score
        assert result1.final_cost == result2.final_cost
        assert result1.best_parameters.assignments == result2.best_parameters.assignments


def _make_cost_model(costs: list[float]) -> LinearCostModel:
    """Helper to create a cost model from a list of costs."""
    mixer_names = [f"mixer_{i}" for i in range(len(costs))]
    mixer_costs_dict = {name: cost for name, cost in zip(mixer_names, costs)}
    config = LinearCostModelConfig(mixer_costs=mixer_costs_dict)
    return config.create_cost_model()


class TestAllocationBoundStress:
    """Stress tests for allocation_at_cost_bound with various K values.

    These tests verify correctness against exhaustive enumeration.
    """

    @pytest.mark.parametrize("num_mixers", [2, 3, 4, 5, 6, 7, 8])
    def test_from_above_optimality_various_k(self, num_mixers: int) -> None:
        """Test from_above finds true optimum for K=2..8 mixers."""
        # Generate mixer costs with good spread
        costs = [1.0 - i * 0.1 for i in range(num_mixers)]
        model = _make_cost_model(costs)

        num_layers = 12  # Keep small enough for exhaustive check

        # Test various bounds across the cost range
        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)

        for fraction in [0.2, 0.4, 0.5, 0.6, 0.8]:
            bound = min_cost + fraction * (max_cost - min_cost)

            alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
            assert alloc is not None, f"Should find allocation for K={num_mixers}, bound={bound}"

            alloc_cost = model.allocation_cost(alloc)
            assert alloc_cost <= bound + 1e-9, f"Cost {alloc_cost} exceeds bound {bound}"
            assert sum(alloc) == num_layers, f"Allocation sum {sum(alloc)} != {num_layers}"

            # Verify optimality via exhaustive enumeration
            best_cost_found = -float("inf")
            for cost, other_alloc in model.allocations_by_cost(num_layers, "decreasing"):
                if cost <= bound + 1e-9:
                    best_cost_found = cost
                    break

            assert abs(alloc_cost - best_cost_found) < 1e-9, (
                f"K={num_mixers}, bound={bound}: "
                f"Algorithm found cost {alloc_cost}, but optimal is {best_cost_found}"
            )

    @pytest.mark.parametrize("num_mixers", [2, 3, 4, 5, 6, 7, 8])
    def test_from_below_optimality_various_k(self, num_mixers: int) -> None:
        """Test from_below finds true optimum for K=2..8 mixers."""
        costs = [1.0 - i * 0.1 for i in range(num_mixers)]
        model = _make_cost_model(costs)

        num_layers = 12

        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)

        for fraction in [0.2, 0.4, 0.5, 0.6, 0.8]:
            bound = min_cost + fraction * (max_cost - min_cost)

            alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=False)
            assert alloc is not None, f"Should find allocation for K={num_mixers}, bound={bound}"

            alloc_cost = model.allocation_cost(alloc)
            assert alloc_cost >= bound - 1e-9, f"Cost {alloc_cost} below bound {bound}"
            assert sum(alloc) == num_layers, f"Allocation sum {sum(alloc)} != {num_layers}"

            # Verify optimality via exhaustive enumeration
            best_cost_found = float("inf")
            for cost, other_alloc in model.allocations_by_cost(num_layers, "increasing"):
                if cost >= bound - 1e-9:
                    best_cost_found = cost
                    break

            assert abs(alloc_cost - best_cost_found) < 1e-9, (
                f"K={num_mixers}, bound={bound}: "
                f"Algorithm found cost {alloc_cost}, but optimal is {best_cost_found}"
            )

    @pytest.mark.parametrize("num_layers", [5, 10, 20, 50])
    def test_varying_num_layers(self, num_layers: int) -> None:
        """Test with varying number of layers."""
        model = _make_cost_model([1.0, 0.6, 0.3, 0.1])

        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)
        bound = (min_cost + max_cost) / 2

        # from_above
        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        assert model.allocation_cost(alloc) <= bound + 1e-9
        assert sum(alloc) == num_layers

        # from_below
        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=False)
        assert alloc is not None
        assert model.allocation_cost(alloc) >= bound - 1e-9
        assert sum(alloc) == num_layers

    def test_edge_case_equal_costs(self) -> None:
        """Test with some mixers having equal costs."""
        model = _make_cost_model([1.0, 0.5, 0.5, 0.2])  # Two mixers with same cost

        num_layers = 10
        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)
        bound = (min_cost + max_cost) / 2

        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        assert model.allocation_cost(alloc) <= bound + 1e-9
        assert sum(alloc) == num_layers

    def test_edge_case_exact_bound_hit(self) -> None:
        """Test when exact bound can be achieved."""
        model = _make_cost_model([1.0, 0.5, 0.2])

        num_layers = 10
        # Cost of (0, 10, 0) = 5.0 exactly
        bound = 5.0

        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        assert abs(model.allocation_cost(alloc) - 5.0) < 1e-9

    def test_edge_case_at_minimum(self) -> None:
        """Test bound at minimum cost."""
        model = _make_cost_model([1.0, 0.5, 0.2])

        num_layers = 10
        bound = model.min_cost(num_layers)

        # from_above at min should return min allocation
        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        assert model.allocation_cost(alloc) == bound

        # from_below at min should also return min allocation
        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=False)
        assert alloc is not None
        assert model.allocation_cost(alloc) == bound

    def test_edge_case_at_maximum(self) -> None:
        """Test bound at maximum cost."""
        model = _make_cost_model([1.0, 0.5, 0.2])

        num_layers = 10
        bound = model.max_cost(num_layers)

        # from_above at max should return max allocation
        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        assert model.allocation_cost(alloc) == bound

        # from_below at max should also return max allocation
        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=False)
        assert alloc is not None
        assert model.allocation_cost(alloc) == bound

    def test_edge_case_below_minimum(self) -> None:
        """Test bound below minimum cost (infeasible)."""
        model = _make_cost_model([1.0, 0.5, 0.2])

        num_layers = 10
        bound = model.min_cost(num_layers) - 0.1

        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is None

    def test_edge_case_above_maximum(self) -> None:
        """Test bound above maximum cost (infeasible for from_below)."""
        model = _make_cost_model([1.0, 0.5, 0.2])

        num_layers = 10
        bound = model.max_cost(num_layers) + 0.1

        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=False)
        assert alloc is None

    def test_irrational_like_costs(self) -> None:
        """Test with costs that might cause floating point issues."""
        import math
        model = _make_cost_model([1.0, 1.0/3, math.pi/10, 0.1])

        num_layers = 15
        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)

        for fraction in [0.3, 0.5, 0.7]:
            bound = min_cost + fraction * (max_cost - min_cost)

            alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
            assert alloc is not None
            assert model.allocation_cost(alloc) <= bound + 1e-9
            assert sum(alloc) == num_layers

    def test_large_k_small_n(self) -> None:
        """Test with many mixers but few layers."""
        costs = [1.0 - i * 0.08 for i in range(10)]  # 10 mixers
        model = _make_cost_model(costs)

        num_layers = 5  # Fewer layers than mixers

        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)
        bound = (min_cost + max_cost) / 2

        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        assert model.allocation_cost(alloc) <= bound + 1e-9
        assert sum(alloc) == num_layers

    @pytest.mark.parametrize("seed", range(10))
    def test_random_costs(self, seed: int) -> None:
        """Test with random mixer costs."""
        import random
        random.seed(seed)

        num_mixers = random.randint(2, 6)
        costs = sorted([random.uniform(0.1, 1.0) for _ in range(num_mixers)], reverse=True)
        model = _make_cost_model(costs)

        num_layers = random.randint(5, 20)
        min_cost = model.min_cost(num_layers)
        max_cost = model.max_cost(num_layers)
        bound = min_cost + random.uniform(0.1, 0.9) * (max_cost - min_cost)

        # from_above
        alloc = model.allocation_at_cost_bound(num_layers, bound, from_above=True)
        assert alloc is not None
        alloc_cost = model.allocation_cost(alloc)
        assert alloc_cost <= bound + 1e-9
        assert sum(alloc) == num_layers

        # Verify optimality
        for cost, _ in model.allocations_by_cost(num_layers, "decreasing"):
            if cost <= bound + 1e-9:
                assert abs(alloc_cost - cost) < 1e-9, (
                    f"Seed {seed}: found {alloc_cost}, optimal {cost}"
                )
                break
