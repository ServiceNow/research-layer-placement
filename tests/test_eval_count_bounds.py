"""Tests to verify eval count calculator formulas against actual search runs.

These tests run actual optimizations with the mock evaluator and verify that
actual eval counts fall within predicted bounds from calculate_bounds().
"""

import pytest

from place_layers.command.eval_count_calculator import calculate_bounds
from place_layers.cost_model import LinearCostModelConfig
from place_layers.evaluator import MockSupernetEvaluatorConfig
from place_layers.problem import MaxPerformanceProblemConfig
from place_layers.search_strategy import HierarchicalConfig

pytestmark = pytest.mark.asyncio(loop_scope="function")


# Test scenarios as tuples of:
# (num_layers, mixer_costs, cost_budget, max_allocations, beam_width, cross_phase_beam_width, description)
TEST_SCENARIOS = [
    # (num_layers, mixer_costs, cost_budget, max_allocs, beam_width, cross_phase_bw, description)
    (10, {"m0": 1.0, "m1": 0.5, "m2": 0.2}, 4.0, 3, 2, 1, "10 layers, 3 mixers, beam=2"),
    (10, {"m0": 1.0, "m1": 0.5, "m2": 0.2}, 5.0, 5, 3, 1, "10 layers, 3 mixers, beam=3"),
    (20, {"m0": 1.0, "m1": 0.5, "m2": 0.2}, 8.0, 5, 3, 1, "20 layers, 3 mixers, beam=3"),
    (20, {"m0": 1.0, "m1": 0.6, "m2": 0.3, "m3": 0.1}, 6.0, 5, 3, 1, "20 layers, 4 mixers"),
    (20, {"m0": 1.0, "m1": 0.5, "m2": 0.2}, 8.0, 5, 3, 2, "cross_phase_beam_width=2"),
    (48, {"FA": 1.0, "SWA": 0.156, "GDN": 0.09, "KDA": 0.11}, 6.4, 10, 3, 1, "48 layers (realistic)"),
]


@pytest.mark.parametrize(
    "num_layers,mixer_costs,cost_budget,max_allocations,beam_width,cross_phase_beam_width,description",
    TEST_SCENARIOS,
    ids=[s[-1] for s in TEST_SCENARIOS],
)
async def test_eval_count_within_bounds(
    num_layers: int,
    mixer_costs: dict[str, float],
    cost_budget: float,
    max_allocations: int,
    beam_width: int,
    cross_phase_beam_width: int,
    description: str,
) -> None:
    """Verify actual eval count falls within predicted bounds.

    Runs optimization and checks: lower_bound <= total_evaluations <= upper_bound
    """
    mixer_names = list(mixer_costs.keys())
    early_stop_threshold = 0.0  # Disable early stopping for predictable bounds

    # Calculate predicted bounds (with matching early_stop_threshold)
    bounds = calculate_bounds(
        num_layers=num_layers,
        mixer_costs=mixer_costs,
        cost_budget=cost_budget,
        max_allocations=max_allocations,
        beam_width=beam_width,
        cross_phase_beam_width=cross_phase_beam_width,
        early_stop_threshold=early_stop_threshold,
    )

    # Create evaluator
    evaluator_config = MockSupernetEvaluatorConfig(
        num_layers=num_layers,
        mixer_names=mixer_names,
        seed=42,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        noise_sigma=0.2,
    )
    evaluator = evaluator_config.create_evaluator()

    # Create problem
    cost_model_config = LinearCostModelConfig(mixer_costs=mixer_costs)
    problem_config = MaxPerformanceProblemConfig(
        cost_model=cost_model_config,
        cost_budget=cost_budget,
    )
    problem = problem_config.create_problem(evaluator)

    # Create search strategy with matching early_stop_threshold
    search_config = HierarchicalConfig(
        max_allocations=max_allocations,
        beam_width=beam_width,
        early_stop_threshold=early_stop_threshold,
        cross_phase_beam_width=cross_phase_beam_width,
    )
    search = search_config.create_search_strategy()

    # Run optimization
    async with evaluator:
        result = await search.search(problem)

    # Verify bounds
    actual = result.total_evaluations
    lower = bounds["lower_bound"]
    upper = bounds["upper_bound"]
    expected = bounds["expected"]

    assert lower <= actual <= upper, (
        f"[{description}] Eval count {actual} outside bounds [{lower}, {upper}]\n"
        f"Expected: {expected}, Allocations analyzed: {bounds['allocations_analyzed']}"
    )


async def test_bounds_with_early_stopping() -> None:
    """Verify that with early stopping enabled, actual evals fall within statistical bounds.

    The statistical model accounts for early stopping, so actual should be within bounds.
    """
    num_layers = 20
    mixer_costs = {"m0": 1.0, "m1": 0.5, "m2": 0.2}
    cost_budget = 8.0
    max_allocations = 5
    beam_width = 3
    cross_phase_beam_width = 1
    early_stop_threshold = 0.001  # Enable early stopping

    # Calculate bounds (with matching early_stop_threshold)
    bounds = calculate_bounds(
        num_layers=num_layers,
        mixer_costs=mixer_costs,
        cost_budget=cost_budget,
        max_allocations=max_allocations,
        beam_width=beam_width,
        cross_phase_beam_width=cross_phase_beam_width,
        early_stop_threshold=early_stop_threshold,
    )

    # Create evaluator
    evaluator_config = MockSupernetEvaluatorConfig(
        num_layers=num_layers,
        mixer_names=list(mixer_costs.keys()),
        seed=42,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        noise_sigma=0.2,
    )
    evaluator = evaluator_config.create_evaluator()

    # Create problem
    cost_model_config = LinearCostModelConfig(mixer_costs=mixer_costs)
    problem_config = MaxPerformanceProblemConfig(
        cost_model=cost_model_config,
        cost_budget=cost_budget,
    )
    problem = problem_config.create_problem(evaluator)

    # Create search strategy WITH early stopping enabled
    search_config = HierarchicalConfig(
        max_allocations=max_allocations,
        beam_width=beam_width,
        early_stop_threshold=early_stop_threshold,
        cross_phase_beam_width=cross_phase_beam_width,
    )
    search = search_config.create_search_strategy()

    # Run optimization
    async with evaluator:
        result = await search.search(problem)

    # With early stopping, actual should be within statistical bounds
    actual = result.total_evaluations
    lower = bounds["lower_bound"]
    upper = bounds["upper_bound"]

    assert lower <= actual <= upper, (
        f"Eval count {actual} outside bounds [{lower}, {upper}] with early stopping"
    )


class TestCalculateBoundsEdgeCases:
    """Test edge cases for calculate_bounds function."""

    def test_single_allocation(self) -> None:
        """Test with max_allocations=1."""
        bounds = calculate_bounds(
            num_layers=10,
            mixer_costs={"m0": 1.0, "m1": 0.5, "m2": 0.2},
            cost_budget=4.0,
            max_allocations=1,
            beam_width=2,
            cross_phase_beam_width=1,
        )

        assert bounds["allocations_analyzed"] == 1
        assert bounds["expected"] > 0
        assert bounds["lower_bound"] > 0
        assert bounds["upper_bound"] >= bounds["lower_bound"]

    def test_two_mixers(self) -> None:
        """Test with only 2 mixers (simplest case)."""
        bounds = calculate_bounds(
            num_layers=10,
            mixer_costs={"m0": 1.0, "m1": 0.2},
            cost_budget=5.0,
            max_allocations=5,
            beam_width=3,
            cross_phase_beam_width=1,
        )

        assert bounds["allocations_analyzed"] > 0
        assert bounds["expected"] > 0
        assert bounds["lower_bound"] > 0
        assert bounds["upper_bound"] >= bounds["lower_bound"]

    def test_beam_width_one(self) -> None:
        """Test with beam_width=1 (greedy search)."""
        bounds = calculate_bounds(
            num_layers=10,
            mixer_costs={"m0": 1.0, "m1": 0.5, "m2": 0.2},
            cost_budget=4.0,
            max_allocations=3,
            beam_width=1,
            cross_phase_beam_width=1,
        )

        assert bounds["expected"] > 0
        assert bounds["lower_bound"] > 0
        assert bounds["upper_bound"] >= bounds["lower_bound"]

    def test_cross_phase_equals_beam_width(self) -> None:
        """Test with cross_phase_beam_width = beam_width."""
        bounds = calculate_bounds(
            num_layers=15,
            mixer_costs={"m0": 1.0, "m1": 0.5, "m2": 0.2},
            cost_budget=6.0,
            max_allocations=5,
            beam_width=3,
            cross_phase_beam_width=3,
        )

        assert bounds["expected"] > 0
        assert bounds["lower_bound"] > 0
        assert bounds["upper_bound"] >= bounds["lower_bound"]

    def test_allocation_details_returned(self) -> None:
        """Verify allocation_details contains expected information."""
        bounds = calculate_bounds(
            num_layers=10,
            mixer_costs={"m0": 1.0, "m1": 0.5, "m2": 0.2},
            cost_budget=4.0,
            max_allocations=3,
            beam_width=2,
            cross_phase_beam_width=1,
        )

        assert "allocation_details" in bounds
        assert len(bounds["allocation_details"]) == bounds["allocations_analyzed"]

        for detail in bounds["allocation_details"]:
            assert "allocation" in detail
            assert "allocation_dict" in detail
            assert "cost" in detail
            assert "budgets" in detail
            assert "expected" in detail
            assert "lower_bound" in detail
            assert "upper_bound" in detail
            assert detail["cost"] <= 4.0  # Within budget

    def test_expected_in_results(self) -> None:
        """Verify expected value and std_dev are returned."""
        bounds = calculate_bounds(
            num_layers=10,
            mixer_costs={"m0": 1.0, "m1": 0.5, "m2": 0.2},
            cost_budget=4.0,
            max_allocations=3,
            beam_width=2,
            cross_phase_beam_width=1,
        )

        assert "expected" in bounds
        assert "std_dev" in bounds
        assert bounds["expected"] > 0
        assert bounds["std_dev"] >= 0
        # Expected should be within bounds
        assert bounds["lower_bound"] <= bounds["expected"] <= bounds["upper_bound"]
