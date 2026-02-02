"""Tests to verify eval count bounds against actual search runs.

These tests run actual optimizations with the mock evaluator and verify that
actual eval counts fall within predicted bounds from the progress estimator.
"""

import pytest

from place_layers.cost_model import LinearCostModelConfig
from place_layers.evaluator import MockSupernetEvaluatorConfig
from place_layers.problem import MaxPerformanceProblemConfig, MinCostProblemConfig
from place_layers.search_strategy import HierarchicalConfig

pytestmark = pytest.mark.asyncio(loop_scope="function")


def _create_estimator_bounds(
    num_layers: int,
    mixer_costs: dict[str, float],
    cost_budget: float,
    max_allocations: int,
    beam_width: int,
    cross_phase_beam_width: int,
    early_stop_threshold: float = 0.001,
):
    """Create progress estimator and return bounds.

    Helper function that creates evaluator, problem, and search config
    to get bounds from the progress estimator.

    Returns:
        Tuple of (estimator, evaluator) - caller owns evaluator lifecycle.
    """
    mixer_names = list(mixer_costs.keys())

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

    # Create search config and get estimator
    search_config = HierarchicalConfig(
        max_allocations=max_allocations,
        beam_width=beam_width,
        early_stop_threshold=early_stop_threshold,
        cross_phase_beam_width=cross_phase_beam_width,
    )
    estimator = search_config.create_progress_estimator(problem)

    return estimator, evaluator, problem, search_config


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
    early_stop_threshold = 0.0  # Disable early stopping for predictable bounds

    # Get bounds from progress estimator
    estimator, evaluator, problem, search_config = _create_estimator_bounds(
        num_layers=num_layers,
        mixer_costs=mixer_costs,
        cost_budget=cost_budget,
        max_allocations=max_allocations,
        beam_width=beam_width,
        cross_phase_beam_width=cross_phase_beam_width,
        early_stop_threshold=early_stop_threshold,
    )

    assert estimator is not None
    initial = estimator.initial

    # Create search strategy and run optimization
    search = search_config.create_search_strategy()
    async with evaluator:
        result = await search.search(problem)

    # Verify bounds
    actual = result.total_evaluations
    lower = initial.lower
    upper = initial.upper
    expected = initial.expected

    assert lower <= actual <= upper, (
        f"[{description}] Eval count {actual} outside bounds [{lower}, {upper}]\n"
        f"Expected: {expected}, Allocations: {len(estimator.allocation_info)}"
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

    # Get bounds from progress estimator
    estimator, evaluator, problem, search_config = _create_estimator_bounds(
        num_layers=num_layers,
        mixer_costs=mixer_costs,
        cost_budget=cost_budget,
        max_allocations=max_allocations,
        beam_width=beam_width,
        cross_phase_beam_width=cross_phase_beam_width,
        early_stop_threshold=early_stop_threshold,
    )

    assert estimator is not None
    initial = estimator.initial

    # Create search strategy and run optimization
    search = search_config.create_search_strategy()
    async with evaluator:
        result = await search.search(problem)

    # With early stopping, actual should be within statistical bounds
    actual = result.total_evaluations
    lower = initial.lower
    upper = initial.upper

    assert lower <= actual <= upper, (
        f"Eval count {actual} outside bounds [{lower}, {upper}] with early stopping"
    )


class TestProgressEstimatorEdgeCases:
    """Test edge cases for progress estimator bounds."""

    def test_single_allocation(self) -> None:
        """Test with max_allocations=1."""
        estimator, _evaluator, _, _ = _create_estimator_bounds(
            num_layers=10,
            mixer_costs={"m0": 1.0, "m1": 0.5, "m2": 0.2},
            cost_budget=4.0,
            max_allocations=1,
            beam_width=2,
            cross_phase_beam_width=1,
        )

        assert estimator is not None
        assert len(estimator.allocation_info) == 1
        assert estimator.initial.expected > 0
        assert estimator.initial.lower > 0
        assert estimator.initial.upper >= estimator.initial.lower

    def test_two_mixers(self) -> None:
        """Test with only 2 mixers (simplest case)."""
        estimator, _evaluator, _, _ = _create_estimator_bounds(
            num_layers=10,
            mixer_costs={"m0": 1.0, "m1": 0.2},
            cost_budget=5.0,
            max_allocations=5,
            beam_width=3,
            cross_phase_beam_width=1,
        )

        assert estimator is not None
        assert len(estimator.allocation_info) > 0
        assert estimator.initial.expected > 0
        assert estimator.initial.lower > 0
        assert estimator.initial.upper >= estimator.initial.lower

    def test_beam_width_one(self) -> None:
        """Test with beam_width=1 (greedy search)."""
        estimator, _evaluator, _, _ = _create_estimator_bounds(
            num_layers=10,
            mixer_costs={"m0": 1.0, "m1": 0.5, "m2": 0.2},
            cost_budget=4.0,
            max_allocations=3,
            beam_width=1,
            cross_phase_beam_width=1,
        )

        assert estimator is not None
        assert estimator.initial.expected > 0
        assert estimator.initial.lower > 0
        assert estimator.initial.upper >= estimator.initial.lower

    def test_cross_phase_equals_beam_width(self) -> None:
        """Test with cross_phase_beam_width = beam_width."""
        estimator, _evaluator, _, _ = _create_estimator_bounds(
            num_layers=15,
            mixer_costs={"m0": 1.0, "m1": 0.5, "m2": 0.2},
            cost_budget=6.0,
            max_allocations=5,
            beam_width=3,
            cross_phase_beam_width=3,
        )

        assert estimator is not None
        assert estimator.initial.expected > 0
        assert estimator.initial.lower > 0
        assert estimator.initial.upper >= estimator.initial.lower

    def test_allocation_info_structure(self) -> None:
        """Verify allocation_info contains expected information."""
        estimator, _evaluator, _, _ = _create_estimator_bounds(
            num_layers=10,
            mixer_costs={"m0": 1.0, "m1": 0.5, "m2": 0.2},
            cost_budget=4.0,
            max_allocations=3,
            beam_width=2,
            cross_phase_beam_width=1,
        )

        assert estimator is not None
        assert len(estimator.allocation_info) > 0

        for info in estimator.allocation_info:
            assert info.allocation is not None
            assert len(info.allocation) == 3  # 3 mixers
            assert info.budgets is not None
            assert info.expected > 0
            assert info.variance >= 0
            assert info.estimated_lower > 0
            assert info.estimated_upper >= info.estimated_lower

    def test_expected_within_bounds(self) -> None:
        """Verify expected value is within bounds."""
        estimator, _evaluator, _, _ = _create_estimator_bounds(
            num_layers=10,
            mixer_costs={"m0": 1.0, "m1": 0.5, "m2": 0.2},
            cost_budget=4.0,
            max_allocations=3,
            beam_width=2,
            cross_phase_beam_width=1,
        )

        assert estimator is not None
        initial = estimator.initial
        assert initial.expected > 0
        assert initial.std_dev >= 0
        # Expected should be within bounds
        assert initial.lower <= initial.expected <= initial.upper


# -----------------------------------------------------------------------------
# MinCostProblem tests
# -----------------------------------------------------------------------------


async def test_min_cost_creates_progress_estimator() -> None:
    """MinCostProblem should create a progress estimator."""
    num_layers = 10
    mixer_costs = {"m0": 1.0, "m1": 0.5, "m2": 0.2}

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

    # Create MinCostProblem
    cost_model_config = LinearCostModelConfig(mixer_costs=mixer_costs)
    problem_config = MinCostProblemConfig(
        cost_model=cost_model_config,
        performance_target=0.5,
    )
    problem = problem_config.create_problem(evaluator)

    # Create search strategy
    search_config = HierarchicalConfig(
        max_allocations=5,
        beam_width=2,
        early_stop_threshold=0.0,
        cross_phase_beam_width=1,
    )

    # Create progress estimator
    estimator = search_config.create_progress_estimator(problem)

    # Should NOT be None for MinCostProblem
    assert estimator is not None
    assert len(estimator.allocation_info) > 0


# Test scenarios for min_cost as tuples of:
# (num_layers, mixer_costs, performance_target, max_allocations, beam_width, description)
MIN_COST_SCENARIOS = [
    (10, {"m0": 1.0, "m1": 0.5, "m2": 0.2}, 0.3, 5, 2, "low target (explore many)"),
    (10, {"m0": 1.0, "m1": 0.5, "m2": 0.2}, 0.5, 5, 2, "medium target"),
    (10, {"m0": 1.0, "m1": 0.5, "m2": 0.2}, 0.7, 5, 2, "high target (early termination)"),
    (20, {"m0": 1.0, "m1": 0.5, "m2": 0.2}, 0.4, 5, 3, "20 layers"),
]


@pytest.mark.parametrize(
    "num_layers,mixer_costs,performance_target,max_allocations,beam_width,description",
    MIN_COST_SCENARIOS,
    ids=[s[-1] for s in MIN_COST_SCENARIOS],
)
async def test_min_cost_eval_count_within_bounds(
    num_layers: int,
    mixer_costs: dict[str, float],
    performance_target: float,
    max_allocations: int,
    beam_width: int,
    description: str,
) -> None:
    """Verify actual eval count falls within predicted bounds for MinCostProblem.

    The bounds assume all max_allocations will be explored (upper estimate).
    Actual may be lower due to early termination when performance drops.
    """
    mixer_names = list(mixer_costs.keys())
    early_stop_threshold = 0.0

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

    # Create MinCostProblem
    cost_model_config = LinearCostModelConfig(mixer_costs=mixer_costs)
    problem_config = MinCostProblemConfig(
        cost_model=cost_model_config,
        performance_target=performance_target,
    )
    problem = problem_config.create_problem(evaluator)

    # Create search strategy
    search_config = HierarchicalConfig(
        max_allocations=max_allocations,
        beam_width=beam_width,
        early_stop_threshold=early_stop_threshold,
        cross_phase_beam_width=1,
    )

    # Get initial bounds from estimator
    estimator = search_config.create_progress_estimator(problem)
    assert estimator is not None
    initial_bounds = estimator.initial

    # Run optimization
    search = search_config.create_search_strategy()
    async with evaluator:
        result = await search.search(problem)

    # Verify bounds: actual should be <= upper (may be < lower if early termination)
    actual = result.total_evaluations

    # For min_cost, the bounds are computed assuming all allocations run.
    # With early termination, actual may be much lower than lower bound.
    # So we only check actual <= upper.
    assert actual <= initial_bounds.upper, (
        f"[{description}] Eval count {actual} exceeds upper bound {initial_bounds.upper}"
    )


async def test_min_cost_early_termination_bounds() -> None:
    """Verify bounds behavior when min_cost terminates early.

    When performance drops below target, search terminates. The bounds
    tighten correctly based on completed allocations.
    """
    num_layers = 10
    mixer_costs = {"m0": 1.0, "m1": 0.5, "m2": 0.2}
    # Very high target - likely to cause early termination
    performance_target = 0.9

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

    # Create MinCostProblem
    cost_model_config = LinearCostModelConfig(mixer_costs=mixer_costs)
    problem_config = MinCostProblemConfig(
        cost_model=cost_model_config,
        performance_target=performance_target,
    )
    problem = problem_config.create_problem(evaluator)

    # Create search strategy with many allocations to see early termination
    search_config = HierarchicalConfig(
        max_allocations=10,
        beam_width=2,
        early_stop_threshold=0.0,
        cross_phase_beam_width=1,
    )

    # Get initial bounds
    estimator = search_config.create_progress_estimator(problem)
    assert estimator is not None
    initial_upper = estimator.initial.upper

    # Run optimization
    search = search_config.create_search_strategy()
    async with evaluator:
        result = await search.search(problem)

    actual = result.total_evaluations

    # With early termination, actual should be well below the upper bound
    # (which assumed all allocations would run)
    assert actual <= initial_upper, (
        f"Eval count {actual} exceeds upper bound {initial_upper}"
    )
