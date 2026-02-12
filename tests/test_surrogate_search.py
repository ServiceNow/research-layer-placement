"""Integration tests for SurrogateSearch strategy."""

import random
from pathlib import Path

import numpy as np
import pytest

from place_layers.command.explore_placements import (
    ExplorationDataset,
    _enumerate_feasible_allocations,
    _sample_batch,
)
from place_layers.core.types import LayerPlacement
from place_layers.cost_model.linear_cost_model import LinearCostModelConfig
from place_layers.evaluator.mock_supernet_evaluator import (
    MockSupernetEvaluator,
    MockSupernetEvaluatorConfig,
)
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
    MinCostProblemConfig,
)
from place_layers.search_strategy.cost_constrained.types import (
    CostConstrainedSearchResult,
)
from place_layers.search_strategy.cost_constrained.surrogate import (
    SurrogateSearchConfig,
)
from place_layers.surrogate.model_selection import ridge_cv_rho


pytestmark = pytest.mark.asyncio(loop_scope="function")


# --- All SurrogateSearchConfig fields ---

_SURROGATE_CONFIG_DEFAULTS = dict(
    strategy_type="surrogate",
    exploration_dataset_path="PLACEHOLDER",
    cv_folds=3,
    ei_xi=0.01,
    num_rounds=2,
    evals_per_round=20,
    sa=dict(
        num_restarts=3,
        steps_per_restart=200,
        initial_temperature=0.1,
        cooling_rate=0.999,
        min_temperature=1e-6,
        mutation_prob=0.3,
        max_ruin_fraction=0.5,
        contiguous_ruin_prob=0.3,
        top_k=50,
        no_op_retries=0,
        reheat=None,
        seed=42,
    ),
)


# =============================================================================
# Fixtures
# =============================================================================


def _make_small_evaluator(
    num_layers: int = 12, mixer_names: list[str] | None = None
) -> MockSupernetEvaluator:
    if mixer_names is None:
        mixer_names = ["m0", "m1"]
    config = MockSupernetEvaluatorConfig(
        evaluator_type="mock_supernet",
        num_layers=num_layers,
        mixer_names=mixer_names,
        seed=42,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        noise_sigma=0.0,
        noise_seed_shift=0,
        eval_delay_seconds=0.0,
    )
    return MockSupernetEvaluator(config=config)


def _make_max_perf_problem(
    evaluator: MockSupernetEvaluator,
) -> CostConstrainedProblem:
    num_layers = evaluator.num_layers
    mixer_names = sorted(evaluator.mixer_names)
    mixer_costs = {name: 1.0 - 0.5 * i for i, name in enumerate(mixer_names)}
    config = MaxPerformanceProblemConfig(
        problem_type="cost_max_performance",
        cost_model=LinearCostModelConfig(
            cost_model_type="linear",
            mixer_costs=mixer_costs,
        ),
        cost_budget=num_layers * 0.8,
    )
    return CostConstrainedProblem(evaluator=evaluator, config=config)


def _make_min_cost_problem(
    evaluator: MockSupernetEvaluator,
) -> CostConstrainedProblem:
    mixer_names = sorted(evaluator.mixer_names)
    mixer_costs = {name: 1.0 - 0.5 * i for i, name in enumerate(mixer_names)}
    config = MinCostProblemConfig(
        problem_type="cost_min_cost",
        cost_model=LinearCostModelConfig(
            cost_model_type="linear",
            mixer_costs=mixer_costs,
        ),
        performance_target=0.0,  # Easy target so test can find it
    )
    return CostConstrainedProblem(evaluator=evaluator, config=config)


async def _build_exploration_dataset(
    evaluator: MockSupernetEvaluator,
    problem: CostConstrainedProblem,
    num_samples: int = 100,
    cost_budget: float | None = None,
) -> ExplorationDataset:
    """Build a small exploration dataset programmatically."""
    cost_model = problem.cost_model
    num_layers = evaluator.num_layers

    if cost_budget is None:
        if isinstance(problem.config, MaxPerformanceProblemConfig):
            cost_budget = problem.config.cost_budget

    feasible = _enumerate_feasible_allocations(
        cost_model, num_layers, cost_budget
    )

    rng = random.Random(42)
    batch = _sample_batch(num_samples, feasible, num_layers, rng)

    assignments: list[tuple[str, ...]] = []
    scores: list[float] = []
    allocations: list[dict[str, int]] = []

    for assignment in batch:
        placement = LayerPlacement(assignments=tuple(assignment))
        score = await evaluator.evaluate(placement)
        assignments.append(tuple(assignment))
        scores.append(score)
        alloc: dict[str, int] = {}
        for m in assignment:
            alloc[m] = alloc.get(m, 0) + 1
        allocations.append(alloc)

    rho = ridge_cv_rho(
        assignments=assignments,
        scores=np.array(scores),
        mixer_names=cost_model.phase_order,
        cv_folds=5,
        seed=42,
    )

    return ExplorationDataset(
        assignments=assignments,
        scores=scores,
        allocations=allocations,
        num_layers=num_layers,
        mixer_names=cost_model.phase_order,
        num_samples=num_samples,
        termination_reason="max_samples",
        rho_history=[rho],
        samples_at_check=[num_samples],
        final_rho=rho,
    )


def _save_exploration_dataset(
    dataset: ExplorationDataset, tmp_path: Path
) -> Path:
    """Save exploration dataset to a temporary file and return its path."""
    path = tmp_path / "exploration_results.json"
    path.write_text(dataset.model_dump_json(indent=2))
    return path


# =============================================================================
# Integration tests
# =============================================================================


class TestSurrogateSearchIntegration:
    """Integration tests: run surrogate search through search() API."""

    async def test_max_performance(self, tmp_path: Path) -> None:
        """Run max_performance search and verify result."""
        evaluator = _make_small_evaluator(num_layers=12)
        problem = _make_max_perf_problem(evaluator)

        exploration = await _build_exploration_dataset(
            evaluator, problem, num_samples=100
        )
        dataset_path = _save_exploration_dataset(exploration, tmp_path)

        config = SurrogateSearchConfig(
            **{**_SURROGATE_CONFIG_DEFAULTS, "exploration_dataset_path": str(dataset_path)}
        )
        strategy = config.create_search_strategy()
        result = await strategy.search(problem)

        assert isinstance(result, CostConstrainedSearchResult)
        assert result.mode == "max_performance"
        assert result.best_score is not None
        assert result.total_evaluations > 0
        # With exploration dataset, total evals should be only from refinement
        # (2 rounds * 20 evals_per_round = 40)
        assert result.total_evaluations == 2 * 20
        # Cost should be within budget
        cost_budget = problem.config.cost_budget
        assert result.final_cost <= cost_budget + 1e-9

    async def test_min_cost(self, tmp_path: Path) -> None:
        """Run min_cost search and verify result."""
        evaluator = _make_small_evaluator(num_layers=12)
        problem = _make_min_cost_problem(evaluator)

        # For min_cost, build dataset with no cost budget filter
        exploration = await _build_exploration_dataset(
            evaluator, problem, num_samples=100, cost_budget=None
        )
        dataset_path = _save_exploration_dataset(exploration, tmp_path)

        config = SurrogateSearchConfig(
            **{**_SURROGATE_CONFIG_DEFAULTS, "exploration_dataset_path": str(dataset_path)}
        )
        strategy = config.create_search_strategy()
        result = await strategy.search(problem)

        assert isinstance(result, CostConstrainedSearchResult)
        assert result.mode == "min_cost"
        assert result.best_score is not None
        assert result.total_evaluations > 0

    async def test_dry_run(self, tmp_path: Path) -> None:
        """Dry run should return None."""
        evaluator = _make_small_evaluator(num_layers=12)
        problem = _make_max_perf_problem(evaluator)

        exploration = await _build_exploration_dataset(
            evaluator, problem, num_samples=100
        )
        dataset_path = _save_exploration_dataset(exploration, tmp_path)

        sa_overrides = {
            **_SURROGATE_CONFIG_DEFAULTS["sa"],
            "num_restarts": 2,
            "steps_per_restart": 100,
            "top_k": 20,
        }
        config = SurrogateSearchConfig(
            **{
                **_SURROGATE_CONFIG_DEFAULTS,
                "exploration_dataset_path": str(dataset_path),
                "num_rounds": 1,
                "evals_per_round": 10,
                "sa": sa_overrides,
            }
        )
        strategy = config.create_search_strategy()
        result = await strategy.search(problem, dry_run=True)
        assert result is None

    async def test_result_serialization(self) -> None:
        """CostConstrainedSearchResult should serialize and deserialize."""
        result = CostConstrainedSearchResult(
            best_parameters=LayerPlacement(assignments=("m0", "m1", "m0")),
            best_score=0.85,
            total_evaluations=40,
            final_cost=2.0,
            mode="max_performance",
            constraint_value=3.0,
            allocations_explored=0,
        )

        json_str = result.model_dump_json()
        loaded = CostConstrainedSearchResult.model_validate_json(json_str)
        assert loaded.best_score == 0.85
        assert loaded.mode == "max_performance"
        assert loaded.allocations_explored == 0
