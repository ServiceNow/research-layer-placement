"""Integration tests for explore_placements command."""

import random
from pathlib import Path

import numpy as np
import pytest

from place_layers.command.explore_placements import (
    _append_to_jsonl,
    _check_plateau,
    _compute_cost_uniform_weights,
    _get_allocation,
    _load_progress,
    _sample_batch,
)
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.surrogate.config import BayesianRidgeSurrogateConfig, RidgeSurrogateConfig
from place_layers.surrogate.dataset import ExplorationDataset
from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.core.types import LayerPlacement
from place_layers.cost_model.linear_cost_model import LinearCostModelConfig
from place_layers.evaluator.mock_supernet_evaluator import (
    MockSupernetEvaluator,
    MockSupernetEvaluatorConfig,
)
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
)
from place_layers.surrogate.model_selection import ridge_cv_rho
from tests.helpers import wrap_evaluator


pytestmark = pytest.mark.asyncio(loop_scope="function")


# =============================================================================
# Fixtures
# =============================================================================


def _make_small_evaluator(
    num_layers: int = 12, mixer_names: list[str] | None = None
) -> MockSupernetEvaluator:
    """Create a small mock evaluator for testing."""
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
        pairwise_decay=0.5,
        pairwise_max_range=1,
        triplet_scale=0.0,
    )
    return MockSupernetEvaluator(config=config)


def _make_small_problem(
    evaluator: MockSupernetEvaluator,
) -> CostConstrainedProblem:
    """Create a small cost-constrained problem."""
    num_layers = evaluator.num_layers
    mixer_names = sorted(evaluator.mixer_names)
    # Cost: m0=1.0, m1=0.5
    mixer_costs = {name: 1.0 - 0.5 * i for i, name in enumerate(mixer_names)}
    config = MaxPerformanceProblemConfig(
        problem_type="cost_max_performance",
        cost_model=LinearCostModelConfig(
            cost_model_type="linear",
            mixer_costs=mixer_costs,
        ),
        cost_budget=num_layers * 0.8,  # Budget between min and max
        cost_floor=None,
    )
    return CostConstrainedProblem(evaluators=wrap_evaluator(evaluator), config=config, objective_name="weighted_average")


# =============================================================================
# Helper function tests
# =============================================================================


class TestHelpers:
    def test_get_allocation(self) -> None:
        assignment = ("m0", "m1", "m0", "m1", "m0")
        alloc = _get_allocation(assignment)
        assert alloc == {"m0": 3, "m1": 2}

    def testenumerate_feasible_allocations(self) -> None:
        """Should enumerate allocations within budget."""
        config = LinearCostModelConfig(
            cost_model_type="linear",
            mixer_costs={"m0": 1.0, "m1": 0.5},
        )
        cost_model = config.create_cost_model()
        num_layers = 6

        # All allocations (no budget)
        all_allocs = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=None
        )
        assert len(all_allocs) == num_layers + 1  # 0..6 for m0

        # With budget
        budget = 4.0  # Excludes allocations costing more than 4.0
        feasible = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=budget
        )
        for alloc in feasible:
            cost = cost_model.allocation_cost(alloc)
            assert cost <= budget + 1e-9

    def test_sample_batch(self) -> None:
        """Should produce correct number of samples with correct lengths."""
        allocs = [{"m0": 3, "m1": 2}, {"m0": 1, "m1": 4}]
        rng = random.Random(42)
        batch = _sample_batch(10, allocs, num_layers=5, rng=rng)
        assert len(batch) == 10
        for assignment in batch:
            assert len(assignment) == 5

    def test_sample_batch_with_weights(self) -> None:
        """Weights should bias allocation selection towards higher-weighted allocations."""
        allocs = [{"m0": 5, "m1": 0}, {"m0": 0, "m1": 5}]
        # Heavily bias towards second allocation
        weights = [0.01, 0.99]
        rng = random.Random(42)
        batch = _sample_batch(200, allocs, num_layers=5, rng=rng, weights=weights)

        # Count how many samples came from each allocation
        second_alloc_count = sum(
            1 for a in batch if a.count("m1") == 5
        )
        # With 99% weight on second allocation, vast majority should be from it
        assert second_alloc_count > 150, (
            f"Expected >150 samples from second allocation, got {second_alloc_count}"
        )

    def test_cost_uniform_weights_basic(self) -> None:
        """Each cost level should receive equal total weight."""
        config = LinearCostModelConfig(
            cost_model_type="linear",
            mixer_costs={"m0": 1.0, "m1": 0.5},
        )
        cost_model = config.create_cost_model()
        num_layers = 4

        # Enumerate all allocations (no budget)
        allocs = enumerate_feasible_allocations(cost_model, num_layers, cost_budget=None)
        weights = _compute_cost_uniform_weights(allocs, cost_model)

        assert len(weights) == len(allocs)

        # Group total weight by cost level
        cost_totals: dict[float, float] = {}
        for alloc, w in zip(allocs, weights):
            cost = round(cost_model.allocation_cost(alloc), 10)
            cost_totals[cost] = cost_totals.get(cost, 0.0) + w

        # Each cost level should have equal total weight (they're all 1.0
        # since each cost level has exactly 1 allocation for linear costs)
        totals = list(cost_totals.values())
        for t in totals:
            assert abs(t - totals[0]) < 1e-9, (
                f"Cost levels have unequal total weights: {cost_totals}"
            )

    def test_check_plateau_no_plateau(self) -> None:
        history = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert not _check_plateau(history, patience=3, min_improvement=0.02)

    def test_check_plateau_detected(self) -> None:
        history = [0.5, 0.501, 0.502, 0.503]
        assert _check_plateau(history, patience=3, min_improvement=0.02)

    def test_check_plateau_too_short(self) -> None:
        history = [0.5, 0.501]
        assert not _check_plateau(history, patience=3, min_improvement=0.02)

    def test_exploration_dataset_load(self, tmp_path: Path) -> None:
        """ExplorationDataset.load() should deserialize from file."""
        dataset = ExplorationDataset(
            assignments=[("m0", "m1")],
            scores=[0.5],
            allocations=[{"m0": 1, "m1": 1}],
            objective_name="weighted_average",
            num_layers=2,
            mixer_names=["m0", "m1"],
            num_samples=1,
            termination_reason="max_samples",
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            surrogate_config=RidgeSurrogateConfig(
                n_bootstraps=50, ei_xi=0.01,
            ),
            rho_history=[0.3],
            samples_at_check=[1],
            final_rho=0.3,
        )
        path = tmp_path / "exploration_results.json"
        path.write_text(dataset.model_dump_json(indent=2))

        loaded = ExplorationDataset.load(path)
        assert loaded.num_samples == 1
        assert loaded.assignments == [("m0", "m1")]
        assert loaded.final_rho == 0.3
        assert loaded.objective_name == "weighted_average"
        assert loaded.surrogate_config is not None
        assert loaded.surrogate_config.surrogate_type == "ridge"

    def test_jsonl_roundtrip(self, tmp_path: Path) -> None:
        """Write and read progress JSONL."""
        path = tmp_path / "progress.jsonl"
        _append_to_jsonl(path, ("m0", "m1", "m0"), 0.75, {"m0": 2, "m1": 1})
        _append_to_jsonl(path, ("m1", "m1", "m0"), 0.82, {"m0": 1, "m1": 2})

        assignments, scores, allocations = _load_progress(path)
        assert len(assignments) == 2
        assert assignments[0] == ("m0", "m1", "m0")
        assert scores[1] == 0.82
        assert allocations[0] == {"m0": 2, "m1": 1}


# =============================================================================
# Integration test
# =============================================================================


class TestExploreIntegration:
    """Integration test: run exploration loop on small problem."""

    async def test_exploration_loop(self) -> None:
        """Run adaptive exploration on a 12-layer, 2-mixer problem."""
        evaluator = _make_small_evaluator(num_layers=12)
        problem = _make_small_problem(evaluator)
        cost_model = problem.cost_model
        num_layers = evaluator.num_layers

        # Enumerate feasible allocations
        assert isinstance(problem.config, MaxPerformanceProblemConfig)
        cost_budget = problem.config.cost_budget
        feasible = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget
        )
        assert len(feasible) > 0

        # Sample and evaluate
        rng = random.Random(42)
        min_samples = 100
        batch_size = 50

        all_assignments: list[tuple[str, ...]] = []
        all_scores: list[float] = []

        # Initial batch
        batch = _sample_batch(min_samples, feasible, num_layers, rng)
        for assignment in batch:
            placement = LayerPlacement(assignments=tuple(assignment))
            score = await evaluator.evaluate(placement)
            all_assignments.append(tuple(assignment))
            all_scores.append(score)

        assert len(all_assignments) == min_samples

        # Quality check via Ridge CV
        rho1 = ridge_cv_rho(
            assignments=all_assignments,
            scores=np.array(all_scores),
            mixer_names=cost_model.phase_order,
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            cv_folds=5,
            seed=42,
        )
        rho_history = [rho1]

        # One more batch
        batch = _sample_batch(batch_size, feasible, num_layers, rng)
        for assignment in batch:
            placement = LayerPlacement(assignments=tuple(assignment))
            score = await evaluator.evaluate(placement)
            all_assignments.append(tuple(assignment))
            all_scores.append(score)

        rho2 = ridge_cv_rho(
            assignments=all_assignments,
            scores=np.array(all_scores),
            mixer_names=cost_model.phase_order,
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            cv_folds=5,
            seed=42,
        )
        rho_history.append(rho2)

        # Assertions
        assert len(all_assignments) == min_samples + batch_size
        assert rho_history[0] > -1.0  # Sanity: rho is computed
        assert len(rho_history) == 2

    async def test_exploration_dataset_serialization(self) -> None:
        """ExplorationDataset should serialize and deserialize correctly."""
        dataset = ExplorationDataset(
            assignments=[("m0", "m1"), ("m1", "m0")],
            scores=[0.5, 0.7],
            allocations=[{"m0": 1, "m1": 1}, {"m0": 1, "m1": 1}],
            objective_name="weighted_average",
            num_layers=2,
            mixer_names=["m0", "m1"],
            num_samples=2,
            termination_reason="max_samples",
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            surrogate_config=RidgeSurrogateConfig(
                n_bootstraps=50, ei_xi=0.01,
            ),
            rho_history=[0.3],
            samples_at_check=[2],
            final_rho=0.3,
        )

        # Serialize to JSON and back
        json_str = dataset.model_dump_json()
        loaded = ExplorationDataset.model_validate_json(json_str)

        assert loaded.num_samples == 2
        assert loaded.assignments == [("m0", "m1"), ("m1", "m0")]
        assert loaded.scores == [0.5, 0.7]
        assert loaded.objective_name == "weighted_average"
        assert loaded.termination_reason == "max_samples"
        assert loaded.surrogate_config is not None
        assert loaded.surrogate_config.surrogate_type == "ridge"

    async def test_exploration_dataset_bayesian_surrogate(self) -> None:
        """ExplorationDataset with BayesianRidgeSurrogateConfig round-trips."""
        cfg = BayesianRidgeSurrogateConfig(
            prior_alpha=1.0,
            prior_a0=1e-6,
            prior_b0=1e-6,
            lcb_kappa=2.0,
            feature_scaling="counts_only",
        )
        dataset = ExplorationDataset(
            assignments=[("m0", "m1")],
            scores=[0.5],
            allocations=[{"m0": 1, "m1": 1}],
            objective_name="weighted_average",
            num_layers=2,
            mixer_names=["m0", "m1"],
            num_samples=1,
            termination_reason="max_samples",
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            surrogate_config=cfg,
            rho_history=[0.3],
            samples_at_check=[1],
            final_rho=0.3,
        )

        json_str = dataset.model_dump_json()
        loaded = ExplorationDataset.model_validate_json(json_str)
        assert loaded.surrogate_config is not None
        assert loaded.surrogate_config.surrogate_type == "bayesian_ridge"
        assert isinstance(loaded.surrogate_config, BayesianRidgeSurrogateConfig)
        assert loaded.surrogate_config.prior_alpha == 1.0
