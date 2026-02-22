"""Integration tests for ViterbiSurrogateSearch and ViterbiBayesianSurrogateSearch."""

import random
from pathlib import Path

import numpy as np
import pytest

from place_layers.command.explore_placements import _sample_batch
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.surrogate.dataset import ExplorationDataset
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
from place_layers.search_strategy.cost_constrained.risk_bucket_selection import (
    RiskBucketConfig,
    select_by_risk_buckets,
)
from place_layers.search_strategy.cost_constrained.viterbi_surrogate import (
    ViterbiBayesianSurrogateSearchConfig,
    ViterbiSurrogateSearchConfig,
)
from place_layers.surrogate.config import RidgeSurrogateConfig
from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.surrogate.model_selection import ridge_cv_rho
from tests.helpers import wrap_evaluator


pytestmark = pytest.mark.asyncio(loop_scope="function")


# --- All ViterbiSurrogateSearchConfig fields ---

_VITERBI_CONFIG_DEFAULTS = dict(
    strategy_type="viterbi_surrogate",
    exploration_dataset_path="PLACEHOLDER",
    cv_folds=3,
    num_rounds=2,
    evals_per_round=20,
    seed=42,
    candidates_per_model=50,
    surrogate=dict(
        surrogate_type="ridge",
        n_bootstraps=50,
        ei_xi=0.01,
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
        pairwise_decay=0.5,
        pairwise_max_range=1,
        triplet_scale=0.0,
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
        cost_floor=None,
    )
    return CostConstrainedProblem(evaluators=wrap_evaluator(evaluator), config=config, objective_name="weighted_average")


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
        cost_floor=None,
    )
    return CostConstrainedProblem(evaluators=wrap_evaluator(evaluator), config=config, objective_name="weighted_average")


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

    feasible = enumerate_feasible_allocations(
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
        feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
        cv_folds=5,
        seed=42,
    )

    return ExplorationDataset(
        assignments=assignments,
        scores=scores,
        allocations=allocations,
        objective_name="weighted_average",
        num_layers=num_layers,
        mixer_names=cost_model.phase_order,
        num_samples=num_samples,
        termination_reason="max_samples",
        feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
        rho_history=[rho],
        samples_at_check=[num_samples],
        final_rho=rho,
        surrogate_config=RidgeSurrogateConfig(n_bootstraps=50, ei_xi=0.01),
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


class TestViterbiSurrogateSearchIntegration:
    """Integration tests: run viterbi surrogate search through search() API."""

    async def test_max_performance(self, tmp_path: Path) -> None:
        """Run max_performance search and verify result."""
        evaluator = _make_small_evaluator(num_layers=12)
        problem = _make_max_perf_problem(evaluator)

        exploration = await _build_exploration_dataset(
            evaluator, problem, num_samples=100
        )
        dataset_path = _save_exploration_dataset(exploration, tmp_path)

        config = ViterbiSurrogateSearchConfig(
            **{**_VITERBI_CONFIG_DEFAULTS, "exploration_dataset_path": str(dataset_path)}
        )
        strategy = config.create_search_strategy()
        result = await strategy.search(problem)

        assert isinstance(result, CostConstrainedSearchResult)
        assert result.mode == "max_performance"
        assert result.best_score is not None
        assert result.total_evaluations > 0
        # 2 rounds × 20 evals_per_round
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

        config = ViterbiSurrogateSearchConfig(
            **{**_VITERBI_CONFIG_DEFAULTS, "exploration_dataset_path": str(dataset_path)}
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

        config = ViterbiSurrogateSearchConfig(
            **{
                **_VITERBI_CONFIG_DEFAULTS,
                "exploration_dataset_path": str(dataset_path),
                "num_rounds": 1,
                "evals_per_round": 10,
            }
        )
        strategy = config.create_search_strategy()
        result = await strategy.search(problem, dry_run=True)
        assert result is None

    async def test_max_performance_bayesian(self, tmp_path: Path) -> None:
        """Run max_performance search with bayesian_ridge surrogate."""
        evaluator = _make_small_evaluator(num_layers=12)
        problem = _make_max_perf_problem(evaluator)

        exploration = await _build_exploration_dataset(
            evaluator, problem, num_samples=100
        )
        dataset_path = _save_exploration_dataset(exploration, tmp_path)

        config = ViterbiBayesianSurrogateSearchConfig(
            strategy_type="viterbi_bayesian_surrogate",
            exploration_dataset_path=str(dataset_path),
            cv_folds=3,
            num_rounds=2,
            evals_per_round=20,
            seed=42,
            candidates_per_model=50,
            surrogate=dict(
                surrogate_type="bayesian_ridge",
                prior_alpha=1.0,
                prior_a0=1e-6,
                prior_b0=1e-6,
                feature_scaling="none",
            ),
            risk_buckets=[
                RiskBucketConfig(name="safe", sign=-1, betas=[0.0, 1.0], quota=0.6, mean_floor_quantile=0.0),
                RiskBucketConfig(name="upside", sign=1, betas=[1.0], quota=0.3, mean_floor_quantile=0.5),
            ],
        )
        strategy = config.create_search_strategy()
        result = await strategy.search(problem)

        assert isinstance(result, CostConstrainedSearchResult)
        assert result.mode == "max_performance"
        assert result.best_score is not None
        assert result.total_evaluations > 0
        # Cost should be within budget
        cost_budget = problem.config.cost_budget
        assert result.final_cost <= cost_budget + 1e-9


# =============================================================================
# Risk bucket config validation tests
# =============================================================================


class TestRiskBucketValidation:
    """Tests for RiskBucketConfig and ViterbiBayesianSurrogateSearchConfig validation."""

    def _base_config(self, **overrides):
        defaults = dict(
            strategy_type="viterbi_bayesian_surrogate",
            exploration_dataset_path="dummy",
            cv_folds=3,
            num_rounds=1,
            evals_per_round=10,
            seed=42,
            candidates_per_model=50,
            surrogate=dict(
                surrogate_type="bayesian_ridge",
                prior_alpha=1.0,
                prior_a0=1e-6,
                prior_b0=1e-6,
                feature_scaling="none",
            ),
        )
        defaults.update(overrides)
        return defaults

    def test_empty_betas_raises(self):
        """Empty betas list in a bucket raises ValueError."""
        with pytest.raises(ValueError, match="betas must be non-empty"):
            ViterbiBayesianSurrogateSearchConfig(
                **self._base_config(
                    risk_buckets=[
                        RiskBucketConfig(name="bad", sign=-1, betas=[], quota=1.0, mean_floor_quantile=0.0),
                    ]
                )
            )

    def test_negative_beta_raises(self):
        """Negative beta value raises ValueError."""
        with pytest.raises(ValueError, match="betas must be >= 0"):
            ViterbiBayesianSurrogateSearchConfig(
                **self._base_config(
                    risk_buckets=[
                        RiskBucketConfig(name="bad", sign=-1, betas=[-1.0], quota=1.0, mean_floor_quantile=0.0),
                    ]
                )
            )

    def test_empty_buckets_raises(self):
        """No buckets raises ValueError."""
        with pytest.raises(ValueError, match="at least one bucket"):
            ViterbiBayesianSurrogateSearchConfig(
                **self._base_config(risk_buckets=[])
            )

    def test_quota_sum_exceeds_one_raises(self):
        """Quota sum > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="quota sum"):
            ViterbiBayesianSurrogateSearchConfig(
                **self._base_config(
                    risk_buckets=[
                        RiskBucketConfig(name="a", sign=-1, betas=[0.0], quota=0.6, mean_floor_quantile=0.0),
                        RiskBucketConfig(name="b", sign=1, betas=[0.0], quota=0.5, mean_floor_quantile=0.0),
                    ]
                )
            )

    def test_valid_config_accepted(self):
        """Valid config with two buckets is accepted."""
        config = ViterbiBayesianSurrogateSearchConfig(
            **self._base_config(
                risk_buckets=[
                    RiskBucketConfig(name="safe", sign=-1, betas=[0.0, 1.0], quota=0.7, mean_floor_quantile=0.0),
                    RiskBucketConfig(name="upside", sign=1, betas=[1.0], quota=0.3, mean_floor_quantile=0.0),
                ]
            )
        )
        assert len(config.risk_buckets) == 2


# =============================================================================
# Risk bucket selection unit tests
# =============================================================================


class TestRiskBucketSelection:
    """Unit tests for risk-bucket candidate selection logic.

    These tests exercise the selection algorithm by constructing mu/sigma
    arrays directly and checking which indices get selected.
    """

    @staticmethod
    def _select_from_pool(
        mu: np.ndarray,
        sigma: np.ndarray,
        buckets: list[RiskBucketConfig],
        n_select: int,
    ) -> list[int]:
        """Delegate to the extracted select_by_risk_buckets utility."""
        selected, _log = select_by_risk_buckets(mu, sigma, n_select, buckets)
        return selected

    def test_safe_bucket_prefers_low_sigma(self):
        """Safe bucket (sign=-1, beta>0) picks lower-sigma candidates when mu is tied."""
        mu = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        sigma = np.array([0.1, 0.5, 0.2, 0.8, 0.3])

        buckets = [RiskBucketConfig(name="safe", sign=-1, betas=[1.0], quota=1.0, mean_floor_quantile=0.0)]
        selected = self._select_from_pool(mu, sigma, buckets, n_select=3)

        assert len(selected) == 3
        # score = mu + (-1)*1.0*sigma = 1.0 - sigma
        # Highest scores: idx 0 (sigma=0.1), idx 2 (sigma=0.2), idx 4 (sigma=0.3)
        assert selected == [0, 2, 4]

    def test_learn_bucket_picks_high_sigma_above_floor(self):
        """Upside bucket (sign=+1) picks highest-sigma candidates that pass mean_floor_quantile."""
        # 10 candidates, mu = 0.1, 0.2, ..., 1.0
        mu = np.arange(0.1, 1.1, 0.1)
        sigma = np.array([0.5, 0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6, 0.05])

        # mean_floor_quantile=0.8 -> only top 20% of mu values eligible (idx 8,9)
        buckets = [RiskBucketConfig(
            name="upside", sign=1, betas=[1.0], quota=1.0, mean_floor_quantile=0.8,
        )]
        selected = self._select_from_pool(mu, sigma, buckets, n_select=2)

        assert len(selected) == 2
        # Eligible: idx 8 (mu=0.9, sigma=0.6) and idx 9 (mu=1.0, sigma=0.05)
        # score = mu + 1.0*sigma: idx 8 -> 1.5, idx 9 -> 1.05
        assert selected[0] == 8
        assert selected[1] == 9

    def test_small_quotas_do_not_over_allocate(self):
        """Many sub-buckets with small quotas don't select more than k_bucket per bucket."""
        mu = np.ones(20)
        sigma = np.random.RandomState(42).rand(20)

        # Two buckets with many betas, small quotas
        buckets = [
            RiskBucketConfig(name="a", sign=-1, betas=[0.0, 0.5, 1.0, 2.0], quota=0.3, mean_floor_quantile=0.0),
            RiskBucketConfig(name="b", sign=1, betas=[0.0, 0.5, 1.0, 2.0], quota=0.3, mean_floor_quantile=0.0),
        ]
        n_select = 10
        selected = self._select_from_pool(mu, sigma, buckets, n_select=n_select)

        # No duplicates
        assert len(selected) == len(set(selected))
        # Total selected should be exactly n_select (backfill covers remainder)
        assert len(selected) == n_select

    def test_within_bucket_backfill(self):
        """When a beta sub-bucket has zero eligible, bucket still fills via backfill."""
        # 5 candidates, all with same mu
        mu = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Bucket with 3 betas, quota=1.0, n_select=3
        # With 3 betas and k_bucket=3: each beta sub-bucket gets 1 slot
        buckets = [RiskBucketConfig(name="safe", sign=-1, betas=[0.0, 0.5, 1.0], quota=1.0, mean_floor_quantile=0.0)]
        selected = self._select_from_pool(mu, sigma, buckets, n_select=3)
        assert len(selected) == 3
        assert len(set(selected)) == 3  # all unique

        # Test backfill: use mean_floor_quantile=0.9 with a pool where only 2 candidates
        # pass the floor. Bucket wants 4, but each beta sub-bucket can only pick from
        # those 2; once they're taken, backfill (beta=0) finds no more eligible → only 2.
        # Then global backfill fills the remaining 2 from the full pool.
        mu2 = np.array([0.5, 0.5, 0.5, 1.0, 1.0])
        sigma2 = np.array([0.1, 0.2, 0.3, 0.01, 0.02])
        # mean_floor_quantile=0.9 → floor = quantile(mu2, 0.9) = 1.0
        # Eligible: idx 3, 4 only
        buckets2 = [RiskBucketConfig(
            name="safe", sign=-1, betas=[1.0, 2.0], quota=1.0, mean_floor_quantile=0.9,
        )]

        selected2 = self._select_from_pool(mu2, sigma2, buckets2, n_select=4)
        assert len(selected2) == 4
        assert len(set(selected2)) == 4
        # First 2 should be from eligible set (idx 3, 4)
        assert set(selected2[:2]) <= {3, 4}
        # Last 2 filled by global backfill from remaining
        assert set(selected2[2:]) <= {0, 1, 2}

    def test_mean_floor_quantile_keeps_top_fraction(self):
        """mean_floor_quantile=0.8 keeps only candidates with mu >= 80th percentile."""
        mu = np.arange(1.0, 11.0)  # [1, 2, ..., 10]
        sigma = np.ones(10) * 0.1

        # quantile(mu, 0.8) = 8.2 -> eligible: idx 8 (mu=9.0) and idx 9 (mu=10.0)
        buckets = [RiskBucketConfig(
            name="top", sign=1, betas=[0.0], quota=1.0, mean_floor_quantile=0.8,
        )]
        selected = self._select_from_pool(mu, sigma, buckets, n_select=2)

        assert len(selected) == 2
        # Only idx 8 and 9 are eligible (mu >= 8.2)
        assert set(selected) == {8, 9}

    def test_global_backfill_covers_remainder(self):
        """When sum(quotas) < 1.0, global backfill fills remaining slots by mu desc."""
        mu = np.array([3.0, 1.0, 2.0, 4.0, 5.0])
        sigma = np.ones(5) * 0.1

        # quota=0.5 -> bucket gets floor(0.5*4)=2, remainder fills 2 more via global backfill
        buckets = [RiskBucketConfig(name="safe", sign=-1, betas=[0.0], quota=0.5, mean_floor_quantile=0.0)]
        selected = self._select_from_pool(mu, sigma, buckets, n_select=4)

        assert len(selected) == 4
        # Bucket picks top 2 by mu (idx 4=5.0, idx 3=4.0)
        # Global backfill picks next 2 by mu (idx 0=3.0, idx 2=2.0)
        assert selected == [4, 3, 0, 2]
