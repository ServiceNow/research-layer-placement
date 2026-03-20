"""Integration tests for the validate_landscape pipeline.

Tests the full pipeline:
1. Build exploration dataset (programmatic)
2. Run validate_landscape per-band search
3. Verify ValidationResults structure and serialization
4. Verify placement_landscape can load validation results
5. Verify incremental per-band saving and resume-on-restart
6. Verify multi-objective support
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from place_layers.command.explore_placements import _sample_batch
from place_layers.landscape import (
    BandResult,
    BandValidationResult,
    LandscapeConfig,
    LandscapeData,
    PlacementResult,
    ValidationResults,
    ValidationSummary,
    compute_band_edges,
    compute_bin_assignments,
    compute_dp_per_band,
    objective_result_path,
)
from place_layers.surrogate.evaluation_log import PooledEvaluationData
from place_layers.core.types import LayerPlacement
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.cost_model.linear_cost_model import LinearCostModelConfig
from place_layers.evaluator.evaluators import EvaluatorsConfig
from place_layers.evaluator.mock_supernet_evaluator import (
    MockSupernetEvaluator,
    MockSupernetEvaluatorConfig,
)
from place_layers.evaluator.objective import (
    IndependentObjectiveConfig,
    MaxObjectiveConfig,
    WeightedAverageObjectiveConfig,
)
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
)
from place_layers.search_strategy.cost_constrained.risk_bucket_selection import RiskBucketConfig
from place_layers.search_strategy.cost_constrained.viterbi_surrogate import (
    ViterbiBayesianSurrogateSearchConfig,
    ViterbiSurrogateSearchConfig,
)
from place_layers.surrogate.config import RidgeSurrogateConfig
from place_layers.surrogate.dataset import ExplorationDataset
from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.surrogate.model_selection import ridge_cv_rho
from tests.helpers import wrap_evaluator


pytestmark = pytest.mark.asyncio(loop_scope="function")


# =============================================================================
# Fixtures
# =============================================================================


_COST_MODEL_CONFIG = LinearCostModelConfig(
    cost_model_type="linear",
    mixer_costs={"m0": 1.0, "m1": 0.5},
)

_NUM_LAYERS = 12


def _make_evaluator(noise_sigma: float = 0.0) -> MockSupernetEvaluator:
    config = MockSupernetEvaluatorConfig(
        evaluator_type="mock_supernet",
        num_layers=_NUM_LAYERS,
        mixer_names=["m0", "m1"],
        seed=42,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        noise_sigma=noise_sigma,
        noise_seed_shift=0,
        eval_delay_seconds=0.0,
        pairwise_decay=0.5,
        pairwise_max_range=1,
        triplet_scale=0.0,
    )
    return MockSupernetEvaluator(config=config)


def _make_evaluator_config(
    noise_sigma: float = 0.0,
    noise_seed_shift: int = 0,
) -> MockSupernetEvaluatorConfig:
    return MockSupernetEvaluatorConfig(
        evaluator_type="mock_supernet",
        num_layers=_NUM_LAYERS,
        mixer_names=["m0", "m1"],
        seed=42,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        noise_sigma=noise_sigma,
        noise_seed_shift=noise_seed_shift,
        eval_delay_seconds=0.0,
        pairwise_decay=0.5,
        pairwise_max_range=1,
        triplet_scale=0.0,
    )


async def _build_exploration_dataset(
    evaluator: MockSupernetEvaluator,
    num_samples: int = 100,
    objective_name: str = "weighted_average",
) -> ExplorationDataset:
    """Build a small exploration dataset programmatically."""
    cost_model = _COST_MODEL_CONFIG.create_cost_model()
    num_layers = evaluator.num_layers
    cost_budget = num_layers * 0.8  # Same budget as problem config

    feasible = enumerate_feasible_allocations(
        cost_model, num_layers, cost_budget
    )

    rng = random.Random(42)
    batch = _sample_batch(num_samples, feasible, num_layers, rng)

    assignments: list[tuple[str, ...]] = []
    scores: list[float] = []

    for assignment in batch:
        placement = LayerPlacement(assignments=tuple(assignment))
        score = await evaluator.evaluate(placement)
        assignments.append(tuple(assignment))
        scores.append(score)

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
        objective_name=objective_name,
        num_layers=num_layers,
        mixer_names=cost_model.phase_order,
        num_samples=num_samples,
        termination_reason="max_samples",
        feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
        quality_history=[rho],
        samples_at_check=[num_samples],
        final_quality=rho,
    )


def _save_dataset(dataset: ExplorationDataset, tmp_path: Path) -> Path:
    path = tmp_path / "exploration_results.json"
    path.write_text(dataset.model_dump_json(indent=2))
    return path


def _pool(dataset: ExplorationDataset) -> PooledEvaluationData:
    """Convert an ExplorationDataset to PooledEvaluationData (no supplemental)."""
    return PooledEvaluationData(
        assignments=list(dataset.assignments),
        scores=list(dataset.scores),
        num_layers=dataset.num_layers,
        mixer_names=dataset.mixer_names,
    )


# =============================================================================
# Tests: shared helpers
# =============================================================================


class TestLandscapeHelpers:
    """Test shared helper functions from landscape_helpers."""

    async def test_collect_and_fit_surrogate(self, tmp_path: Path) -> None:
        """Pool dataset + create/fit surrogate returns fitted surrogate."""
        evaluator = _make_evaluator()
        dataset = await _build_exploration_dataset(evaluator, num_samples=100)
        _save_dataset(dataset, tmp_path)

        pooled = _pool(dataset)
        config = RidgeSurrogateConfig(n_bootstraps=50, ei_xi=0.01, cv_folds=5, target_rank_correlation=0.95)
        surrogate = config.create_surrogate(
            mixer_names=pooled.mixer_names, seed=42,
            feature_config=dataset.feature_config,
        )
        surrogate.fit(pooled.assignments, np.array(pooled.scores))

        assert pooled.num_samples <= 100
        assert pooled.num_layers == _NUM_LAYERS
        # Surrogate should be able to predict
        preds = surrogate.predict_mean(dataset.assignments[:5])
        assert len(preds) == 5

    async def test_pool_with_supplemental_dirs(self, tmp_path: Path) -> None:
        """from_jsonl + merge pools supplemental data from directories."""
        from place_layers.surrogate.evaluation_log import (
            EVALUATION_RECORDS_FILENAME,
            EvaluationLogMeta,
            EvaluationRecord,
            PooledEvaluationData,
            append_record,
            discover_evaluation_records,
            write_meta,
        )

        evaluator = _make_evaluator()
        dataset = await _build_exploration_dataset(evaluator, num_samples=80)
        dataset_path = _save_dataset(dataset, tmp_path)

        # Write supplemental JSONL in a subdirectory using standard naming
        supplemental_dir = tmp_path / "supplemental" / "weighted_average"
        supplemental_dir.mkdir(parents=True)
        jsonl_path = supplemental_dir / EVALUATION_RECORDS_FILENAME
        extra_records = [
            EvaluationRecord(
                assignment=("m0",) * 6 + ("m1",) * 6,
                score=0.42,
            ),
            EvaluationRecord(
                assignment=("m1",) * 6 + ("m0",) * 6,
                score=0.38,
            ),
        ]
        with jsonl_path.open("w") as f:
            for rec in extra_records:
                append_record(f, rec)
        meta = EvaluationLogMeta(
            version=1,
            objective_name="weighted_average",
            num_layers=_NUM_LAYERS,
            mixer_names=["m0", "m1"],
            source_command="test",
            source_detail="test",
            created_at="2026-01-01T00:00:00+00:00",
            num_records=2,
        )
        write_meta(jsonl_path, meta)

        # Base: exploration data only
        ridge_config = RidgeSurrogateConfig(n_bootstraps=50, ei_xi=0.01, cv_folds=5, target_rank_correlation=0.95)
        base_pooled = PooledEvaluationData.from_exploration(dataset)
        base_assignments = list(base_pooled.assignments)
        base_scores = list(base_pooled.scores)
        surr_base = ridge_config.create_surrogate(
            mixer_names=list(dataset.mixer_names), seed=42,
            feature_config=dataset.feature_config,
        )
        surr_base.fit(base_assignments, np.array(base_scores))

        # With supplemental: exploration + discovered JSONL files
        supplemental_files = discover_evaluation_records(
            [tmp_path / "supplemental"], objective_name=dataset.objective_name,
        )
        pooled = PooledEvaluationData.from_exploration(dataset)
        pooled = pooled.merge(PooledEvaluationData.from_jsonl(supplemental_files))
        all_assignments = list(pooled.assignments)
        all_scores = list(pooled.scores)

        surr_pooled = ridge_config.create_surrogate(
            mixer_names=list(dataset.mixer_names), seed=42,
            feature_config=dataset.feature_config,
        )
        surr_pooled.fit(all_assignments, np.array(all_scores))

        # Merged with supplemental has more samples than base
        assert len(all_assignments) > len(base_assignments)
        # Both surrogates produce valid predictions
        test_assignment = ("m0",) * 6 + ("m1",) * 6
        pred_base = surr_base.predict_mean_one(test_assignment)
        pred_pooled = surr_pooled.predict_mean_one(test_assignment)
        assert np.isfinite(pred_base)
        assert np.isfinite(pred_pooled)

    async def test_compute_band_edges(self, tmp_path: Path) -> None:
        """Band edges should be sorted ascending with correct count."""
        cost_model = _COST_MODEL_CONFIG.create_cost_model()
        all_allocs = enumerate_feasible_allocations(
            cost_model, _NUM_LAYERS, cost_budget=None
        )
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocs]
        )
        baseline_cost = cost_model.max_cost(_NUM_LAYERS)

        edges = compute_band_edges(alloc_costs, baseline_cost, num_cost_bins=10)
        assert len(edges) == 11  # 10 bins => 11 edges
        # Should be sorted ascending
        assert np.all(np.diff(edges) >= 0)

    async def test_compute_bin_assignments(self) -> None:
        """Bin assignment should clip to [0, num_cost_bins-1]."""
        num_cost_bins = 5
        band_edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 6 edges for 5 bins
        alloc_costs = np.array([0.5, 1.5, 2.5, 4.5, 7.0])

        bins = compute_bin_assignments(alloc_costs, band_edges, num_cost_bins)

        assert len(bins) == len(alloc_costs)
        # All bins should be in [0, num_cost_bins - 1]
        assert np.all(bins >= 0)
        assert np.all(bins <= num_cost_bins - 1)
        # Below first edge -> clipped to 0
        assert bins[0] == 0
        # Above last edge -> clipped to num_cost_bins - 1
        assert bins[4] == num_cost_bins - 1

    async def test_compute_dp_per_band(self, tmp_path: Path) -> None:
        """DP per band should return sorted BandResult list."""
        evaluator = _make_evaluator()
        dataset = await _build_exploration_dataset(evaluator, num_samples=100)
        _save_dataset(dataset, tmp_path)

        pooled = _pool(dataset)
        config = RidgeSurrogateConfig(n_bootstraps=50, ei_xi=0.01, cv_folds=5, target_rank_correlation=0.95)
        surrogate = config.create_surrogate(
            mixer_names=pooled.mixer_names, seed=42,
            feature_config=dataset.feature_config,
        )
        surrogate.fit(pooled.assignments, np.array(pooled.scores))
        cost_model = _COST_MODEL_CONFIG.create_cost_model()

        all_allocs = enumerate_feasible_allocations(
            cost_model, _NUM_LAYERS, cost_budget=None
        )
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocs]
        )
        baseline_cost = cost_model.max_cost(_NUM_LAYERS)

        bands, _edges = compute_dp_per_band(
            surrogate, all_allocs, alloc_costs,
            _NUM_LAYERS, baseline_cost, cost_model,
            num_cost_bins=10,
        )

        assert len(bands) > 0
        # All bands should have valid structure
        for b in bands:
            assert isinstance(b, BandResult)
            assert len(b.assignments) == _NUM_LAYERS
            assert b.cost_low <= b.cost <= b.cost_high or b.cost <= b.cost_high
            assert b.band_index >= 0

        # Should be sorted by cost
        costs = [b.cost for b in bands]
        assert costs == sorted(costs)


# =============================================================================
# Tests: Pydantic model serialization
# =============================================================================


class TestPydanticModels:
    """Test Pydantic model serialization for validation results."""

    def test_validation_results_roundtrip(self, tmp_path: Path) -> None:
        """ValidationResults should serialize and deserialize correctly."""
        results = ValidationResults(
            objective_name="gsm8k",
            num_layers=12,
            baseline_cost=12.0,
            num_cost_bins=10,
            cost_bin_edges=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            bands=[
                BandValidationResult(
                    band_index=0,
                    cost_low=6.0,
                    cost_high=7.0,
                    validated=PlacementResult(
                        cost=6.5, score=0.28,
                        assignments=("m0",) * 6 + ("m1",) * 6,
                    ),
                    total_evaluations=20,
                ),
            ],
            summary=ValidationSummary(
                total_bands=1,
                total_evaluations=20,
            ),
        )

        path = tmp_path / "validation_results.json"
        results.save(path)
        loaded = ValidationResults.load(path)

        assert loaded.objective_name == "gsm8k"
        assert loaded.num_layers == 12
        assert loaded.baseline_cost == 12.0
        assert len(loaded.bands) == 1
        assert loaded.bands[0].validated.score == 0.28
        assert loaded.summary.total_bands == 1

    def test_validation_results_backward_compat(self, tmp_path: Path) -> None:
        """ValidationResults without objective_name loads with default ""."""
        results = ValidationResults(
            num_layers=12,
            baseline_cost=12.0,
            num_cost_bins=10,
            cost_bin_edges=[6.0, 7.0, 8.0],
            bands=[],
            summary=ValidationSummary(
                total_bands=0,
                total_evaluations=0,
            ),
        )
        path = tmp_path / "validation_results.json"
        results.save(path)
        loaded = ValidationResults.load(path)
        assert loaded.objective_name == ""

    def test_validation_results_backward_compat_old_fields(self) -> None:
        """Old JSON with surrogate_dp/spearman_rho fields still loads OK."""
        old_json = json.dumps({
            "objective_name": "gsm8k",
            "num_layers": 12,
            "baseline_cost": 12.0,
            "num_cost_bins": 10,
            "cost_bin_edges": [6.0, 7.0, 8.0],
            "bands": [
                {
                    "band_index": 0,
                    "cost_low": 6.0,
                    "cost_high": 7.0,
                    "surrogate_dp": {
                        "cost": 6.5,
                        "score": 0.3,
                        "assignments": ["m0"] * 6 + ["m1"] * 6,
                    },
                    "validated": {
                        "cost": 6.5,
                        "score": 0.28,
                        "assignments": ["m0"] * 6 + ["m1"] * 6,
                    },
                    "total_evaluations": 20,
                },
            ],
            "summary": {
                "spearman_rho": 0.95,
                "mean_residual": -0.02,
                "total_bands": 1,
                "total_evaluations": 20,
            },
        })
        loaded = ValidationResults.model_validate_json(old_json)
        assert loaded.objective_name == "gsm8k"
        assert len(loaded.bands) == 1
        assert loaded.bands[0].validated.score == 0.28
        assert loaded.summary.total_bands == 1
        assert loaded.summary.total_evaluations == 20

    def test_landscape_data_roundtrip(self, tmp_path: Path) -> None:
        """LandscapeData should serialize and deserialize correctly."""
        data = LandscapeData(
            objective_name="mmlu",
            num_layers=12,
            baseline_cost=12.0,
            num_cost_bins=10,
            cost_bin_edges=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            bands=[
                BandResult(
                    band_index=0,
                    cost_low=6.0,
                    cost_high=7.0,
                    cost=6.5,
                    score=0.3,
                    assignments=("m0",) * 6 + ("m1",) * 6,
                ),
            ],
        )

        path = tmp_path / "landscape_data.json"
        data.save(path)
        loaded = LandscapeData.load(path)

        assert loaded.objective_name == "mmlu"
        assert loaded.num_layers == 12
        assert len(loaded.bands) == 1
        assert loaded.bands[0].cost == 6.5

    def test_landscape_data_backward_compat(self, tmp_path: Path) -> None:
        """LandscapeData without objective_name loads with default ""."""
        data = LandscapeData(
            num_layers=12,
            baseline_cost=12.0,
            num_cost_bins=10,
            cost_bin_edges=[6.0, 7.0, 8.0],
            bands=[],
        )
        path = tmp_path / "landscape_data.json"
        data.save(path)
        loaded = LandscapeData.load(path)
        assert loaded.objective_name == ""


# =============================================================================
# Tests: per-band validation pipeline
# =============================================================================


class TestValidateLandscapePipeline:
    """Integration test: per-band search with mock evaluator."""

    async def test_per_band_validation(self, tmp_path: Path) -> None:
        """Run per-band ViterbiSurrogateSearch and verify results."""
        evaluator = _make_evaluator(noise_sigma=0.0)
        dataset = await _build_exploration_dataset(evaluator, num_samples=100)
        dataset_path = _save_dataset(dataset, tmp_path)

        # 1. Fit surrogate, compute bands
        pooled = _pool(dataset)
        ridge_config = RidgeSurrogateConfig(n_bootstraps=50, ei_xi=0.01, cv_folds=5, target_rank_correlation=0.95)
        surrogate = ridge_config.create_surrogate(
            mixer_names=pooled.mixer_names, seed=42,
            feature_config=dataset.feature_config,
        )
        surrogate.fit(pooled.assignments, np.array(pooled.scores))
        cost_model = _COST_MODEL_CONFIG.create_cost_model()

        all_allocs = enumerate_feasible_allocations(
            cost_model, _NUM_LAYERS, cost_budget=None
        )
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocs]
        )
        baseline_cost = cost_model.max_cost(_NUM_LAYERS)

        # Use fewer bins for faster test
        num_cost_bins = 5
        band_results, band_edges = compute_dp_per_band(
            surrogate, all_allocs, alloc_costs,
            _NUM_LAYERS, baseline_cost, cost_model,
            num_cost_bins,
        )

        assert len(band_results) > 0

        # 2. Run per-band search (just first 2 bands for speed)
        bands_to_validate = band_results[:2]
        validated_bands: list[BandValidationResult] = []

        async with evaluator:
            for band in bands_to_validate:
                problem_config = MaxPerformanceProblemConfig(
                    problem_type="cost_max_performance",
                    cost_model=_COST_MODEL_CONFIG,
                    cost_budget=band.cost_high,
                    cost_floor=band.cost_low,
                )
                problem = CostConstrainedProblem(
                    evaluators=wrap_evaluator(evaluator), config=problem_config,
                    objective_name="weighted_average",
                )

                search_config = ViterbiSurrogateSearchConfig(
                    strategy_type="viterbi_surrogate",
                    exploration_dataset_path=str(dataset_path),
                    evaluation_log_path=None,
                    num_rounds=1,
                    evals_per_round=10,
                    candidates_per_model=50,
                    surrogate=dict(
                        surrogate_type="ridge",
                        n_bootstraps=50,
                        ei_xi=0.01,
                        cv_folds=3,
                        target_rank_correlation=0.95,
                    ),
                    seed=42,
                )
                strategy = search_config.create_search_strategy()
                result = await strategy.search(problem)

                validated_bands.append(
                    BandValidationResult(
                        band_index=band.band_index,
                        cost_low=band.cost_low,
                        cost_high=band.cost_high,
                        validated=PlacementResult(
                            cost=result.final_cost,
                            score=result.best_score,
                            assignments=result.best_parameters.assignments,
                        ),
                        total_evaluations=result.total_evaluations,
                    )
                )

        # 3. Verify results
        assert len(validated_bands) == 2
        for vb in validated_bands:
            assert vb.total_evaluations > 0
            assert len(vb.validated.assignments) == _NUM_LAYERS

        # 4. Save and reload
        total_evals = sum(b.total_evaluations for b in validated_bands)

        results = ValidationResults(
            num_layers=_NUM_LAYERS,
            baseline_cost=baseline_cost,
            num_cost_bins=num_cost_bins,
            cost_bin_edges=band_edges.tolist(),
            bands=validated_bands,
            summary=ValidationSummary(
                total_bands=len(validated_bands),
                total_evaluations=total_evals,
            ),
        )

        results_path = tmp_path / "validation_results.json"
        results.save(results_path)
        loaded = ValidationResults.load(results_path)

        assert loaded.num_layers == _NUM_LAYERS
        assert len(loaded.bands) == 2
        assert loaded.summary.total_evaluations == total_evals

    async def test_incremental_save_and_resume(self, tmp_path: Path) -> None:
        """Per-band incremental save: pre-existing band is skipped on resume."""
        evaluator = _make_evaluator(noise_sigma=0.0)
        dataset = await _build_exploration_dataset(evaluator, num_samples=100)
        dataset_path = _save_dataset(dataset, tmp_path)

        # 1. Fit surrogate, compute bands
        pooled = _pool(dataset)
        ridge_config = RidgeSurrogateConfig(n_bootstraps=50, ei_xi=0.01, cv_folds=5, target_rank_correlation=0.95)
        surrogate = ridge_config.create_surrogate(
            mixer_names=pooled.mixer_names, seed=42,
            feature_config=dataset.feature_config,
        )
        surrogate.fit(pooled.assignments, np.array(pooled.scores))
        cost_model = _COST_MODEL_CONFIG.create_cost_model()

        all_allocs = enumerate_feasible_allocations(
            cost_model, _NUM_LAYERS, cost_budget=None
        )
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocs]
        )
        baseline_cost = cost_model.max_cost(_NUM_LAYERS)

        num_cost_bins = 5
        band_results, _band_edges = compute_dp_per_band(
            surrogate, all_allocs, alloc_costs,
            _NUM_LAYERS, baseline_cost, cost_model,
            num_cost_bins,
        )
        assert len(band_results) >= 2

        # 2. "Run" band 0 and save to disk (simulate partial completion)
        bands_dir = tmp_path / "bands"
        band = band_results[0]
        band_dir = bands_dir / f"band_{band.band_index:03d}"
        band_dir.mkdir(parents=True)

        async with evaluator:
            problem_config = MaxPerformanceProblemConfig(
                problem_type="cost_max_performance",
                cost_model=_COST_MODEL_CONFIG,
                cost_budget=band.cost_high,
                cost_floor=band.cost_low,
            )
            problem = CostConstrainedProblem(
                evaluators=wrap_evaluator(evaluator), config=problem_config,
                objective_name="weighted_average",
            )
            search_config = ViterbiSurrogateSearchConfig(
                strategy_type="viterbi_surrogate",
                exploration_dataset_path=str(dataset_path),
                evaluation_log_path=None,
                num_rounds=1,
                evals_per_round=10,
                candidates_per_model=50,
                surrogate=dict(
                    surrogate_type="ridge",
                    n_bootstraps=50,
                    ei_xi=0.01,
                    cv_folds=3,
                    target_rank_correlation=0.95,
                ),
                seed=42,
            )
            strategy = search_config.create_search_strategy()
            result = await strategy.search(problem)

        result_path = band_dir / "result.json"
        result_path.write_text(result.model_dump_json(indent=2))

        # 3. Now simulate resume: detect the pre-saved band
        loaded_result = json.loads(result_path.read_text())

        loaded_score = loaded_result["best_score"]

        resumed_band = BandValidationResult(
            band_index=band.band_index,
            cost_low=band.cost_low,
            cost_high=band.cost_high,
            validated=PlacementResult(
                cost=loaded_result["final_cost"],
                score=loaded_score,
                assignments=tuple(
                    loaded_result["best_parameters"]["assignments"]
                ),
            ),
            total_evaluations=loaded_result["total_evaluations"],
        )

        # Verify restored band matches original result
        assert resumed_band.total_evaluations == result.total_evaluations
        expected_score = result.best_score
        assert resumed_band.validated.score == expected_score
        assert resumed_band.validated.cost == result.final_cost

    async def test_per_band_validation_bayesian(self, tmp_path: Path) -> None:
        """Run per-band ViterbiBayesianSurrogateSearch and verify results."""
        from place_layers.surrogate.config import BayesianRidgeSurrogateConfig

        evaluator = _make_evaluator(noise_sigma=0.0)
        dataset = await _build_exploration_dataset(evaluator, num_samples=100)
        dataset_path = _save_dataset(dataset, tmp_path)

        # 1. Fit Bayesian surrogate, compute bands
        bayesian_config = BayesianRidgeSurrogateConfig(
            prior_alpha=1.0,
            prior_a0=1e-6,
            prior_b0=1e-6,
            alpha_selection="fixed",
            feature_scaling="none",
        )
        pooled = _pool(dataset)
        surrogate = bayesian_config.create_surrogate(
            mixer_names=pooled.mixer_names, seed=42,
            feature_config=dataset.feature_config,
        )
        surrogate.fit(pooled.assignments, np.array(pooled.scores))
        cost_model = _COST_MODEL_CONFIG.create_cost_model()

        all_allocs = enumerate_feasible_allocations(
            cost_model, _NUM_LAYERS, cost_budget=None
        )
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocs]
        )
        baseline_cost = cost_model.max_cost(_NUM_LAYERS)

        num_cost_bins = 5
        band_results, band_edges = compute_dp_per_band(
            surrogate, all_allocs, alloc_costs,
            _NUM_LAYERS, baseline_cost, cost_model,
            num_cost_bins,
        )

        assert len(band_results) > 0

        # 2. Run per-band Bayesian search (just first 2 bands for speed)
        bands_to_validate = band_results[:2]
        validated_bands: list[BandValidationResult] = []

        async with evaluator:
            for band in bands_to_validate:
                problem_config = MaxPerformanceProblemConfig(
                    problem_type="cost_max_performance",
                    cost_model=_COST_MODEL_CONFIG,
                    cost_budget=band.cost_high,
                    cost_floor=band.cost_low,
                )
                problem = CostConstrainedProblem(
                    evaluators=wrap_evaluator(evaluator), config=problem_config,
                    objective_name="weighted_average",
                )

                search_config = ViterbiBayesianSurrogateSearchConfig(
                    strategy_type="viterbi_bayesian_surrogate",
                    exploration_dataset_path=str(dataset_path),
                    evaluation_log_path=None,
                    num_rounds=1,
                    evals_per_round=10,
                    candidates_per_model=50,
                    surrogate=dict(
                        surrogate_type="bayesian_ridge",
                        prior_alpha=1.0,
                        prior_a0=1e-6,
                        prior_b0=1e-6,
                        alpha_selection="fixed",
                        feature_scaling="none",
                    ),
                    risk_buckets=[
                        RiskBucketConfig(name="safe", sign=-1, betas=[0.0, 1.0], quota=0.6, mean_floor_quantile=0.0),
                        RiskBucketConfig(name="upside", sign=1, betas=[1.0], quota=0.3, mean_floor_quantile=0.5),
                    ],
                    seed=42,
                )
                strategy = search_config.create_search_strategy()
                result = await strategy.search(problem)

                validated_bands.append(
                    BandValidationResult(
                        band_index=band.band_index,
                        cost_low=band.cost_low,
                        cost_high=band.cost_high,
                        validated=PlacementResult(
                            cost=result.final_cost,
                            score=result.best_score,
                            assignments=result.best_parameters.assignments,
                        ),
                        total_evaluations=result.total_evaluations,
                    )
                )

        # 3. Verify results
        assert len(validated_bands) == 2
        for vb in validated_bands:
            assert vb.total_evaluations > 0
            assert len(vb.validated.assignments) == _NUM_LAYERS


# =============================================================================
# Tests: LandscapeConfig
# =============================================================================


class TestLandscapeConfig:
    """Test LandscapeConfig Pydantic model (no defaults)."""

    def test_landscape_config_requires_all_fields(self) -> None:
        """LandscapeConfig has no defaults — all fields are required."""
        with pytest.raises(Exception):
            LandscapeConfig()  # type: ignore[call-arg]

    def test_landscape_config_construction(self) -> None:
        """LandscapeConfig can be constructed with all fields."""
        cfg = LandscapeConfig(
            cost_model=_COST_MODEL_CONFIG,
            explore_output_dir="/tmp/test_dir",
            seed=42,
            num_cost_bins=80,
            surrogate=RidgeSurrogateConfig(n_bootstraps=50, ei_xi=0.01, cv_folds=5, target_rank_correlation=0.95),
            supplemental_data_dirs=[],
            min_layers_per_mixer=0,
        )
        assert cfg.seed == 42
        assert cfg.num_cost_bins == 80
        assert cfg.explore_output_dir == "/tmp/test_dir"

    def test_landscape_config_frozen(self) -> None:
        """LandscapeConfig should be immutable."""
        cfg = LandscapeConfig(
            cost_model=_COST_MODEL_CONFIG,
            explore_output_dir="/tmp/test_dir",
            seed=42,
            num_cost_bins=80,
            surrogate=RidgeSurrogateConfig(n_bootstraps=50, ei_xi=0.01, cv_folds=5, target_rank_correlation=0.95),
            supplemental_data_dirs=[],
            min_layers_per_mixer=0,
        )
        with pytest.raises(Exception):
            cfg.seed = 99  # type: ignore[misc]


# =============================================================================
# Tests: objective_result_path helper
# =============================================================================


class TestObjectiveResultPath:
    """Test the shared objective_result_path helper."""

    def test_produces_correct_path(self) -> None:
        """objective_result_path produces expected filename."""
        p = objective_result_path(Path("/out"), "exploration_results", "gsm8k")
        assert p == Path("/out/gsm8k/exploration_results.json")

    def test_different_prefixes(self) -> None:
        """Works with different prefixes."""
        p1 = objective_result_path(Path("/d"), "exploration_results", "mmlu")
        p2 = objective_result_path(Path("/d"), "validation_results", "mmlu")
        p3 = objective_result_path(Path("/d"), "landscape_data", "mmlu")
        assert p1 == Path("/d/mmlu/exploration_results.json")
        assert p2 == Path("/d/mmlu/validation_results.json")
        assert p3 == Path("/d/mmlu/landscape_data.json")

    def test_objective_name_with_special_chars(self) -> None:
        """Objective names are used as-is in the directory."""
        p = objective_result_path(Path("/d"), "exploration_results", "weighted_average")
        assert p == Path("/d/weighted_average/exploration_results.json")


# =============================================================================
# Tests: EvaluatorsConfig.objective_names property
# =============================================================================


class TestEvaluatorsConfigObjectiveNames:
    """Test EvaluatorsConfig.objective_names derives names from config data."""

    def test_independent_objective(self) -> None:
        """Independent objective -> evaluator names as objectives."""
        cfg = EvaluatorsConfig(
            configs={
                "gsm8k": _make_evaluator_config(),
                "mmlu": _make_evaluator_config(noise_seed_shift=1),
            },
            objective=IndependentObjectiveConfig(objective_type="independent"),
        )
        assert cfg.objective_names == ("gsm8k", "mmlu")

    def test_weighted_average_no_individual(self) -> None:
        """Weighted average without include_individual -> single aggregate name."""
        cfg = EvaluatorsConfig(
            configs={
                "gsm8k": _make_evaluator_config(),
                "mmlu": _make_evaluator_config(noise_seed_shift=1),
            },
            objective=WeightedAverageObjectiveConfig(
                objective_type="weighted_average",
                include_individual=False,
                weights={"gsm8k": 0.5, "mmlu": 0.5},
            ),
        )
        assert cfg.objective_names == ("weighted_average",)

    def test_weighted_average_with_individual(self) -> None:
        """Weighted average with include_individual -> aggregate + evaluator names."""
        cfg = EvaluatorsConfig(
            configs={
                "gsm8k": _make_evaluator_config(),
                "mmlu": _make_evaluator_config(noise_seed_shift=1),
            },
            objective=WeightedAverageObjectiveConfig(
                objective_type="weighted_average",
                include_individual=True,
                weights={"gsm8k": 0.5, "mmlu": 0.5},
            ),
        )
        assert cfg.objective_names == ("weighted_average", "gsm8k", "mmlu")

    def test_max_objective_with_individual(self) -> None:
        """Max objective with include_individual -> ('max',) + evaluator names."""
        cfg = EvaluatorsConfig(
            configs={
                "gsm8k": _make_evaluator_config(),
                "mmlu": _make_evaluator_config(noise_seed_shift=1),
            },
            objective=MaxObjectiveConfig(
                objective_type="max",
                include_individual=True,
            ),
        )
        assert cfg.objective_names == ("max", "gsm8k", "mmlu")

    def test_single_evaluator_weighted_average(self) -> None:
        """Single evaluator with weighted_average -> ('weighted_average',)."""
        cfg = EvaluatorsConfig(
            configs={"main": _make_evaluator_config()},
            objective=WeightedAverageObjectiveConfig(
                objective_type="weighted_average",
                include_individual=False,
            ),
        )
        assert cfg.objective_names == ("weighted_average",)


# =============================================================================
# Tests: multi-objective validation pipeline
# =============================================================================


class TestMultiObjectiveValidation:
    """Test multi-objective support in the validation pipeline."""

    async def test_multi_objective_exploration_datasets(self, tmp_path: Path) -> None:
        """Save and load per-objective exploration datasets via objective_result_path."""
        evaluator = _make_evaluator()
        explore_dir = tmp_path / "explore_output"
        explore_dir.mkdir()

        # Simulate 2-objective exploration output
        for obj_name in ("gsm8k", "mmlu"):
            dataset = await _build_exploration_dataset(
                evaluator, num_samples=50, objective_name=obj_name,
            )
            path = objective_result_path(explore_dir, "exploration_results", obj_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dataset.model_dump_json(indent=2))

        # Verify both files exist and can be loaded
        for obj_name in ("gsm8k", "mmlu"):
            path = objective_result_path(explore_dir, "exploration_results", obj_name)
            assert path.exists()
            loaded = ExplorationDataset.load(path)
            assert loaded.objective_name == obj_name
            assert loaded.num_samples == 50

    async def test_multi_objective_validation_results(self, tmp_path: Path) -> None:
        """Save per-objective validation results via objective_result_path."""
        output_dir = tmp_path / "validate_output"
        output_dir.mkdir()

        for obj_name in ("gsm8k", "mmlu"):
            results = ValidationResults(
                objective_name=obj_name,
                num_layers=12,
                baseline_cost=12.0,
                num_cost_bins=5,
                cost_bin_edges=[6.0, 8.0, 10.0, 12.0],
                bands=[],
                summary=ValidationSummary(
                    total_bands=0,
                    total_evaluations=0,
                ),
            )
            path = objective_result_path(output_dir, "validation_results", obj_name)
            results.save(path)

        # Verify both exist and have correct objective_name
        for obj_name in ("gsm8k", "mmlu"):
            path = objective_result_path(output_dir, "validation_results", obj_name)
            assert path.exists()
            loaded = ValidationResults.load(path)
            assert loaded.objective_name == obj_name


# =============================================================================
# Tests: incremental validation results
# =============================================================================


class TestIncrementalValidationResults:
    """Test incremental saving of validation results."""

    async def test_incremental_validation_results_written(self, tmp_path: Path) -> None:
        """validation_results_*.json should be written after each band."""
        from place_layers.command.validate_landscape import ValidateLandscapeCommand

        evaluator = _make_evaluator(noise_sigma=0.0)
        dataset = await _build_exploration_dataset(evaluator, num_samples=100)
        dataset_path = _save_dataset(dataset, tmp_path)

        pooled = _pool(dataset)
        ridge_config = RidgeSurrogateConfig(n_bootstraps=50, ei_xi=0.01, cv_folds=5, target_rank_correlation=0.95)
        surrogate = ridge_config.create_surrogate(
            mixer_names=pooled.mixer_names, seed=42,
            feature_config=dataset.feature_config,
        )
        surrogate.fit(pooled.assignments, np.array(pooled.scores))
        cost_model = _COST_MODEL_CONFIG.create_cost_model()

        all_allocs = enumerate_feasible_allocations(
            cost_model, _NUM_LAYERS, cost_budget=None
        )
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocs]
        )
        baseline_cost = cost_model.max_cost(_NUM_LAYERS)

        num_cost_bins = 5
        band_results, band_edges = compute_dp_per_band(
            surrogate, all_allocs, alloc_costs,
            _NUM_LAYERS, baseline_cost, cost_model, num_cost_bins,
        )

        # Validate 3 bands and check the results file after each
        bands_to_validate = band_results[:3]
        validated_bands: list[BandValidationResult] = []
        total_evals = 0
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        results_path = objective_result_path(
            output_dir, "validation_results", "weighted_average",
        )

        async with evaluator:
            for i, band in enumerate(bands_to_validate):
                problem_config = MaxPerformanceProblemConfig(
                    problem_type="cost_max_performance",
                    cost_model=_COST_MODEL_CONFIG,
                    cost_budget=band.cost_high,
                    cost_floor=band.cost_low,
                )
                problem = CostConstrainedProblem(
                    evaluators=wrap_evaluator(evaluator), config=problem_config,
                    objective_name="weighted_average",
                )

                search_config = ViterbiSurrogateSearchConfig(
                    strategy_type="viterbi_surrogate",
                    exploration_dataset_path=str(dataset_path),
                    evaluation_log_path=None,
                    num_rounds=1,
                    evals_per_round=10,
                    candidates_per_model=50,
                    surrogate=dict(
                        surrogate_type="ridge",
                        n_bootstraps=50,
                        ei_xi=0.01,
                        cv_folds=3,
                        target_rank_correlation=0.95,
                    ),
                    seed=42,
                )
                strategy = search_config.create_search_strategy()
                result = await strategy.search(problem)

                total_evals += result.total_evaluations
                validated_bands.append(
                    BandValidationResult(
                        band_index=band.band_index,
                        cost_low=band.cost_low,
                        cost_high=band.cost_high,
                        validated=PlacementResult(
                            cost=result.final_cost,
                            score=result.best_score,
                            assignments=result.best_parameters.assignments,
                        ),
                        total_evaluations=result.total_evaluations,
                    )
                )

                # Use the helper to save incrementally
                ValidateLandscapeCommand._save_validation_results(
                    obj_name="weighted_average",
                    validated_bands=validated_bands,
                    num_layers=_NUM_LAYERS,
                    baseline_cost=baseline_cost,
                    num_cost_bins=num_cost_bins,
                    band_edges=band_edges,
                    total_evals=total_evals,
                    output_dir=output_dir,
                )

                # Verify file exists after this band
                assert results_path.exists(), (
                    f"validation_results should exist after band {i}"
                )
                loaded = ValidationResults.load(results_path)
                assert len(loaded.bands) == i + 1
                assert loaded.summary.total_evaluations == total_evals

        # Final file should have all bands
        final = ValidationResults.load(results_path)
        assert len(final.bands) == 3


# =============================================================================
# Tests: cross-band data loading
# =============================================================================


class TestCrossBandDataLoading:
    """Test _load_completed_band_records and cross-band accumulation."""

    def test_load_completed_band_records(self, tmp_path: Path) -> None:
        """_load_completed_band_records returns paths and record count."""
        from place_layers.command.validate_landscape import _load_completed_band_records
        from place_layers.surrogate.evaluation_log import (
            EVALUATION_RECORDS_FILENAME,
            EvaluationRecord,
            append_record,
            load_records,
        )

        bands_dir = tmp_path / "bands"

        # Create band 0 with 2 strategy-only records
        band_dir = bands_dir / "band_000"
        band_dir.mkdir(parents=True)
        eval_log = band_dir / EVALUATION_RECORDS_FILENAME
        with eval_log.open("w") as f:
            for i in range(2):
                append_record(f, EvaluationRecord(
                    assignment=("m1",) * 6 + ("m0",) * 6,
                    score=10.0 + i,
                ))
        result_path = band_dir / "result.json"
        result_path.write_text('{"total_evaluations": 2}')

        # Load paths for band 0
        paths = _load_completed_band_records(bands_dir, [0])

        assert len(paths) == 1
        records = load_records(paths[0])
        assert records[0].score == 10.0
        assert records[1].score == 11.0

    def test_load_completed_band_records_skips_missing(self, tmp_path: Path) -> None:
        """_load_completed_band_records skips bands without result.json."""
        from place_layers.command.validate_landscape import _load_completed_band_records

        bands_dir = tmp_path / "bands"
        bands_dir.mkdir()

        # No band dirs exist
        paths = _load_completed_band_records(bands_dir, [0, 1, 2])
        assert paths == []

    def test_cross_band_seed_paths(self, tmp_path: Path) -> None:
        """Prior band eval logs become seed_data_paths for the next band."""
        from place_layers.command.validate_landscape import _load_completed_band_records
        from place_layers.surrogate.evaluation_log import (
            EVALUATION_RECORDS_FILENAME,
            EvaluationRecord,
            append_record,
        )

        bands_dir = tmp_path / "bands"

        # Create band 0 with only strategy records
        band0_dir = bands_dir / "band_000"
        band0_dir.mkdir(parents=True)
        eval_log = band0_dir / EVALUATION_RECORDS_FILENAME
        with eval_log.open("w") as f:
            append_record(f, EvaluationRecord(
                assignment=("m0",) * 6 + ("m1",) * 6,
                score=0.5,
            ))
            append_record(f, EvaluationRecord(
                assignment=("m1",) * 6 + ("m0",) * 6,
                score=0.6,
            ))
        (band0_dir / "result.json").write_text('{"total_evaluations": 2}')

        # Loading prior records for band 1
        paths = _load_completed_band_records(bands_dir, [0])

        assert len(paths) == 1

    def test_multi_band_seed_paths(self, tmp_path: Path) -> None:
        """Multiple completed bands produce multiple seed paths."""
        from place_layers.command.validate_landscape import _load_completed_band_records
        from place_layers.surrogate.evaluation_log import (
            EVALUATION_RECORDS_FILENAME,
            EvaluationRecord,
            append_record,
            load_records,
        )

        bands_dir = tmp_path / "bands"

        # Create band 0 with 2 strategy records
        band0_dir = bands_dir / "band_000"
        band0_dir.mkdir(parents=True)
        eval_log0 = band0_dir / EVALUATION_RECORDS_FILENAME
        with eval_log0.open("w") as f:
            for i in range(2):
                append_record(f, EvaluationRecord(
                    assignment=("m0",) * 6 + ("m1",) * 6,
                    score=2.0 + i,
                ))
        (band0_dir / "result.json").write_text('{"total_evaluations": 2}')

        # Create band 1 with 3 strategy records
        band1_dir = bands_dir / "band_001"
        band1_dir.mkdir(parents=True)
        eval_log1 = band1_dir / EVALUATION_RECORDS_FILENAME
        with eval_log1.open("w") as f:
            for i in range(3):
                append_record(f, EvaluationRecord(
                    assignment=("m1",) * 12,
                    score=5.0 + i,
                ))
        (band1_dir / "result.json").write_text('{"total_evaluations": 3}')

        # Load both bands
        paths = _load_completed_band_records(bands_dir, [0, 1])

        assert len(paths) == 2
        # Verify records from each band
        records0 = load_records(paths[0])
        records1 = load_records(paths[1])
        assert len(records0) == 2
        assert len(records1) == 3
        assert records0[0].score == 2.0
        assert records1[0].score == 5.0
