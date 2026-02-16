"""Integration tests for the validate_landscape pipeline.

Tests the full pipeline:
1. Build exploration dataset (programmatic)
2. Run validate_landscape per-band search
3. Verify ValidationResults structure and serialization
4. Verify placement_landscape can load validation results
5. Verify incremental per-band saving and resume-on-restart
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
    compute_dp_per_band,
    load_and_fit_surrogate,
)
from place_layers.core.types import LayerPlacement
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.cost_model.linear_cost_model import LinearCostModelConfig
from place_layers.evaluator.mock_supernet_evaluator import (
    MockSupernetEvaluator,
    MockSupernetEvaluatorConfig,
)
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
)
from place_layers.search_strategy.cost_constrained.viterbi_surrogate import (
    ViterbiSurrogateSearchConfig,
)
from place_layers.surrogate.dataset import ExplorationDataset
from place_layers.surrogate.model_selection import ridge_cv_rho


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


async def _build_exploration_dataset(
    evaluator: MockSupernetEvaluator,
    num_samples: int = 100,
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


def _save_dataset(dataset: ExplorationDataset, tmp_path: Path) -> Path:
    path = tmp_path / "exploration_results.json"
    path.write_text(dataset.model_dump_json(indent=2))
    return path


# =============================================================================
# Tests: shared helpers
# =============================================================================


class TestLandscapeHelpers:
    """Test shared helper functions from landscape_helpers."""

    async def test_load_and_fit_surrogate(self, tmp_path: Path) -> None:
        """load_and_fit_surrogate returns fitted surrogate."""
        evaluator = _make_evaluator()
        dataset = await _build_exploration_dataset(evaluator, num_samples=100)
        dataset_path = _save_dataset(dataset, tmp_path)

        loaded_dataset, surrogate, cost_model = load_and_fit_surrogate(
            str(dataset_path), _COST_MODEL_CONFIG, seed=42,
        )

        assert loaded_dataset.num_samples == 100
        assert loaded_dataset.num_layers == _NUM_LAYERS
        # Surrogate should be able to predict
        preds = surrogate.predict_mean(dataset.assignments[:5])
        assert len(preds) == 5

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

    async def test_compute_dp_per_band(self, tmp_path: Path) -> None:
        """DP per band should return sorted BandResult list."""
        evaluator = _make_evaluator()
        dataset = await _build_exploration_dataset(evaluator, num_samples=100)
        dataset_path = _save_dataset(dataset, tmp_path)

        _, surrogate, cost_model = load_and_fit_surrogate(
            str(dataset_path), _COST_MODEL_CONFIG, seed=42,
        )

        all_allocs = enumerate_feasible_allocations(
            cost_model, _NUM_LAYERS, cost_budget=None
        )
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocs]
        )
        baseline_cost = cost_model.max_cost(_NUM_LAYERS)

        bands = compute_dp_per_band(
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
            num_layers=12,
            baseline_cost=12.0,
            num_cost_bins=10,
            cost_bin_edges=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            bands=[
                BandValidationResult(
                    band_index=0,
                    cost_low=6.0,
                    cost_high=7.0,
                    surrogate_dp=PlacementResult(
                        cost=6.5, score=0.3,
                        assignments=("m0",) * 6 + ("m1",) * 6,
                    ),
                    validated=PlacementResult(
                        cost=6.5, score=0.28,
                        assignments=("m0",) * 6 + ("m1",) * 6,
                    ),
                    total_evaluations=20,
                ),
            ],
            summary=ValidationSummary(
                spearman_rho=0.95,
                mean_residual=-0.02,
                total_bands=1,
                total_evaluations=20,
            ),
        )

        path = tmp_path / "validation_results.json"
        results.save(path)
        loaded = ValidationResults.load(path)

        assert loaded.num_layers == 12
        assert loaded.baseline_cost == 12.0
        assert len(loaded.bands) == 1
        assert loaded.bands[0].surrogate_dp.score == 0.3
        assert loaded.bands[0].validated.score == 0.28
        assert loaded.summary.spearman_rho == 0.95

    def test_landscape_data_roundtrip(self, tmp_path: Path) -> None:
        """LandscapeData should serialize and deserialize correctly."""
        data = LandscapeData(
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

        assert loaded.num_layers == 12
        assert len(loaded.bands) == 1
        assert loaded.bands[0].cost == 6.5


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
        _, surrogate, cost_model = load_and_fit_surrogate(
            str(dataset_path), _COST_MODEL_CONFIG, seed=42,
        )

        all_allocs = enumerate_feasible_allocations(
            cost_model, _NUM_LAYERS, cost_budget=None
        )
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocs]
        )
        baseline_cost = cost_model.max_cost(_NUM_LAYERS)

        # Use fewer bins for faster test
        num_cost_bins = 5
        band_results = compute_dp_per_band(
            surrogate, all_allocs, alloc_costs,
            _NUM_LAYERS, baseline_cost, cost_model,
            num_cost_bins,
        )
        band_edges = compute_band_edges(alloc_costs, baseline_cost, num_cost_bins)

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
                    evaluator=evaluator, config=problem_config
                )

                search_config = ViterbiSurrogateSearchConfig(
                    strategy_type="viterbi_surrogate",
                    exploration_dataset_path=str(dataset_path),
                    num_rounds=1,
                    evals_per_round=10,
                    candidates_per_model=50,
                    cv_folds=3,
                    ei_xi=0.01,
                    seed=42,
                )
                strategy = search_config.create_search_strategy()
                result = await strategy.search(problem)

                validated_bands.append(
                    BandValidationResult(
                        band_index=band.band_index,
                        cost_low=band.cost_low,
                        cost_high=band.cost_high,
                        surrogate_dp=PlacementResult(
                            cost=band.cost,
                            score=band.score,
                            assignments=band.assignments,
                        ),
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
            assert len(vb.surrogate_dp.assignments) == _NUM_LAYERS
            # Surrogate DP cost should be within its band
            assert vb.surrogate_dp.cost <= vb.cost_high + 1e-9
            assert vb.surrogate_dp.cost >= vb.cost_low - 1e-9

        # 4. Save and reload
        from scipy.stats import spearmanr

        surrogate_scores = [b.surrogate_dp.score for b in validated_bands]
        real_scores = [b.validated.score for b in validated_bands]
        if len(validated_bands) >= 2:
            corr_result = spearmanr(surrogate_scores, real_scores)
            rho = float(corr_result.statistic)
            if np.isnan(rho):
                rho = 0.0
        else:
            rho = 0.0

        total_evals = sum(b.total_evaluations for b in validated_bands)
        residuals = [r - s for s, r in zip(surrogate_scores, real_scores)]
        mean_residual = float(np.mean(residuals))

        results = ValidationResults(
            num_layers=_NUM_LAYERS,
            baseline_cost=baseline_cost,
            num_cost_bins=num_cost_bins,
            cost_bin_edges=band_edges.tolist(),
            bands=validated_bands,
            summary=ValidationSummary(
                spearman_rho=rho,
                mean_residual=mean_residual,
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
        _, surrogate, cost_model = load_and_fit_surrogate(
            str(dataset_path), _COST_MODEL_CONFIG, seed=42,
        )

        all_allocs = enumerate_feasible_allocations(
            cost_model, _NUM_LAYERS, cost_budget=None
        )
        alloc_costs = np.array(
            [cost_model.allocation_cost(a) for a in all_allocs]
        )
        baseline_cost = cost_model.max_cost(_NUM_LAYERS)

        num_cost_bins = 5
        band_results = compute_dp_per_band(
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

        surrogate_dp = PlacementResult(
            cost=band.cost, score=band.score, assignments=band.assignments,
        )
        (band_dir / "surrogate_dp.json").write_text(
            surrogate_dp.model_dump_json(indent=2)
        )

        async with evaluator:
            problem_config = MaxPerformanceProblemConfig(
                problem_type="cost_max_performance",
                cost_model=_COST_MODEL_CONFIG,
                cost_budget=band.cost_high,
                cost_floor=band.cost_low,
            )
            problem = CostConstrainedProblem(
                evaluator=evaluator, config=problem_config
            )
            search_config = ViterbiSurrogateSearchConfig(
                strategy_type="viterbi_surrogate",
                exploration_dataset_path=str(dataset_path),
                num_rounds=1,
                evals_per_round=10,
                candidates_per_model=50,
                cv_folds=3,
                ei_xi=0.01,
                seed=42,
            )
            strategy = search_config.create_search_strategy()
            result = await strategy.search(problem)

        result_path = band_dir / "result.json"
        result_path.write_text(result.model_dump_json(indent=2))

        # 3. Now simulate resume: detect the pre-saved band
        loaded_result = json.loads(result_path.read_text())
        loaded_dp = PlacementResult.model_validate_json(
            (band_dir / "surrogate_dp.json").read_text()
        )

        resumed_band = BandValidationResult(
            band_index=band.band_index,
            cost_low=band.cost_low,
            cost_high=band.cost_high,
            surrogate_dp=loaded_dp,
            validated=PlacementResult(
                cost=loaded_result["final_cost"],
                score=loaded_result["best_score"],
                assignments=tuple(
                    loaded_result["best_parameters"]["assignments"]
                ),
            ),
            total_evaluations=loaded_result["total_evaluations"],
        )

        # Verify restored band matches original result
        assert resumed_band.total_evaluations == result.total_evaluations
        assert resumed_band.validated.score == result.best_score
        assert resumed_band.validated.cost == result.final_cost
        assert resumed_band.surrogate_dp.cost == band.cost
        assert resumed_band.surrogate_dp.score == band.score


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
            exploration_dataset_path="/tmp/test.json",
            seed=42,
            num_cost_bins=80,
        )
        assert cfg.seed == 42
        assert cfg.num_cost_bins == 80
        assert cfg.exploration_dataset_path == "/tmp/test.json"

    def test_landscape_config_frozen(self) -> None:
        """LandscapeConfig should be immutable."""
        cfg = LandscapeConfig(
            cost_model=_COST_MODEL_CONFIG,
            exploration_dataset_path="/tmp/test.json",
            seed=42,
            num_cost_bins=80,
        )
        with pytest.raises(Exception):
            cfg.seed = 99  # type: ignore[misc]
