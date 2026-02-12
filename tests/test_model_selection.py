"""Unit tests for RF model selection (regularization grid search)."""

import numpy as np
import pytest

from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.surrogate.model_selection import (
    REGULARIZATION_GRID,
    ExpansionSelectionResult,
    LevelResult,
    ModelSelectionResult,
    RFParams,
    ridge_cv_rho,
    select_expansion_level,
    select_rf_params,
)


# =============================================================================
# RFParams tests
# =============================================================================


class TestRFParams:
    def test_frozen(self) -> None:
        params = RFParams(n_estimators=100, max_depth=8, min_samples_leaf=3)
        with pytest.raises(Exception):
            params.n_estimators = 200  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        params = RFParams(n_estimators=100, max_depth=8, min_samples_leaf=3)
        json_str = params.model_dump_json()
        loaded = RFParams.model_validate_json(json_str)
        assert loaded == params

    def test_none_max_depth(self) -> None:
        params = RFParams(n_estimators=50, max_depth=None, min_samples_leaf=1)
        assert params.max_depth is None
        json_str = params.model_dump_json()
        loaded = RFParams.model_validate_json(json_str)
        assert loaded.max_depth is None


# =============================================================================
# REGULARIZATION_GRID tests
# =============================================================================


class TestRegularizationGrid:
    def test_grid_length(self) -> None:
        assert len(REGULARIZATION_GRID) == 10

    def test_required_keys(self) -> None:
        for entry in REGULARIZATION_GRID:
            assert "max_depth" in entry
            assert "min_samples_leaf" in entry


# =============================================================================
# Helpers
# =============================================================================


def _random_placements(
    rng: np.random.RandomState,
    mixer_names: list[str],
    num_layers: int,
    n: int,
) -> list[tuple[str, ...]]:
    return [
        tuple(rng.choice(mixer_names, size=num_layers))
        for _ in range(n)
    ]


# =============================================================================
# select_rf_params tests
# =============================================================================


class TestSelectRFParams:
    def test_smoke_returns_correct_type(self) -> None:
        """Basic smoke test: returns ModelSelectionResult."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=5, n=60)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        result = select_rf_params(
            assignments=assignments,
            y=y,
            n_estimators=20,
            cv_folds=3,
            seed=42,
            mixer_names=mixer_names,
        )
        assert isinstance(result, ModelSelectionResult)
        assert isinstance(result.best_params, RFParams)
        assert result.best_params.n_estimators == 20
        assert result.test_rho > -1.0
        assert result.train_rho >= result.test_rho - 0.01  # train >= test (roughly)
        assert result.gap >= -0.01

    def test_easy_landscape_selects_light_regularization(self) -> None:
        """On an easy landscape, model selection should pick a config with good test_rho."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=5, n=200)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])
        y += 0.05 * rng.randn(200)  # Tiny noise

        result = select_rf_params(
            assignments=assignments,
            y=y,
            n_estimators=50,
            cv_folds=5,
            seed=42,
            mixer_names=mixer_names,
        )
        # Easy landscape → high test rho
        assert result.test_rho > 0.5

    def test_noisy_data_selects_heavier_regularization(self) -> None:
        """On noisy data with small n, selection should pick more regularized config."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b", "c"]
        assignments = _random_placements(rng, mixer_names, num_layers=5, n=40)
        # Weak signal + strong noise
        y = np.array([sum(0.1 for m in a if m == "a") for a in assignments])
        y += 2.0 * rng.randn(40)

        result = select_rf_params(
            assignments=assignments,
            y=y,
            n_estimators=20,
            cv_folds=3,
            seed=42,
            mixer_names=mixer_names,
        )
        # With heavy noise, best config should have *some* regularization
        # (min_samples_leaf > 1 or limited max_depth)
        p = result.best_params
        regularized = p.min_samples_leaf > 1 or (
            p.max_depth is not None and p.max_depth < 20
        )
        assert regularized, (
            f"Expected regularized config for noisy data, got "
            f"max_depth={p.max_depth}, min_samples_leaf={p.min_samples_leaf}"
        )

    def test_top_region_rho_none_for_tiny_data(self) -> None:
        """top_region_rho should be None when top 5% < 2 samples."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        # 20 samples → top 5% = 1 sample → None
        assignments = _random_placements(rng, mixer_names, num_layers=5, n=20)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        result = select_rf_params(
            assignments=assignments,
            y=y,
            n_estimators=20,
            cv_folds=3,
            seed=42,
            mixer_names=mixer_names,
        )
        assert result.top_region_rho is None

    def test_top_region_rho_computed_for_large_data(self) -> None:
        """top_region_rho should be a float when there are enough samples."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        # 200 samples → top 5% = 10 samples → computed
        assignments = _random_placements(rng, mixer_names, num_layers=5, n=200)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        result = select_rf_params(
            assignments=assignments,
            y=y,
            n_estimators=50,
            cv_folds=5,
            seed=42,
            mixer_names=mixer_names,
        )
        assert result.top_region_rho is not None
        assert isinstance(result.top_region_rho, float)

    def test_max_gap_filters_overfitting_configs(self) -> None:
        """With strict max_gap, selected config should have acceptable gap."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b", "c"]
        # Small dataset with noise → likely overfit for unrestricted configs
        assignments = _random_placements(rng, mixer_names, num_layers=5, n=40)
        y = np.array([sum(0.5 for m in a if m == "a") for a in assignments])
        y += 0.5 * rng.randn(40)

        # Without max_gap
        result_no_gap = select_rf_params(
            assignments=assignments,
            y=y,
            n_estimators=20,
            cv_folds=3,
            seed=42,
            mixer_names=mixer_names,
        )
        # With strict max_gap
        result_with_gap = select_rf_params(
            assignments=assignments,
            y=y,
            n_estimators=20,
            cv_folds=3,
            seed=42,
            mixer_names=mixer_names,
            max_gap=0.05,
        )
        # The gap-constrained result should have gap <= max_gap if any
        # config qualifies, or at worst fall back gracefully
        assert isinstance(result_with_gap, ModelSelectionResult)
        # If unconstrained had larger gap, constrained should be <= it
        if result_no_gap.gap > 0.05:
            assert result_with_gap.gap <= result_no_gap.gap

    def test_max_gap_fallback_when_all_exceed(self) -> None:
        """With impossibly strict max_gap, should still return a valid result."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=5, n=60)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        # max_gap=0.0 is impossibly strict (any model will have some gap)
        result = select_rf_params(
            assignments=assignments,
            y=y,
            n_estimators=20,
            cv_folds=3,
            seed=42,
            mixer_names=mixer_names,
            max_gap=0.0,
        )
        # Should fall back to all configs and still return a valid result
        assert isinstance(result, ModelSelectionResult)
        assert isinstance(result.best_params, RFParams)


# =============================================================================
# select_expansion_level tests
# =============================================================================


class TestSelectExpansionLevel:
    def test_easy_landscape_with_pairwise_signal(self) -> None:
        """On MRF-like data with pairwise signal, level 1 should produce
        dramatically higher rho than level 0."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        num_layers = 12
        n_samples = 200

        assignments = _random_placements(rng, mixer_names, num_layers, n_samples)

        # Build MRF-like scores: unary + pairwise
        scores = np.zeros(n_samples)
        for idx, a in enumerate(assignments):
            # Unary: +1 for 'a', -0.5 for 'b'
            for layer_type in a:
                if layer_type == "a":
                    scores[idx] += 1.0
                else:
                    scores[idx] -= 0.5
            # Pairwise: +0.3 for same adjacent, -0.3 for different
            for i in range(len(a) - 1):
                if a[i] == a[i + 1]:
                    scores[idx] += 0.3
                else:
                    scores[idx] -= 0.3

        result = select_expansion_level(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            cv_folds=5,
            target_rho=0.99,
            min_improvement=0.005,
            max_order=3,
            seed=42,
        )

        assert isinstance(result, ExpansionSelectionResult)
        assert len(result.level_results) >= 2

        # Level 0 (unary) should have lower rho than level 1 (pairwise)
        level_0_rho = result.level_results[0].rho
        level_1_rho = result.level_results[1].rho
        assert level_1_rho > level_0_rho + 0.05, (
            f"Expected pairwise level to improve significantly: "
            f"level_0={level_0_rho:.4f}, level_1={level_1_rho:.4f}"
        )

        # Selected level should be >= 1 since pairwise features help
        assert result.selected_level >= 1

    def test_unary_only_landscape(self) -> None:
        """On unary-only landscape, should select level 0."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        num_layers = 12
        n_samples = 200

        assignments = _random_placements(rng, mixer_names, num_layers, n_samples)

        # Purely unary scores: just count of 'a'
        scores = np.array([
            sum(1.0 for m in a if m == "a") for a in assignments
        ], dtype=np.float64)

        result = select_expansion_level(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            cv_folds=5,
            target_rho=0.99,
            min_improvement=0.01,
            max_order=3,
            seed=42,
        )

        assert isinstance(result, ExpansionSelectionResult)
        # Level 0 should already have very high rho for this linear signal
        level_0_rho = result.level_results[0].rho
        assert level_0_rho > 0.9, (
            f"Expected high rho for linear signal: level_0={level_0_rho:.4f}"
        )
        # Should select level 0 since it already meets target or doesn't improve
        assert result.selected_level == 0

    def test_level_result_types(self) -> None:
        """LevelResult and ExpansionSelectionResult should be well-formed."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=6, n=60)
        scores = rng.randn(60)

        result = select_expansion_level(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            cv_folds=3,
            max_order=2,
            seed=42,
        )

        assert isinstance(result.selected_config, FeatureExpansionConfig)
        for lr in result.level_results:
            assert isinstance(lr, LevelResult)
            assert isinstance(lr.config, FeatureExpansionConfig)
            assert lr.num_features > 0
            assert isinstance(lr.rho, float)

    def test_serialization_roundtrip(self) -> None:
        """ExpansionSelectionResult should serialize and deserialize."""
        cfg = FeatureExpansionConfig(max_order=2, pairwise_max_range=1)
        result = ExpansionSelectionResult(
            selected_config=cfg,
            selected_level=1,
            level_results=[
                LevelResult(
                    level=0,
                    config=FeatureExpansionConfig(max_order=1),
                    num_features=10,
                    rho=0.5,
                ),
                LevelResult(
                    level=1,
                    config=cfg,
                    num_features=30,
                    rho=0.9,
                ),
            ],
        )
        json_str = result.model_dump_json()
        loaded = ExpansionSelectionResult.model_validate_json(json_str)
        assert loaded.selected_level == 1
        assert loaded.selected_config == cfg
        assert len(loaded.level_results) == 2


# =============================================================================
# ridge_cv_rho tests
# =============================================================================


class TestRidgeCvRho:
    def test_easy_landscape_high_rho(self) -> None:
        """Ridge CV rho should be high on an easy landscape."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=5, n=200)
        scores = np.array([sum(1.0 for m in a if m == "a") for a in assignments])
        scores += 0.05 * rng.randn(200)

        rho = ridge_cv_rho(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            cv_folds=5,
            seed=42,
        )
        assert rho > 0.8

    def test_with_feature_config(self) -> None:
        """Ridge CV rho should work with expanded features."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=8, n=200)
        # MRF-like scores with pairwise signal
        scores = np.zeros(200)
        for idx, a in enumerate(assignments):
            for m in a:
                scores[idx] += 1.0 if m == "a" else -0.5
            for i in range(len(a) - 1):
                scores[idx] += 0.3 if a[i] == a[i + 1] else -0.3

        cfg = FeatureExpansionConfig(max_order=2, pairwise_max_range=1)
        rho = ridge_cv_rho(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            feature_config=cfg,
            cv_folds=5,
            seed=42,
        )
        assert rho > 0.8

    def test_noise_low_rho(self) -> None:
        """Ridge CV rho on pure noise should be near 0."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b", "c"]
        assignments = _random_placements(rng, mixer_names, num_layers=5, n=100)
        scores = rng.randn(100)

        rho = ridge_cv_rho(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            cv_folds=5,
            seed=42,
        )
        assert abs(rho) < 0.4
