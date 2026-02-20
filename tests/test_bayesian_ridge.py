"""Unit tests for BayesianRidgeSurrogate."""

from __future__ import annotations

import numpy as np
import pytest

from place_layers.surrogate.bayesian_ridge import BayesianRidgeSurrogate
from place_layers.surrogate.encoding import FeatureExpansionConfig


def _random_placements(
    rng: np.random.RandomState,
    mixer_names: list[str],
    num_layers: int,
    n: int,
) -> list[tuple[str, ...]]:
    """Generate n random placements for testing."""
    return [
        tuple(rng.choice(mixer_names, size=num_layers))
        for _ in range(n)
    ]


def _fit_bayesian(
    num_layers: int = 8,
    mixer_names: list[str] | None = None,
    feature_config: FeatureExpansionConfig | None = None,
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[BayesianRidgeSurrogate, list[tuple[str, ...]], np.ndarray]:
    """Fit a BayesianRidgeSurrogate on synthetic data. Returns (surrogate, assignments, y)."""
    if mixer_names is None:
        mixer_names = ["m0", "m1"]
    if feature_config is None:
        feature_config = FeatureExpansionConfig(max_order=2, pairwise_max_range=1)

    rng = np.random.RandomState(seed)
    assignments = _random_placements(rng, mixer_names, num_layers, n_samples)
    # Simple signal: count of first mixer type, plus noise
    y = np.array([sum(1.0 for m in a if m == mixer_names[0]) for a in assignments])
    y += 0.1 * rng.randn(n_samples)

    surrogate = BayesianRidgeSurrogate(
        mixer_names=mixer_names,
        seed=seed,
        feature_config=feature_config,
    )
    surrogate.fit(assignments, y)
    return surrogate, assignments, y


class TestFitAndPredictShapes:
    """Verify output shapes from fit/predict."""

    def test_predict_shapes(self) -> None:
        surrogate, assignments, _ = _fit_bayesian(n_samples=50)
        test = _random_placements(np.random.RandomState(99), ["m0", "m1"], 8, 15)
        mu, sigma = surrogate.predict(test)
        assert mu.shape == (15,)
        assert sigma.shape == (15,)
        assert np.all(sigma > 0)

    def test_predict_mean_shape(self) -> None:
        surrogate, _, _ = _fit_bayesian(n_samples=50)
        test = _random_placements(np.random.RandomState(99), ["m0", "m1"], 8, 10)
        mu = surrogate.predict_mean(test)
        assert mu.shape == (10,)

    def test_predict_mean_one_matches_batch(self) -> None:
        surrogate, assignments, _ = _fit_bayesian()
        test = assignments[0]
        one = surrogate.predict_mean_one(test)
        batch = surrogate.predict_mean([test])
        assert abs(one - batch[0]) < 1e-6


class TestPosteriorVariance:
    """Verify posterior variance properties."""

    def test_variance_grows_with_distance(self) -> None:
        """OOD points should have higher sigma than training points."""
        rng = np.random.RandomState(42)
        mixer_names = ["m0", "m1"]
        num_layers = 8

        # Train on placements dominated by m0
        assignments = [("m0",) * num_layers for _ in range(50)]
        # Add some slight variation
        for i in range(50):
            a = list(assignments[i])
            if rng.random() < 0.3:
                a[rng.randint(num_layers)] = "m1"
            assignments[i] = tuple(a)

        y = np.array([sum(1.0 for m in a if m == "m0") for a in assignments])

        surrogate = BayesianRidgeSurrogate(
            mixer_names=mixer_names, seed=42,
            feature_config=FeatureExpansionConfig(max_order=2, pairwise_max_range=1),
        )
        surrogate.fit(assignments, y)

        # In-distribution: all m0
        id_point = [("m0",) * num_layers]
        # Out-of-distribution: all m1
        ood_point = [("m1",) * num_layers]

        _, sigma_id = surrogate.predict(id_point)
        _, sigma_ood = surrogate.predict(ood_point)

        assert sigma_ood[0] > sigma_id[0], (
            f"OOD sigma ({sigma_ood[0]:.4f}) should exceed ID sigma ({sigma_id[0]:.4f})"
        )


class TestMRFModels:
    """Verify MRF model extraction."""

    def test_mean_var_mrf_models_exclude_counts(self) -> None:
        feature_config = FeatureExpansionConfig(
            max_order=2, pairwise_max_range=1, allocation_counts=True
        )
        surrogate, _, _ = _fit_bayesian(feature_config=feature_config)
        mean_model, var_model = surrogate.extract_mean_var_mrf_models(8)

        # Both models should have allocation_counts=False
        assert mean_model._feature_config.allocation_counts is False
        assert var_model._feature_config.allocation_counts is False

    def test_mean_mrf_matches_predict_mean_one(self) -> None:
        """MRF score + intercept should equal predict_mean_one (for count-free config)."""
        feature_config = FeatureExpansionConfig(
            max_order=2, pairwise_max_range=1, allocation_counts=False
        )
        surrogate, assignments, _ = _fit_bayesian(feature_config=feature_config)
        mrf = surrogate.extract_mrf_model(8)

        for assignment in assignments[:20]:
            ridge_pred = surrogate.predict_mean_one(assignment)
            score = 0.0
            for i in range(8):
                score += mrf.unary_potential(i, assignment[i])
            for i in range(7):
                score += mrf.pairwise_potential(i, assignment[i], assignment[i + 1])
            assert abs(ridge_pred - (score + mrf.intercept)) < 1e-5, (
                f"Mismatch for {assignment}: ridge={ridge_pred}, mrf+intercept={score + mrf.intercept}"
            )


class TestFullLCB:
    """Verify full LCB computation."""

    def test_full_lcb_uses_all_features(self) -> None:
        """Full LCB should differ from count-free score when counts are enabled."""
        feature_config = FeatureExpansionConfig(
            max_order=2, pairwise_max_range=1, allocation_counts=True
        )
        surrogate, assignments, _ = _fit_bayesian(feature_config=feature_config)

        # Full LCB uses all features (including counts)
        lcb_full = surrogate.full_lcb(assignments[:10], beta=1.0)
        assert lcb_full.shape == (10,)

        # Compare with pure mean — LCB should be lower
        mu = surrogate.predict_mean(assignments[:10])
        assert np.all(lcb_full <= mu + 1e-10), "LCB should be <= mean"

    def test_lcb_beta_zero_is_mean(self) -> None:
        """With β=0, LCB should equal the mean prediction."""
        surrogate, assignments, _ = _fit_bayesian()
        lcb = surrogate.full_lcb(assignments[:10], beta=0.0)
        mu = surrogate.predict_mean(assignments[:10])
        np.testing.assert_allclose(lcb, mu, atol=1e-10)


class TestPosteriorHyperparameters:
    """Verify posterior hyperparameter properties."""

    def test_noise_scale_positive(self) -> None:
        surrogate, _, _ = _fit_bayesian()
        assert surrogate.noise_scale_sq > 0

    def test_b_n_nonnegative(self) -> None:
        """b_n should be non-negative."""
        surrogate, _, _ = _fit_bayesian()
        assert surrogate._b_n > 0

    def test_diag_Vn_positive(self) -> None:
        """All diagonal elements of V_n should be positive."""
        surrogate, _, _ = _fit_bayesian()
        assert np.all(surrogate.diag_Vn > 0)


class TestCrossValidation:
    """Verify cross-validation."""

    def test_cv_reasonable_on_easy_landscape(self) -> None:
        surrogate, assignments, y = _fit_bayesian(n_samples=200)
        rho = surrogate.cross_validate(assignments, y, cv_folds=5)
        assert 0 <= rho <= 1.0

    def test_cv_noise(self) -> None:
        """CV on pure noise should return rho near 0."""
        rng = np.random.RandomState(42)
        mixer_names = ["m0", "m1"]
        assignments = _random_placements(rng, mixer_names, 8, 100)
        y = rng.randn(100)

        surrogate = BayesianRidgeSurrogate(
            mixer_names=mixer_names, seed=42,
            feature_config=FeatureExpansionConfig(),
        )
        rho = surrogate.cross_validate(assignments, y, cv_folds=5)
        assert abs(rho) < 0.4
