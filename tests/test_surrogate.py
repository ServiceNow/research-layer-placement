"""Unit tests for the surrogate package: encoding, forest, acquisition."""

import numpy as np
import pytest

from place_layers.surrogate.acquisition import expected_improvement
from place_layers.surrogate.encoding import (
    build_type_mapping,
    decode_config,
    encode_batch,
    encode_config,
    feature_names,
)
from place_layers.surrogate.forest_surrogate import ForestSurrogate


# =============================================================================
# Encoding tests
# =============================================================================


class TestEncoding:
    """Tests for ordinal + allocation-count encoding."""

    @pytest.fixture
    def mixer_names(self) -> list[str]:
        return ["attention", "gdn", "sliding_window"]

    @pytest.fixture
    def type_mapping(
        self, mixer_names: list[str]
    ) -> tuple[dict[str, int], dict[int, str]]:
        return build_type_mapping(mixer_names)

    def test_build_type_mapping_sorted(self, mixer_names: list[str]) -> None:
        """Type mapping should be sorted alphabetically."""
        type_to_int, int_to_type = build_type_mapping(mixer_names)
        assert type_to_int == {"attention": 0, "gdn": 1, "sliding_window": 2}
        assert int_to_type == {0: "attention", 1: "gdn", 2: "sliding_window"}

    def test_encode_dimensions(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """Encoded vector should have shape (num_layers + num_types,)."""
        type_to_int, _ = type_mapping
        assignment = ("attention", "gdn", "gdn", "sliding_window", "attention")
        features = encode_config(assignment, type_to_int)
        # 5 layers + 3 types = 8 features
        assert features.shape == (8,)
        assert features.dtype == np.float32

    def test_encode_allocation_counts(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """Count features should sum to num_layers."""
        type_to_int, _ = type_mapping
        assignment = ("attention", "gdn", "gdn", "sliding_window", "attention")
        features = encode_config(assignment, type_to_int)
        num_layers = 5
        num_types = 3
        counts = features[num_layers : num_layers + num_types]
        assert counts.sum() == num_layers
        # attention=2, gdn=2, sliding_window=1
        assert counts[0] == 2  # attention
        assert counts[1] == 2  # gdn
        assert counts[2] == 1  # sliding_window

    def test_encode_roundtrip(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """Encode then decode should recover original placement."""
        type_to_int, int_to_type = type_mapping
        original = ("attention", "gdn", "gdn", "sliding_window", "attention")
        features = encode_config(original, type_to_int)
        decoded = decode_config(features, int_to_type, num_layers=5)
        assert decoded == original

    def test_encode_batch_shape(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """Batch encoding should return (n, d) matrix."""
        type_to_int, _ = type_mapping
        configs = [
            ("attention", "gdn", "sliding_window"),
            ("gdn", "gdn", "gdn"),
        ]
        X = encode_batch(configs, type_to_int)
        assert X.shape == (2, 6)  # 3 layers + 3 types

    def test_encode_batch_empty(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """Empty batch should return empty matrix."""
        type_to_int, _ = type_mapping
        X = encode_batch([], type_to_int)
        assert X.shape[0] == 0

    def test_feature_names_length(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """Feature names should have correct count."""
        _, int_to_type = type_mapping
        names = feature_names(num_layers=5, int_to_type=int_to_type)
        assert len(names) == 8  # 5 layers + 3 types
        assert names[0] == "layer_0"
        assert names[4] == "layer_4"
        assert names[5] == "count_attention"
        assert names[7] == "count_sliding_window"


# =============================================================================
# Expected Improvement tests
# =============================================================================


class TestExpectedImprovement:
    """Tests for EI acquisition function."""

    def test_ei_zero_sigma(self) -> None:
        """EI with sigma=0 should return max(mu - f_best, 0)."""
        mu = np.array([1.0, 2.0, 0.5])
        sigma = np.array([0.0, 0.0, 0.0])
        f_best = 1.0
        ei = expected_improvement(mu, sigma, f_best, xi=0.0)
        np.testing.assert_array_almost_equal(ei, [0.0, 1.0, 0.0])

    def test_ei_below_best(self) -> None:
        """EI should be approximately 0 when mu << f_best and sigma is small."""
        mu = np.array([0.0, 0.1])
        sigma = np.array([0.01, 0.01])
        f_best = 10.0
        ei = expected_improvement(mu, sigma, f_best)
        assert np.all(ei < 1e-6)

    def test_ei_high_uncertainty(self) -> None:
        """EI > 0 even when mu < f_best if sigma is large."""
        mu = np.array([0.5])
        sigma = np.array([5.0])
        f_best = 1.0
        ei = expected_improvement(mu, sigma, f_best, xi=0.0)
        assert ei[0] > 0.0

    def test_ei_nonnegative(self) -> None:
        """EI should always be non-negative."""
        rng = np.random.RandomState(42)
        mu = rng.randn(100)
        sigma = np.abs(rng.randn(100))
        f_best = 0.0
        ei = expected_improvement(mu, sigma, f_best)
        assert np.all(ei >= -1e-12)

    def test_ei_higher_for_better_mu(self) -> None:
        """Higher mu with same sigma should give higher EI."""
        mu = np.array([0.5, 1.5])
        sigma = np.array([0.5, 0.5])
        f_best = 0.0
        ei = expected_improvement(mu, sigma, f_best, xi=0.0)
        assert ei[1] > ei[0]


# =============================================================================
# ForestSurrogate tests
# =============================================================================


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


class TestForestSurrogate:
    """Tests for RF surrogate wrapper."""

    def test_fit_predict_shapes(self) -> None:
        """Fit on synthetic data, predict returns correct shapes."""
        mixer_names = ["a", "b", "c"]
        num_layers = 10
        rng = np.random.RandomState(42)

        assignments = _random_placements(rng, mixer_names, num_layers, 50)
        # Simple target: count of "a" in placement
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        surrogate = ForestSurrogate(
            n_estimators=20, max_depth=None, seed=42, mixer_names=mixer_names
        )
        surrogate.fit(assignments, y)

        test_assignments = _random_placements(rng, mixer_names, num_layers, 15)
        mean, std = surrogate.predict(test_assignments)

        assert mean.shape == (15,)
        assert std.shape == (15,)
        assert np.all(std >= 0)

    def test_fit_predict_reasonable(self) -> None:
        """Predictions should be reasonable on an easy landscape."""
        mixer_names = ["a", "b"]
        num_layers = 5
        rng = np.random.RandomState(42)

        assignments = _random_placements(rng, mixer_names, num_layers, 200)
        # Target: count of "a" (easy to learn)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        surrogate = ForestSurrogate(
            n_estimators=50, max_depth=None, seed=42, mixer_names=mixer_names
        )
        surrogate.fit(assignments, y)

        mean, std = surrogate.predict(assignments[:20])
        errors = np.abs(mean - y[:20])
        assert np.mean(errors) < np.std(y)

    def test_predict_mean_one(self) -> None:
        """predict_mean_one should match predict_mean for a single assignment."""
        mixer_names = ["a", "b"]
        num_layers = 5
        rng = np.random.RandomState(42)

        assignments = _random_placements(rng, mixer_names, num_layers, 100)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        surrogate = ForestSurrogate(
            n_estimators=20, max_depth=None, seed=42, mixer_names=mixer_names
        )
        surrogate.fit(assignments, y)

        test = assignments[0]
        one = surrogate.predict_mean_one(test)
        batch = surrogate.predict_mean([test])
        assert abs(one - batch[0]) < 1e-10

    def test_feature_importances(self) -> None:
        """Feature importances should sum to approximately 1."""
        mixer_names = ["a", "b", "c"]
        num_layers = 8
        rng = np.random.RandomState(42)

        assignments = _random_placements(rng, mixer_names, num_layers, 100)
        # Target: count of "a" — feature importances should peak at "a"-related features
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        surrogate = ForestSurrogate(
            n_estimators=50, max_depth=None, seed=42, mixer_names=mixer_names
        )
        surrogate.fit(assignments, y)

        importances = surrogate.feature_importances()
        # 8 layers + 3 types = 11 features
        assert importances.shape == (11,)
        assert abs(importances.sum() - 1.0) < 1e-6

    def test_feature_names(self) -> None:
        """feature_names should return correct names."""
        mixer_names = ["attention", "gdn"]
        surrogate = ForestSurrogate(
            n_estimators=10, max_depth=None, seed=42, mixer_names=mixer_names
        )
        names = surrogate.feature_names(num_layers=3)
        assert names == [
            "layer_0", "layer_1", "layer_2",
            "count_attention", "count_gdn",
        ]

    def test_cross_validate_easy_landscape(self) -> None:
        """CV on an easy landscape should return high rho."""
        mixer_names = ["a", "b"]
        num_layers = 5
        rng = np.random.RandomState(42)

        assignments = _random_placements(rng, mixer_names, num_layers, 200)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])
        # Add small noise
        y += 0.1 * rng.randn(200)

        surrogate = ForestSurrogate(
            n_estimators=50, max_depth=None, seed=42, mixer_names=mixer_names
        )
        rho = surrogate.cross_validate(assignments, y, cv_folds=5)
        assert rho > 0.5

    def test_cross_validate_noise(self) -> None:
        """CV on pure noise should return rho near 0."""
        mixer_names = ["a", "b", "c"]
        num_layers = 5
        rng = np.random.RandomState(42)

        assignments = _random_placements(rng, mixer_names, num_layers, 100)
        y = rng.randn(100)

        surrogate = ForestSurrogate(
            n_estimators=20, max_depth=None, seed=42, mixer_names=mixer_names
        )
        rho = surrogate.cross_validate(assignments, y, cv_folds=5)
        assert abs(rho) < 0.4  # Should be near 0
