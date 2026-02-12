"""Unit tests for the surrogate package: encoding, forest, ridge, acquisition."""

import numpy as np
import pytest

from place_layers.surrogate.acquisition import expected_improvement
from place_layers.surrogate.encoding import (
    FeatureExpansionConfig,
    build_type_mapping,
    compute_feature_dim,
    decode_config,
    encode_batch,
    encode_config,
    feature_names,
)
from place_layers.surrogate.forest_surrogate import ForestSurrogate
from place_layers.surrogate.ridge_surrogate import RidgeSurrogate


# =============================================================================
# Encoding tests
# =============================================================================


class TestEncoding:
    """Tests for one-hot + allocation-count encoding."""

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
        """Encoded vector should have shape (num_layers * num_types + num_types,)."""
        type_to_int, _ = type_mapping
        assignment = ("attention", "gdn", "gdn", "sliding_window", "attention")
        features = encode_config(assignment, type_to_int)
        # 5 layers * 3 types + 3 types = 18 features
        assert features.shape == (18,)
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
        counts = features[num_layers * num_types : num_layers * num_types + num_types]
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
        assert X.shape == (2, 12)  # 3 layers * 3 types + 3 types

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
        assert len(names) == 18  # 5 layers * 3 types + 3 types
        assert names[0] == "layer_0_attention"
        assert names[1] == "layer_0_gdn"
        assert names[2] == "layer_0_sliding_window"
        assert names[15] == "count_attention"
        assert names[17] == "count_sliding_window"


# =============================================================================
# Cluster expansion encoding tests
# =============================================================================


class TestClusterExpansion:
    """Tests for pairwise and higher-order feature expansion."""

    @pytest.fixture
    def mixer_names(self) -> list[str]:
        return ["a", "b"]

    @pytest.fixture
    def type_mapping(
        self, mixer_names: list[str]
    ) -> tuple[dict[str, int], dict[int, str]]:
        return build_type_mapping(mixer_names)

    def test_default_config_backward_compatible(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """None config should produce same features as default config."""
        type_to_int, _ = type_mapping
        assignment = ("a", "b", "a", "b")
        f_none = encode_config(assignment, type_to_int, feature_config=None)
        f_default = encode_config(
            assignment, type_to_int,
            feature_config=FeatureExpansionConfig(),
        )
        np.testing.assert_array_equal(f_none, f_default)

    def test_pairwise_feature_dimensions(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """Verify shape for max_order=2 at various ranges."""
        type_to_int, _ = type_mapping
        N, K = 6, 2
        assignment = ("a", "b", "a", "b", "a", "b")

        # Range 1: N*K + (N-1)*K^2 + K = 12 + 20 + 2 = 34
        cfg_r1 = FeatureExpansionConfig(max_order=2, pairwise_max_range=1)
        f = encode_config(assignment, type_to_int, cfg_r1)
        assert f.shape == (compute_feature_dim(N, K, cfg_r1),)
        assert f.shape == (34,)

        # Range 2: 12 + 20 + 16 + 2 = 50
        cfg_r2 = FeatureExpansionConfig(max_order=2, pairwise_max_range=2)
        f2 = encode_config(assignment, type_to_int, cfg_r2)
        expected = N * K + (N - 1) * K * K + (N - 2) * K * K + K
        assert f2.shape == (expected,)

        # Range 3: 12 + 20 + 16 + 12 + 2 = 62
        cfg_r3 = FeatureExpansionConfig(max_order=2, pairwise_max_range=3)
        f3 = encode_config(assignment, type_to_int, cfg_r3)
        expected3 = N * K + (N - 1) * K * K + (N - 2) * K * K + (N - 3) * K * K + K
        assert f3.shape == (expected3,)

    def test_pairwise_feature_values(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """Verify correct indicator is set for adjacent pair."""
        type_to_int, _ = type_mapping
        # a=0, b=1 (sorted)
        assignment = ("a", "b", "a")
        cfg = FeatureExpansionConfig(max_order=2, pairwise_max_range=1)
        f = encode_config(assignment, type_to_int, cfg)

        N, K = 3, 2
        # Unary block: N*K = 6 features
        # Pair block for r=1: 2 pairs, each K^2=4 indicators
        unary_end = N * K  # 6
        # Pair (0,1): types = (a=0, b=1) -> flat_idx = 0*2 + 1 = 1
        pair_0_start = unary_end
        assert f[pair_0_start + 1] == 1.0  # (a, b)
        assert f[pair_0_start + 0] == 0.0  # (a, a)
        # Pair (1,2): types = (b=1, a=0) -> flat_idx = 1*2 + 0 = 2
        pair_1_start = pair_0_start + K * K
        assert f[pair_1_start + 2] == 1.0  # (b, a)
        assert f[pair_1_start + 3] == 0.0  # (b, b)

    def test_higher_order_dimensions(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """Verify shape for max_order=3 and max_order=4."""
        type_to_int, _ = type_mapping
        N, K = 6, 2
        assignment = ("a", "b", "a", "b", "a", "b")

        # max_order=3, range=1: unary + pairs(r=1) + triplets + counts
        # 12 + 20 + (N-2)*K^3 + 2 = 12 + 20 + 32 + 2 = 66
        cfg3 = FeatureExpansionConfig(max_order=3, pairwise_max_range=1)
        f3 = encode_config(assignment, type_to_int, cfg3)
        expected3 = N * K + (N - 1) * K * K + (N - 2) * (K ** 3) + K
        assert f3.shape == (expected3,)
        assert f3.shape == (66,)

        # max_order=4, range=1: + (N-3)*K^4
        # 66 + (N-3)*K^4 = 66 + 48 = 114
        cfg4 = FeatureExpansionConfig(max_order=4, pairwise_max_range=1)
        f4 = encode_config(assignment, type_to_int, cfg4)
        expected4 = expected3 + (N - 3) * (K ** 4)
        assert f4.shape == (expected4,)

    def test_higher_order_feature_values(
        self,
        type_mapping: tuple[dict[str, int], dict[int, str]],
    ) -> None:
        """Verify correct indicator for a triple."""
        type_to_int, _ = type_mapping
        # a=0, b=1
        assignment = ("a", "b", "a")
        cfg = FeatureExpansionConfig(max_order=3, pairwise_max_range=1)
        f = encode_config(assignment, type_to_int, cfg)

        N, K = 3, 2
        # Layout: unary(6) + pairs_r1(2 * 4 = 8) + triplets(1 * 8) + counts(2)
        triplet_start = N * K + (N - 1) * K * K  # 6 + 8 = 14
        # Triple (0,1,2): types = (0,1,0) -> flat_idx = 0*4 + 1*2 + 0 = 2
        assert f[triplet_start + 2] == 1.0
        # All other triplet indicators should be 0
        for i in range(K ** 3):
            if i != 2:
                assert f[triplet_start + i] == 0.0

    def test_feature_names_with_interactions(self) -> None:
        """Check naming pattern for pairwise and higher-order features."""
        _, int_to_type = build_type_mapping(["a", "b"])
        cfg = FeatureExpansionConfig(max_order=3, pairwise_max_range=1)
        names = feature_names(num_layers=3, int_to_type=int_to_type, feature_config=cfg)

        # Unary: 6 names
        assert names[0] == "layer_0_a"
        assert names[5] == "layer_2_b"

        # Pairwise (r=1): 2 pairs * 4 indicators = 8 names
        assert names[6] == "pair_0_1_a_a"
        assert names[7] == "pair_0_1_a_b"
        assert names[8] == "pair_0_1_b_a"
        assert names[9] == "pair_0_1_b_b"
        assert names[10] == "pair_1_2_a_a"

        # Triplets: 1 tuple * 8 indicators
        assert names[14] == "cluster3_0_a_a_a"
        assert names[15] == "cluster3_0_a_a_b"

        # Counts at end
        assert names[-2] == "count_a"
        assert names[-1] == "count_b"

        # Total: 6 + 8 + 8 + 2 = 24
        assert len(names) == 24

    def test_compute_feature_dim(self) -> None:
        """compute_feature_dim should match actual encoded shape at each order."""
        type_to_int, _ = build_type_mapping(["a", "b", "c"])
        N, K = 8, 3
        assignment = tuple(["a", "b", "c"] * 2 + ["a", "b"])

        for max_order in [1, 2, 3]:
            for pairwise_range in [1, 2, 3]:
                cfg = FeatureExpansionConfig(
                    max_order=max_order,
                    pairwise_max_range=pairwise_range,
                )
                expected_dim = compute_feature_dim(N, K, cfg)
                actual = encode_config(assignment, type_to_int, cfg)
                assert actual.shape == (expected_dim,), (
                    f"Mismatch for max_order={max_order}, range={pairwise_range}: "
                    f"expected {expected_dim}, got {actual.shape[0]}"
                )

    def test_encode_roundtrip_with_interactions(self) -> None:
        """Encode->decode still works with expanded features (decode ignores interactions)."""
        type_to_int, int_to_type = build_type_mapping(["a", "b", "c"])
        original = ("a", "b", "c", "a", "b")
        cfg = FeatureExpansionConfig(max_order=3, pairwise_max_range=2)
        features = encode_config(original, type_to_int, cfg)
        decoded = decode_config(features, int_to_type, num_layers=5)
        assert decoded == original


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
        # 8 layers * 3 types + 3 types = 27 features
        assert importances.shape == (27,)
        assert abs(importances.sum() - 1.0) < 1e-6

    def test_feature_importances_with_expansion(self) -> None:
        """Feature importances with expansion should have correct shape."""
        mixer_names = ["a", "b"]
        num_layers = 6
        rng = np.random.RandomState(42)

        assignments = _random_placements(rng, mixer_names, num_layers, 100)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        cfg = FeatureExpansionConfig(max_order=2, pairwise_max_range=1)
        surrogate = ForestSurrogate(
            n_estimators=50, max_depth=None, seed=42,
            mixer_names=mixer_names, feature_config=cfg,
        )
        surrogate.fit(assignments, y)

        importances = surrogate.feature_importances()
        expected_dim = compute_feature_dim(num_layers, 2, cfg)
        assert importances.shape == (expected_dim,)
        assert abs(importances.sum() - 1.0) < 1e-6

    def test_feature_names(self) -> None:
        """feature_names should return correct names."""
        mixer_names = ["attention", "gdn"]
        surrogate = ForestSurrogate(
            n_estimators=10, max_depth=None, seed=42, mixer_names=mixer_names
        )
        names = surrogate.feature_names(num_layers=3)
        assert names == [
            "layer_0_attention", "layer_0_gdn",
            "layer_1_attention", "layer_1_gdn",
            "layer_2_attention", "layer_2_gdn",
            "count_attention", "count_gdn",
        ]

    def test_feature_names_with_expansion(self) -> None:
        """feature_names with expansion config should include interaction names."""
        mixer_names = ["a", "b"]
        cfg = FeatureExpansionConfig(max_order=2, pairwise_max_range=1)
        surrogate = ForestSurrogate(
            n_estimators=10, max_depth=None, seed=42,
            mixer_names=mixer_names, feature_config=cfg,
        )
        names = surrogate.feature_names(num_layers=3)
        # Should include pairwise names
        assert any("pair_" in n for n in names)
        # Total: 6 + 8 + 2 = 16
        assert len(names) == compute_feature_dim(3, 2, cfg)

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

    def test_min_samples_leaf_default(self) -> None:
        """Default min_samples_leaf should be 1."""
        mixer_names = ["a", "b"]
        surrogate = ForestSurrogate(
            n_estimators=10, max_depth=None, seed=42, mixer_names=mixer_names
        )
        assert surrogate._min_samples_leaf == 1
        assert surrogate._model.min_samples_leaf == 1

    def test_min_samples_leaf_passthrough(self) -> None:
        """Explicit min_samples_leaf should be passed to the RF model."""
        mixer_names = ["a", "b"]
        num_layers = 5
        rng = np.random.RandomState(42)

        surrogate = ForestSurrogate(
            n_estimators=20, max_depth=None, seed=42,
            mixer_names=mixer_names, min_samples_leaf=10,
        )
        assert surrogate._min_samples_leaf == 10
        assert surrogate._model.min_samples_leaf == 10

        # Fit and cross-validate to ensure it propagates to CV folds
        assignments = _random_placements(rng, mixer_names, num_layers, 100)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])
        rho = surrogate.cross_validate(assignments, y, cv_folds=3)
        assert isinstance(rho, float)

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


# =============================================================================
# RidgeSurrogate tests
# =============================================================================


class TestRidgeSurrogate:
    """Tests for Ridge + Bootstrap surrogate wrapper."""

    def test_fit_predict_shapes(self) -> None:
        """Fit on synthetic data, predict returns correct shapes."""
        mixer_names = ["a", "b", "c"]
        num_layers = 10
        rng = np.random.RandomState(42)

        assignments = _random_placements(rng, mixer_names, num_layers, 50)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        surrogate = RidgeSurrogate(
            mixer_names=mixer_names, seed=42, n_bootstraps=10,
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
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        surrogate = RidgeSurrogate(
            mixer_names=mixer_names, seed=42, n_bootstraps=20,
        )
        surrogate.fit(assignments, y)

        mean, _ = surrogate.predict(assignments[:20])
        errors = np.abs(mean - y[:20])
        assert np.mean(errors) < np.std(y)

    def test_predict_mean_one(self) -> None:
        """predict_mean_one should match predict_mean for a single assignment."""
        mixer_names = ["a", "b"]
        num_layers = 5
        rng = np.random.RandomState(42)

        assignments = _random_placements(rng, mixer_names, num_layers, 100)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])

        surrogate = RidgeSurrogate(
            mixer_names=mixer_names, seed=42, n_bootstraps=10,
        )
        surrogate.fit(assignments, y)

        test = assignments[0]
        one = surrogate.predict_mean_one(test)
        batch = surrogate.predict_mean([test])
        assert abs(one - batch[0]) < 1e-6

    def test_cross_validate_easy_landscape(self) -> None:
        """CV on an easy landscape should return high rho."""
        mixer_names = ["a", "b"]
        num_layers = 5
        rng = np.random.RandomState(42)

        assignments = _random_placements(rng, mixer_names, num_layers, 200)
        y = np.array([sum(1.0 for m in a if m == "a") for a in assignments])
        y += 0.1 * rng.randn(200)

        surrogate = RidgeSurrogate(
            mixer_names=mixer_names, seed=42, n_bootstraps=10,
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

        surrogate = RidgeSurrogate(
            mixer_names=mixer_names, seed=42, n_bootstraps=10,
        )
        rho = surrogate.cross_validate(assignments, y, cv_folds=5)
        assert abs(rho) < 0.4
