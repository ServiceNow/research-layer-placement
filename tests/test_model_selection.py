"""Unit tests for surrogate model selection (expansion level selection)."""

import numpy as np

from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.surrogate.model_selection import (
    ExpansionSelectionResult,
    StepResult,
    ridge_cv_rho,
    select_expansion_level,
)


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
# select_expansion_level tests
# =============================================================================


class TestSelectExpansionLevel:
    def test_easy_landscape_with_pairwise_signal(self) -> None:
        """On MRF-like data with pairwise signal, range search should find
        that pairwise features improve significantly over unary baseline."""
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
            max_pairwise_range=5,
            max_order=3,
            seed=42,
        )

        assert isinstance(result, ExpansionSelectionResult)
        # range_results[0] is unary baseline, range_results[1+] are pairwise
        assert len(result.range_results) >= 2

        # Unary baseline should have lower rho than first pairwise step
        unary_rho = result.range_results[0].rho
        pairwise_rho = result.range_results[1].rho
        assert pairwise_rho > unary_rho + 0.05, (
            f"Expected pairwise to improve significantly: "
            f"unary={unary_rho:.4f}, pairwise={pairwise_rho:.4f}"
        )

        # Selected range should be >= 1 since pairwise features help
        assert result.selected_range >= 1

    def test_unary_only_landscape(self) -> None:
        """On unary-only landscape, should select order 1 or small range."""
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
            max_pairwise_range=5,
            max_order=3,
            seed=42,
        )

        assert isinstance(result, ExpansionSelectionResult)
        # Unary baseline should already have very high rho for this linear signal
        unary_rho = result.range_results[0].rho
        assert unary_rho > 0.9, (
            f"Expected high rho for linear signal: unary={unary_rho:.4f}"
        )
        # Should select order <= 2 (likely 1 since unary captures the signal)
        assert result.selected_order <= 2

    def test_level_result_types(self) -> None:
        """StepResult and ExpansionSelectionResult should be well-formed."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=6, n=60)
        scores = rng.randn(60)

        result = select_expansion_level(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            cv_folds=3,
            max_pairwise_range=3,
            max_order=2,
            seed=42,
        )

        assert isinstance(result.selected_config, FeatureExpansionConfig)
        for step in result.range_results:
            assert isinstance(step, StepResult)
            assert isinstance(step.config, FeatureExpansionConfig)
            assert step.num_features > 0
            assert isinstance(step.rho, float)
        for step in result.order_results:
            assert isinstance(step, StepResult)
            assert isinstance(step.config, FeatureExpansionConfig)
            assert step.num_features > 0
            assert isinstance(step.rho, float)

    def test_serialization_roundtrip(self) -> None:
        """ExpansionSelectionResult should serialize and deserialize."""
        cfg = FeatureExpansionConfig(max_order=2, pairwise_max_range=2)
        result = ExpansionSelectionResult(
            selected_config=cfg,
            selected_range=2,
            selected_order=2,
            range_results=[
                StepResult(
                    config=FeatureExpansionConfig(max_order=1),
                    num_features=10,
                    rho=0.5,
                ),
                StepResult(
                    config=FeatureExpansionConfig(max_order=2, pairwise_max_range=1),
                    num_features=20,
                    rho=0.7,
                ),
                StepResult(
                    config=cfg,
                    num_features=30,
                    rho=0.9,
                ),
            ],
            order_results=[],
        )
        json_str = result.model_dump_json()
        loaded = ExpansionSelectionResult.model_validate_json(json_str)
        assert loaded.selected_range == 2
        assert loaded.selected_order == 2
        assert loaded.selected_config == cfg
        assert len(loaded.range_results) == 3
        assert len(loaded.order_results) == 0

    def test_max_order_1_returns_unary(self) -> None:
        """With max_order=1, should return unary config immediately."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=6, n=60)
        scores = rng.randn(60)

        result = select_expansion_level(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            cv_folds=3,
            max_pairwise_range=5,
            max_order=1,
            seed=42,
        )

        assert result.selected_order == 1
        assert result.selected_range == 0
        assert result.selected_config.max_order == 1
        assert len(result.range_results) == 1  # just unary baseline
        assert len(result.order_results) == 0


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
