"""Unit tests for surrogate model selection (expansion level selection)."""

import numpy as np

from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.surrogate.encoding import FeatureScaling
from place_layers.surrogate.model_selection import (
    ExpansionSelectionResult,
    StepResult,
    ridge_cv_rho,
    select_expansion_bayes_evidence,
    select_expansion_ridge_cv,
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
# select_expansion_ridge_cv tests
# =============================================================================


class TestSelectExpansionRidgeCv:
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

        result = select_expansion_ridge_cv(
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

        # Unary baseline should have lower score than first pairwise step
        unary_score = result.range_results[0].score
        pairwise_score = result.range_results[1].score
        assert pairwise_score > unary_score + 0.05, (
            f"Expected pairwise to improve significantly: "
            f"unary={unary_score:.4f}, pairwise={pairwise_score:.4f}"
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

        result = select_expansion_ridge_cv(
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
        # Unary baseline should already have very high score for this linear signal
        unary_score = result.range_results[0].score
        assert unary_score > 0.9, (
            f"Expected high score for linear signal: unary={unary_score:.4f}"
        )
        # Should select order <= 2 (likely 1 since unary captures the signal)
        assert result.selected_order <= 2

    def test_level_result_types(self) -> None:
        """StepResult and ExpansionSelectionResult should be well-formed."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=6, n=60)
        scores = rng.randn(60)

        result = select_expansion_ridge_cv(
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
            assert isinstance(step.score, float)
            assert step.rho is not None  # ridge_cv path always sets rho
        for step in result.order_results:
            assert isinstance(step, StepResult)
            assert isinstance(step.config, FeatureExpansionConfig)
            assert step.num_features > 0
            assert isinstance(step.score, float)
            assert step.rho is not None  # ridge_cv path always sets rho

        # Exactly one step in range_results should have a non-None stop_reason
        range_stops = [s for s in result.range_results if s.stop_reason is not None]
        assert len(range_stops) == 1, (
            f"Expected exactly 1 terminal step in range_results, got {len(range_stops)}"
        )
        assert result.phase1_stop_reason is not None

    def test_serialization_roundtrip(self) -> None:
        """ExpansionSelectionResult should serialize and deserialize."""
        cfg = FeatureExpansionConfig(max_order=2, pairwise_max_range=2, allocation_counts=True)
        result = ExpansionSelectionResult(
            selected_config=cfg,
            selected_range=2,
            selected_order=2,
            range_results=[
                StepResult(
                    config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
                    num_features=10,
                    score=0.5,
                    rho=0.5,
                ),
                StepResult(
                    config=FeatureExpansionConfig(max_order=2, pairwise_max_range=1, allocation_counts=True),
                    num_features=20,
                    score=0.7,
                    rho=0.7,
                ),
                StepResult(
                    config=cfg,
                    num_features=30,
                    score=0.9,
                    rho=0.9,
                    stop_reason="target_reached",
                ),
            ],
            order_results=[],
            phase1_stop_reason="target_reached",
            phase2_stop_reason=None,
        )
        json_str = result.model_dump_json()
        loaded = ExpansionSelectionResult.model_validate_json(json_str)
        assert loaded.selected_range == 2
        assert loaded.selected_order == 2
        assert loaded.selected_config == cfg
        assert len(loaded.range_results) == 3
        assert len(loaded.order_results) == 0
        assert loaded.range_results[0].stop_reason is None
        assert loaded.range_results[1].stop_reason is None
        assert loaded.range_results[2].stop_reason == "target_reached"
        assert loaded.phase1_stop_reason == "target_reached"
        assert loaded.phase2_stop_reason is None

    def test_guard_unlocks_higher_orders_with_more_data(self) -> None:
        """Feature-to-sample guard should relax with more data,
        unlocking higher-order expansions."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b", "c"]
        num_layers = 8

        # Scores with strong triplet signal
        def score_fn(a):
            s = sum({"a": 1.0, "b": -0.3, "c": 0.0}[m] for m in a)
            for i in range(len(a) - 1):
                s += 0.3 if a[i] == a[i + 1] else -0.1
            for i in range(len(a) - 2):
                if a[i] == a[i + 1] == a[i + 2]:
                    s += 0.8
            return s

        # Small batch: 50 samples, guard = 100
        # Order 2 range 1 = 90 features → fits
        # Order 3 range 1 = 252 features → blocked
        small = _random_placements(rng, mixer_names, num_layers, 50)
        small_scores = np.array([score_fn(a) for a in small])
        result_small = select_expansion_ridge_cv(
            assignments=small, scores=small_scores,
            mixer_names=mixer_names, cv_folds=5, target_rho=0.99,
            max_pairwise_range=3, max_order=3, seed=42,
        )
        assert result_small.selected_order <= 2

        # Large batch: 200 samples, guard = 400
        # Order 3 range 1 = 252 features → fits
        large = _random_placements(np.random.RandomState(42), mixer_names, num_layers, 200)
        large_scores = np.array([score_fn(a) for a in large])
        result_large = select_expansion_ridge_cv(
            assignments=large, scores=large_scores,
            mixer_names=mixer_names, cv_folds=5, target_rho=0.99,
            max_pairwise_range=3, max_order=3, seed=42,
        )
        assert result_large.selected_order == 3

    def test_stop_reasons_populated(self) -> None:
        """Stop reasons should be populated on target_reached scenario."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        num_layers = 12
        n_samples = 200

        assignments = _random_placements(rng, mixer_names, num_layers, n_samples)

        # Purely unary scores — high rho at unary, should hit target_reached
        scores = np.array([
            sum(1.0 for m in a if m == "a") for a in assignments
        ], dtype=np.float64)

        result = select_expansion_ridge_cv(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            cv_folds=5,
            target_rho=0.95,  # Low enough for unary to reach
            min_improvement=0.01,
            max_pairwise_range=5,
            max_order=3,
            seed=42,
        )

        # Unary should reach target; phase 1 stops with target_reached
        assert result.phase1_stop_reason == "target_reached"
        terminal = [s for s in result.range_results if s.stop_reason is not None]
        assert len(terminal) == 1
        assert terminal[0].stop_reason == "target_reached"

        # Phase 2 skipped (target already reached)
        assert result.phase2_stop_reason is None
        assert result.order_results == []

    def test_feature_guard_stop_reason(self) -> None:
        """Feature guard stop reason should be set when guard blocks expansion."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b", "c"]
        num_layers = 8

        def score_fn(a):
            s = sum({"a": 1.0, "b": -0.3, "c": 0.0}[m] for m in a)
            for i in range(len(a) - 1):
                s += 0.3 if a[i] == a[i + 1] else -0.1
            for i in range(len(a) - 2):
                if a[i] == a[i + 1] == a[i + 2]:
                    s += 0.8
            return s

        # Small batch: 50 samples, guard = 100
        # Order 3 features would exceed guard → feature_guard
        small = _random_placements(rng, mixer_names, num_layers, 50)
        small_scores = np.array([score_fn(a) for a in small])
        result = select_expansion_ridge_cv(
            assignments=small, scores=small_scores,
            mixer_names=mixer_names, cv_folds=5, target_rho=0.99,
            max_pairwise_range=3, max_order=3, seed=42,
        )

        # At least one phase should have feature_guard as stop reason
        all_stops = (
            [s.stop_reason for s in result.range_results if s.stop_reason]
            + [s.stop_reason for s in result.order_results if s.stop_reason]
        )
        has_guard = (
            "feature_guard" in all_stops
            or result.phase1_stop_reason == "feature_guard"
            or result.phase2_stop_reason == "feature_guard"
        )
        assert has_guard, (
            f"Expected feature_guard stop reason with 50 samples, "
            f"got phase1={result.phase1_stop_reason}, "
            f"phase2={result.phase2_stop_reason}, "
            f"step_stops={all_stops}"
        )

    def test_max_order_1_returns_unary(self) -> None:
        """With max_order=1, should return unary config immediately."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=6, n=60)
        scores = rng.randn(60)

        result = select_expansion_ridge_cv(
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
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
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

        cfg = FeatureExpansionConfig(max_order=2, pairwise_max_range=1, allocation_counts=True)
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
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            cv_folds=5,
            seed=42,
        )
        assert abs(rho) < 0.4


# =============================================================================
# select_expansion_bayes_evidence tests
# =============================================================================


class TestSelectExpansionBayesEvidence:
    def test_unary_prefers_simpler(self) -> None:
        """Unary signal with noise, evidence should select order=1."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        num_layers = 12
        n_samples = 200

        assignments = _random_placements(rng, mixer_names, num_layers, n_samples)

        scores = np.array([
            sum(1.0 for m in a if m == "a") for a in assignments
        ], dtype=np.float64)
        scores += 0.5 * rng.randn(n_samples)  # Add noise

        result = select_expansion_bayes_evidence(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            prior_alpha=1.0,
            prior_a0=1e-6,
            prior_b0=1e-6,
            feature_scaling=FeatureScaling.COUNTS_ONLY,
            max_pairwise_range=5,
            max_order=3,
        )

        assert isinstance(result, ExpansionSelectionResult)
        assert result.selected_order == 1

    def test_genuine_pairwise_selects_order2(self) -> None:
        """Non-unary-expressible signal (adjacent-different count),
        evidence should select order>=2."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        num_layers = 12
        n_samples = 200

        assignments = _random_placements(rng, mixer_names, num_layers, n_samples)

        # Adjacent-different count: sum(1 if a[i] != a[i+1] else 0)
        scores = np.array([
            sum(1 if a[i] != a[i + 1] else 0 for i in range(len(a) - 1))
            for a in assignments
        ], dtype=np.float64)

        result = select_expansion_bayes_evidence(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            prior_alpha=1.0,
            prior_a0=1e-6,
            prior_b0=1e-6,
            max_pairwise_range=5,
            max_order=3,
        )

        assert isinstance(result, ExpansionSelectionResult)
        assert result.selected_order >= 2

    def test_result_structure(self) -> None:
        """Verify ExpansionSelectionResult fields for evidence path."""
        rng = np.random.RandomState(42)
        mixer_names = ["a", "b"]
        assignments = _random_placements(rng, mixer_names, num_layers=6, n=60)
        scores = rng.randn(60)

        result = select_expansion_bayes_evidence(
            assignments=assignments,
            scores=scores,
            mixer_names=mixer_names,
            prior_alpha=1.0,
            prior_a0=1e-6,
            prior_b0=1e-6,
            max_pairwise_range=3,
            max_order=2,
        )

        assert result.phase1_stop_reason == "exhaustive_argmax"
        assert result.phase2_stop_reason is None
        # All steps from evidence path have rho=None
        for step in result.range_results:
            assert step.rho is None
        for step in result.order_results:
            assert step.rho is None
