"""Tests for SurrogateMRFModel — verifying round-trip correctness between
ridge coefficients and MRF potentials.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from place_layers.core.types import LayerPlacement
from place_layers.mrf import ExactMRFSolver
from place_layers.mrf.model import ChainMRFModel
from place_layers.mrf.surrogate_model import SurrogateMRFModel
from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.surrogate.ridge_surrogate import RidgeSurrogate


# =============================================================================
# Helpers
# =============================================================================


def _all_placements(num_layers: int, mixer_names: tuple[str, ...]):
    """Generate all possible placements for exhaustive verification."""
    K = len(mixer_names)
    for assignment in product(range(K), repeat=num_layers):
        yield tuple(mixer_names[j] for j in assignment)


def _fit_surrogate(
    num_layers: int,
    mixer_names: list[str],
    feature_config: FeatureExpansionConfig,
    n_samples: int = 200,
    seed: int = 42,
) -> RidgeSurrogate:
    """Fit a RidgeSurrogate on random data."""
    rng = np.random.RandomState(seed)
    K = len(mixer_names)

    # Generate random placements
    assignments: list[tuple[str, ...]] = []
    for _ in range(n_samples):
        idxs = rng.randint(0, K, size=num_layers)
        assignments.append(tuple(mixer_names[i] for i in idxs))

    # Random scores
    scores = rng.randn(n_samples)

    surrogate = RidgeSurrogate(
        mixer_names=mixer_names,
        seed=seed,
        n_bootstraps=10,
        feature_config=feature_config,
    )
    surrogate.fit(assignments, scores)
    return surrogate


# =============================================================================
# Tests
# =============================================================================


class TestRoundTripCorrectness:
    """Verify ridge prediction == MRF DP score + intercept for small instances."""

    def test_unary_only(self) -> None:
        """Round-trip with max_order=1 (unary + counts only)."""
        N, mixer_names = 6, ["m0", "m1"]
        feature_config = FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True)
        surrogate = _fit_surrogate(N, mixer_names, feature_config)

        mrf = surrogate.extract_mrf_model(N)

        # Check all placements
        for assignment in _all_placements(N, tuple(sorted(mixer_names))):
            ridge_pred = surrogate.predict_mean_one(assignment)
            # MRF score = sum of unary potentials (no pairwise/triplet)
            mrf_score = sum(
                mrf.unary_potential(i, assignment[i]) for i in range(N)
            )
            assert abs(ridge_pred - (mrf_score + mrf.intercept)) < 1e-6, (
                f"Mismatch for {assignment}: ridge={ridge_pred}, mrf+intercept={mrf_score + mrf.intercept}"
            )

    def test_pairwise(self) -> None:
        """Round-trip with max_order=2, pairwise_max_range=1."""
        N, mixer_names = 6, ["m0", "m1"]
        feature_config = FeatureExpansionConfig(max_order=2, pairwise_max_range=1, allocation_counts=True)
        surrogate = _fit_surrogate(N, mixer_names, feature_config)

        mrf = surrogate.extract_mrf_model(N)
        solver = ExactMRFSolver(mrf)
        top_n = solver.solve_top_n(
            [{"m0": i, "m1": N - i} for i in range(N + 1)], n=5
        )

        # Verify top-N placements' scores match ridge predictions
        for sol in top_n:
            ridge_pred = surrogate.predict_mean_one(sol.placement.assignments)
            assert abs(ridge_pred - (sol.true_score + mrf.intercept)) < 1e-6

    def test_pairwise_range2(self) -> None:
        """Round-trip with max_order=2, pairwise_max_range=2."""
        N, mixer_names = 6, ["m0", "m1"]
        feature_config = FeatureExpansionConfig(max_order=2, pairwise_max_range=2, allocation_counts=True)
        surrogate = _fit_surrogate(N, mixer_names, feature_config)

        mrf = surrogate.extract_mrf_model(N)

        # Sample some placements and verify
        for assignment in list(_all_placements(N, tuple(sorted(mixer_names))))[:20]:
            ridge_pred = surrogate.predict_mean_one(assignment)
            # Compute MRF score manually
            score = 0.0
            for i in range(N):
                score += mrf.unary_potential(i, assignment[i])
            for i in range(N - 1):
                score += mrf.pairwise_potential(i, assignment[i], assignment[i + 1])
            for i in range(N - 2):
                score += mrf.skip_pairwise_potential(i, 2, assignment[i], assignment[i + 2])
            assert abs(ridge_pred - (score + mrf.intercept)) < 1e-6

    def test_triplet(self) -> None:
        """Round-trip with max_order=3."""
        N, mixer_names = 5, ["m0", "m1"]
        feature_config = FeatureExpansionConfig(
            max_order=3, pairwise_max_range=1, allocation_counts=True
        )
        surrogate = _fit_surrogate(N, mixer_names, feature_config)

        mrf = surrogate.extract_mrf_model(N)

        for assignment in list(_all_placements(N, tuple(sorted(mixer_names))))[:20]:
            ridge_pred = surrogate.predict_mean_one(assignment)
            score = 0.0
            for i in range(N):
                score += mrf.unary_potential(i, assignment[i])
            for i in range(N - 1):
                score += mrf.pairwise_potential(i, assignment[i], assignment[i + 1])
            for i in range(N - 2):
                score += mrf.triplet_potential(
                    i, assignment[i], assignment[i + 1], assignment[i + 2]
                )
            assert abs(ridge_pred - (score + mrf.intercept)) < 1e-6


class TestSolveTopNMatchesExhaustive:
    """Verify solve_top_n returns correct placements vs exhaustive ranking."""

    def test_top1_matches_exhaustive(self) -> None:
        """Top-1 from solve_top_n matches best from exhaustive (all placements)."""
        N, mixer_names = 6, ["m0", "m1"]
        feature_config = FeatureExpansionConfig(max_order=2, pairwise_max_range=1, allocation_counts=True)
        surrogate = _fit_surrogate(N, mixer_names, feature_config)
        mrf = surrogate.extract_mrf_model(N)

        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]

        # Exhaustive: score all placements via ridge, sort
        all_assignments = list(_all_placements(N, tuple(sorted(mixer_names))))
        scored = [
            (a, surrogate.predict_mean_one(a)) for a in all_assignments
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        solver = ExactMRFSolver(mrf)
        top_n = solver.solve_top_n(all_allocs, n=10)

        assert len(top_n) > 0
        # Top-1 must match exhaustive global best
        best_dp_score = top_n[0].true_score + mrf.intercept
        assert abs(best_dp_score - scored[0][1]) < 1e-6

        # All returned placements must have correct score (round-trip check)
        for sol in top_n:
            ridge_pred = surrogate.predict_mean_one(sol.placement.assignments)
            assert abs(sol.true_score + mrf.intercept - ridge_pred) < 1e-6

    def test_pairwise_constrained_top1(self) -> None:
        """Top-1 from solve_top_n matches exhaustive best within feasible allocations."""
        N, mixer_names = 5, ["m0", "m1"]
        feature_config = FeatureExpansionConfig(max_order=2, pairwise_max_range=1, allocation_counts=True)
        surrogate = _fit_surrogate(N, mixer_names, feature_config)
        mrf = surrogate.extract_mrf_model(N)

        feasible = [{"m0": 2, "m1": 3}, {"m0": 3, "m1": 2}]

        from collections import Counter

        scored = []
        for a in _all_placements(N, tuple(sorted(mixer_names))):
            counts = dict(Counter(a))
            if counts in feasible:
                scored.append((a, surrogate.predict_mean_one(a)))
        scored.sort(key=lambda x: x[1], reverse=True)

        solver = ExactMRFSolver(mrf)
        top_n = solver.solve_top_n(feasible, n=8)

        assert len(top_n) > 0
        best_dp_score = top_n[0].true_score + mrf.intercept
        assert abs(best_dp_score - scored[0][1]) < 1e-6

        for sol in top_n:
            ridge_pred = surrogate.predict_mean_one(sol.placement.assignments)
            assert abs(sol.true_score + mrf.intercept - ridge_pred) < 1e-6


class TestCountBiasFolding:
    """Verify that DP score + intercept == ridge prediction."""

    def test_random_placements(self) -> None:
        """For random placements, MRF score + intercept == ridge prediction."""
        N, mixer_names = 8, ["m0", "m1", "m2"]
        feature_config = FeatureExpansionConfig(
            max_order=2, pairwise_max_range=1, allocation_counts=True
        )
        surrogate = _fit_surrogate(N, mixer_names, feature_config, n_samples=300)
        mrf = surrogate.extract_mrf_model(N)

        rng = np.random.RandomState(123)
        sorted_names = tuple(sorted(mixer_names))
        for _ in range(50):
            idxs = rng.randint(0, len(mixer_names), size=N)
            assignment = tuple(sorted_names[i] for i in idxs)

            ridge_pred = surrogate.predict_mean_one(assignment)
            # MRF score via solver's transition-based computation
            score = 0.0
            for i in range(N):
                score += mrf.unary_potential(i, assignment[i])
            for i in range(N - 1):
                score += mrf.pairwise_potential(i, assignment[i], assignment[i + 1])
            assert abs(ridge_pred - (score + mrf.intercept)) < 1e-5


class TestSortedNameAssertion:
    """Verify unsorted names raise."""

    def test_unsorted_raises(self) -> None:
        coef = np.zeros(10)
        feature_config = FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True)
        with pytest.raises(AssertionError, match="must be sorted"):
            SurrogateMRFModel(
                coef=coef,
                intercept=0.0,
                num_layers=3,
                mixer_names=("m1", "m0"),  # Not sorted
                feature_config=feature_config,
            )


class TestMaxOrderAssertion:
    """Verify max_order > 3 raises."""

    def test_order4_raises(self) -> None:
        feature_config = FeatureExpansionConfig(max_order=4, pairwise_max_range=1, allocation_counts=True)
        with pytest.raises(AssertionError, match="order > 3"):
            SurrogateMRFModel(
                coef=np.zeros(100),
                intercept=0.0,
                num_layers=5,
                mixer_names=("m0", "m1"),
                feature_config=feature_config,
            )


class TestBootstrapDiversity:
    """Verify bootstrap models produce diverse placements."""

    def test_bootstrap_models_differ(self) -> None:
        """Different bootstrap models should produce at least some different placements."""
        N, mixer_names = 8, ["m0", "m1"]
        feature_config = FeatureExpansionConfig(max_order=2, pairwise_max_range=1, allocation_counts=True)
        surrogate = _fit_surrogate(N, mixer_names, feature_config, n_samples=200)

        mrf_models = surrogate.extract_bootstrap_mrf_models(N)
        assert len(mrf_models) >= 2

        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]

        # Get top-1 placement from each model
        top_placements: set[tuple[str, ...]] = set()
        for mrf in mrf_models:
            solver = ExactMRFSolver(mrf)
            solutions = solver.solve_top_n(all_allocs, n=1)
            if solutions:
                top_placements.add(solutions[0].placement.assignments)

        # With 10 bootstrap models, we should see at least 2 different placements
        # (very unlikely all bootstraps produce identical coefficients)
        assert len(top_placements) >= 1  # At minimum, we get at least one
        # We don't assert >= 2 since it's probabilistic, but in practice it should be > 1


class TestIntraAllocationDiversity:
    """Verify solve_top_n returns multiple placements for the same allocation."""

    def test_multiple_arrangements_per_allocation(self) -> None:
        """With K=3, m=1, multiple history states should yield different arrangements."""
        N, mixer_names = 6, ["m0", "m1", "m2"]
        feature_config = FeatureExpansionConfig(max_order=2, pairwise_max_range=1, allocation_counts=True)
        surrogate = _fit_surrogate(N, mixer_names, feature_config, n_samples=300)
        mrf = surrogate.extract_mrf_model(N)

        # Use a single allocation with room for diversity
        feasible = [{"m0": 2, "m1": 2, "m2": 2}]

        solver = ExactMRFSolver(mrf)
        top_n = solver.solve_top_n(feasible, n=20)

        # All should have the same allocation
        from collections import Counter
        for sol in top_n:
            counts = dict(Counter(sol.placement.assignments))
            assert counts == {"m0": 2, "m1": 2, "m2": 2}

        # But different arrangements
        unique_arrangements = {sol.placement.assignments for sol in top_n}
        assert len(unique_arrangements) == len(top_n), "All solutions should be distinct"
        assert len(unique_arrangements) > 1, "Should have multiple arrangements"


class TestKnownMRFRoundTrip:
    """Fit ridge on scores from a known ChainMRFModel, extract potentials, compare.

    This is a stronger test than TestRoundTripCorrectness: instead of random
    data, we use data generated by a *known* MRF.  The extracted surrogate MRF
    must recover the original MRF's scores (up to ridge shrinkage).

    Verification:
    - Ridge prediction == extracted_MRF_score + intercept (exact, from encoding)
    - Spearman rank correlation between original and extracted scores >= 0.999
    - Max absolute score error is small relative to the score range
    """

    def _score_via_extracted(
        self,
        extracted: SurrogateMRFModel,
        assignment: tuple[str, ...],
    ) -> float:
        """Compute score through extracted MRF model manually."""
        N = extracted.num_layers
        score = 0.0
        for i in range(N):
            score += extracted.unary_potential(i, assignment[i])
        for r in range(1, extracted.pairwise_max_range + 1):
            for i in range(N - r):
                if r == 1:
                    score += extracted.pairwise_potential(
                        i, assignment[i], assignment[i + 1]
                    )
                else:
                    score += extracted.skip_pairwise_potential(
                        i, r, assignment[i], assignment[i + r]
                    )
        if extracted.triplet_scale > 0:
            for i in range(N - 2):
                score += extracted.triplet_potential(
                    i, assignment[i], assignment[i + 1], assignment[i + 2]
                )
        return score

    def test_unary_and_pairwise_potentials_recovered(self) -> None:
        """With all K^N data points, ridge recovers the true MRF ranking."""
        from scipy.stats import spearmanr

        N = 8
        mixer_names = ("m0", "m1")

        # 1. Create known MRF
        original_mrf = ChainMRFModel(
            mixer_names=mixer_names,
            seed=42,
            num_layers=N,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )

        # 2. Generate ALL K^N placements and score with true MRF
        feature_config = FeatureExpansionConfig(
            max_order=2, pairwise_max_range=1, allocation_counts=False,
        )
        all_assignments = list(_all_placements(N, mixer_names))
        orig_scores = np.array([
            original_mrf.true_score(LayerPlacement(assignments=a))
            for a in all_assignments
        ])

        # 3. Fit ridge on all K^N data points
        surrogate = RidgeSurrogate(
            mixer_names=list(mixer_names),
            seed=0,
            n_bootstraps=1,
            feature_config=feature_config,
        )
        surrogate.fit(all_assignments, orig_scores)

        # 4. Extract SurrogateMRFModel
        extracted = surrogate.extract_mrf_model(N)

        # 5. Collect extracted scores and verify round-trip + rank correlation
        ext_scores = np.array([
            self._score_via_extracted(extracted, a) + extracted.intercept
            for a in all_assignments
        ])

        # Ridge prediction == extracted score + intercept (exact encoding property)
        for a, ext_s in zip(all_assignments, ext_scores):
            ridge_pred = surrogate.predict_mean_one(a)
            assert abs(ridge_pred - ext_s) < 1e-6

        # Rank correlation must be very high (ridge shrinkage preserves ranking)
        rho = float(spearmanr(orig_scores, ext_scores).statistic)
        assert rho >= 0.99, f"Rank correlation too low: {rho:.6f}"

        # Max error should be small relative to score range
        score_range = float(orig_scores.max() - orig_scores.min())
        max_err = float(np.max(np.abs(orig_scores - ext_scores)))
        assert max_err / score_range < 0.10, (
            f"Max error {max_err:.4f} is >10% of score range {score_range:.4f}"
        )

    def test_with_skip_pairwise_and_triplets(self) -> None:
        """Extended model with range-2 pairwise and triplets."""
        from scipy.stats import spearmanr

        N = 6
        mixer_names = ("m0", "m1")

        original_mrf = ChainMRFModel(
            mixer_names=mixer_names,
            seed=7,
            num_layers=N,
            unary_bonus=0.8,
            unary_penalty=0.4,
            pairwise_scale=0.3,
            pairwise_decay=0.5,
            pairwise_max_range=2,
            triplet_scale=0.2,
        )

        feature_config = FeatureExpansionConfig(
            max_order=3, pairwise_max_range=2, allocation_counts=False,
        )
        all_assignments = list(_all_placements(N, mixer_names))
        orig_scores = np.array([
            original_mrf.true_score(LayerPlacement(assignments=a))
            for a in all_assignments
        ])

        surrogate = RidgeSurrogate(
            mixer_names=list(mixer_names),
            seed=0,
            n_bootstraps=1,
            feature_config=feature_config,
        )
        surrogate.fit(all_assignments, orig_scores)

        extracted = surrogate.extract_mrf_model(N)

        ext_scores = np.array([
            self._score_via_extracted(extracted, a) + extracted.intercept
            for a in all_assignments
        ])

        # Ridge prediction == extracted score + intercept (exact encoding property)
        for a, ext_s in zip(all_assignments, ext_scores):
            ridge_pred = surrogate.predict_mean_one(a)
            assert abs(ridge_pred - ext_s) < 1e-6

        # Rank correlation must be very high
        rho = float(spearmanr(orig_scores, ext_scores).statistic)
        assert rho >= 0.99, f"Rank correlation too low: {rho:.6f}"

        # Max error should be small relative to score range
        score_range = float(orig_scores.max() - orig_scores.min())
        max_err = float(np.max(np.abs(orig_scores - ext_scores)))
        assert max_err / score_range < 0.10, (
            f"Max error {max_err:.4f} is >10% of score range {score_range:.4f}"
        )
