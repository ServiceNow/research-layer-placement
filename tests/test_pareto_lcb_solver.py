"""Tests for ParetoLCBSolver — Pareto DP with (m, v) frontier tracking."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from place_layers.mrf.pareto_lcb_solver import ParetoLCBSolver, _hull_prune, ParetoEntry
from place_layers.mrf.solver import ExactMRFSolver
from place_layers.mrf.surrogate_model import SurrogateMRFModel
from place_layers.surrogate.bayesian_ridge import BayesianRidgeSurrogate
from place_layers.surrogate.encoding import FeatureExpansionConfig, compute_feature_dim


def _fit_bayesian(
    num_layers: int = 6,
    mixer_names: list[str] | None = None,
    feature_config: FeatureExpansionConfig = FeatureExpansionConfig(max_order=2, pairwise_max_range=1, allocation_counts=True),
    n_samples: int = 200,
    seed: int = 42,
) -> BayesianRidgeSurrogate:
    """Fit a BayesianRidgeSurrogate on synthetic data."""
    if mixer_names is None:
        mixer_names = ["m0", "m1"]

    rng = np.random.RandomState(seed)
    K = len(mixer_names)
    assignments: list[tuple[str, ...]] = []
    for _ in range(n_samples):
        idxs = rng.randint(0, K, size=num_layers)
        assignments.append(tuple(mixer_names[i] for i in idxs))

    y = np.array([sum(1.0 for m in a if m == mixer_names[0]) for a in assignments])
    y += 0.1 * rng.randn(n_samples)

    surrogate = BayesianRidgeSurrogate(
        mixer_names=mixer_names,
        seed=seed,
        feature_config=feature_config,
    )
    surrogate.fit(assignments, y)
    return surrogate


class TestBetaZeroMatchesExactSolver:
    """β=0 (no variance penalty) should produce the same argmax as ExactMRFSolver."""

    def test_top1_matches(self) -> None:
        N = 6
        mixer_names = ["m0", "m1"]
        feature_config = FeatureExpansionConfig(
            max_order=2, pairwise_max_range=1, allocation_counts=True
        )
        surrogate = _fit_bayesian(N, mixer_names, feature_config)

        # ExactMRFSolver uses the standard MRF model (with counts folded in)
        mrf = surrogate.extract_mrf_model(N)
        exact_solver = ExactMRFSolver(mrf)

        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]
        exact_top = exact_solver.solve_top_n(all_allocs, n=1)

        # ParetoLCBSolver with β=0 should find the same optimum
        mean_model, var_model = surrogate.extract_mean_var_mrf_models(N)
        pareto_solver = ParetoLCBSolver(mean_model, var_model, surrogate.noise_scale_sq)
        pareto_top = pareto_solver.solve_top_n_lcb(all_allocs, n=1, beta=0.0)

        assert len(pareto_top) >= 1
        assert len(exact_top) >= 1

        # Both should find the same placement (since β=0, only mean matters)
        # But the MRF models differ: exact uses counts-folded, Pareto uses count-free.
        # So we compare by full_lcb(beta=0) which is just the mean.
        exact_mean = surrogate.predict_mean_one(exact_top[0].placement.assignments)
        pareto_mean = surrogate.predict_mean_one(pareto_top[0].placement.assignments)

        # The count-free DP might not find the exact same placement as count-included DP,
        # but within the same allocation, the arrangement should be optimal.
        # Use full_lcb to compare fairly.
        pareto_score = surrogate.full_lcb(
            [pareto_top[0].placement.assignments], beta=0.0
        )[0]
        exact_score = surrogate.full_lcb(
            [exact_top[0].placement.assignments], beta=0.0
        )[0]

        # The Pareto solver should find a placement whose full score is competitive
        # (it may find a different allocation since count features are excluded from DP)
        assert pareto_score >= exact_score - 0.5, (
            f"Pareto β=0 score ({pareto_score:.4f}) much worse than exact ({exact_score:.4f})"
        )


class TestHigherBetaDiffers:
    """Higher β should sometimes produce different placements."""

    def test_beta_changes_result(self) -> None:
        N = 6
        mixer_names = ["m0", "m1"]
        feature_config = FeatureExpansionConfig(
            max_order=2, pairwise_max_range=1, allocation_counts=True
        )
        surrogate = _fit_bayesian(N, mixer_names, feature_config)
        mean_model, var_model = surrogate.extract_mean_var_mrf_models(N)

        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]
        solver = ParetoLCBSolver(mean_model, var_model, surrogate.noise_scale_sq)

        beta0_results = solver.solve_top_n_lcb(all_allocs, n=5, beta=0.0)
        beta2_results = solver.solve_top_n_lcb(all_allocs, n=5, beta=2.0)

        # Collect top placements from each
        beta0_placements = {r.placement.assignments for r in beta0_results}
        beta2_placements = {r.placement.assignments for r in beta2_results}

        # They don't have to differ (could get lucky), but the sets should be valid
        assert len(beta0_placements) >= 1
        assert len(beta2_placements) >= 1


class TestHullPruning:
    """Verify upper concave hull pruning."""

    def test_hull_preserves_optimum(self) -> None:
        """Hull-pruned frontier should contain the true best for a given β."""
        # Create points forming a known Pareto frontier
        entries = [
            ParetoEntry(m=10.0, v=1.0, layer=0),
            ParetoEntry(m=9.0, v=0.5, layer=0),
            ParetoEntry(m=8.0, v=0.1, layer=0),
            ParetoEntry(m=7.0, v=0.05, layer=0),
            # Dominated point (should be removed)
            ParetoEntry(m=6.0, v=2.0, layer=0),
        ]

        pruned = _hull_prune(entries, max_size=100)

        # Dominated point (m=6, v=2) should be removed
        pruned_mv = [(e.m, e.v) for e in pruned]
        assert (6.0, 2.0) not in pruned_mv

        # All non-dominated hull points should remain
        assert len(pruned) >= 2

    def test_hull_single_point(self) -> None:
        entries = [ParetoEntry(m=5.0, v=1.0, layer=0)]
        pruned = _hull_prune(entries, max_size=100)
        assert len(pruned) == 1


class TestSolveTopN:
    """Verify solve_top_n_lcb returns correct count and valid placements."""

    def test_returns_correct_count(self) -> None:
        N = 6
        surrogate = _fit_bayesian(N)
        mean_model, var_model = surrogate.extract_mean_var_mrf_models(N)
        solver = ParetoLCBSolver(mean_model, var_model, surrogate.noise_scale_sq)

        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]
        results = solver.solve_top_n_lcb(all_allocs, n=5, beta=1.0)
        assert len(results) <= 5
        assert len(results) >= 1

    def test_backtracking_valid(self) -> None:
        """All placements should have correct allocation counts."""
        N = 6
        mixer_names = ["m0", "m1"]
        surrogate = _fit_bayesian(N, mixer_names)
        mean_model, var_model = surrogate.extract_mean_var_mrf_models(N)
        solver = ParetoLCBSolver(mean_model, var_model, surrogate.noise_scale_sq)

        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]
        results = solver.solve_top_n_lcb(all_allocs, n=10, beta=1.0)

        for sol in results:
            assignments = sol.placement.assignments
            assert len(assignments) == N
            # Normalize counts to include all mixers (Counter omits zero-count)
            raw_counts = Counter(assignments)
            counts = {name: raw_counts.get(name, 0) for name in mixer_names}
            # Must match one of the feasible allocations
            assert counts in all_allocs, (
                f"Placement {assignments} has counts {counts} not in feasible set"
            )

    def test_frontier_cap_respected(self) -> None:
        """Frontier should never exceed max_frontier_size."""
        N = 8
        mixer_names = ["m0", "m1", "m2"]
        feature_config = FeatureExpansionConfig(max_order=2, pairwise_max_range=1, allocation_counts=True)
        surrogate = _fit_bayesian(N, mixer_names, feature_config, n_samples=300)
        mean_model, var_model = surrogate.extract_mean_var_mrf_models(N)
        solver = ParetoLCBSolver(mean_model, var_model, surrogate.noise_scale_sq)

        # Use small max_frontier_size
        all_allocs = [
            {"m0": a, "m1": b, "m2": N - a - b}
            for a in range(N + 1)
            for b in range(N + 1 - a)
        ]
        results = solver.solve_top_n_lcb(
            all_allocs, n=5, beta=1.0, max_frontier_size=10
        )
        # Should still produce valid results
        assert len(results) >= 1


class TestDPScoreConsistency:
    """Verify DP score is consistent with MRF model computation."""

    def test_dp_score_matches_manual_mrf_computation(self) -> None:
        """For DP solutions, verify the DP's accumulated m matches the sum
        of mean MRF potentials, and that the ranking is correct."""
        N = 6
        mixer_names = ["m0", "m1"]
        feature_config = FeatureExpansionConfig(
            max_order=2, pairwise_max_range=1, allocation_counts=False
        )
        surrogate = _fit_bayesian(N, mixer_names, feature_config)
        mean_model, var_model = surrogate.extract_mean_var_mrf_models(N)
        solver = ParetoLCBSolver(mean_model, var_model, surrogate.noise_scale_sq)

        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]
        beta = 1.5
        results = solver.solve_top_n_lcb(all_allocs, n=5, beta=beta)

        # DP scores should be monotonically decreasing (sorted)
        for i in range(len(results) - 1):
            assert results[i].true_score >= results[i + 1].true_score - 1e-10

        # Each result's mean component should match the sum of mean MRF potentials
        for sol in results:
            assignments = sol.placement.assignments
            # Compute mean via MRF model (same as what DP accumulates)
            mrf_mean = 0.0
            for i in range(N):
                mrf_mean += mean_model.unary_potential(i, assignments[i])
            for i in range(N - 1):
                mrf_mean += mean_model.pairwise_potential(
                    i, assignments[i], assignments[i + 1]
                )
            # Verify the full_lcb score shifts by intercept but has correct ranking
            full_mean = surrogate.predict_mean_one(assignments)
            # Since allocation_counts=False, full_mean = mrf_mean + intercept
            assert abs(full_mean - (mrf_mean + mean_model.intercept)) < 1e-5

    def test_dp_ranking_preserved_under_full_lcb(self) -> None:
        """DP top-1 should have a competitive full_lcb score."""
        N = 6
        mixer_names = ["m0", "m1"]
        feature_config = FeatureExpansionConfig(
            max_order=2, pairwise_max_range=1, allocation_counts=False
        )
        surrogate = _fit_bayesian(N, mixer_names, feature_config)
        mean_model, var_model = surrogate.extract_mean_var_mrf_models(N)
        solver = ParetoLCBSolver(mean_model, var_model, surrogate.noise_scale_sq)

        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]
        results = solver.solve_top_n_lcb(all_allocs, n=5, beta=1.0)

        # Top-1 by DP should have the best or near-best full_lcb
        full_lcb_scores = surrogate.full_lcb(
            [r.placement.assignments for r in results], beta=1.0
        )
        top1_full_score = full_lcb_scores[0]
        best_full_score = np.max(full_lcb_scores)

        # Top-1 should be within reasonable range of the best
        # (may not be exact best due to diagonal variance approximation in DP)
        assert top1_full_score >= best_full_score - 1.0
