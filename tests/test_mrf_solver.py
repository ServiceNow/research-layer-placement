"""Tests for MRF solver correctness.

The core guarantees of the MRF solver are:
1. Unconstrained solver finds the global optimum (verified via exhaustive search)
2. Constrained solver finds the global optimum within allocation constraints
3. True score computation matches the evaluator's score without noise
4. solve_top_n returns globally top-N placements across feasible allocations
5. solve_top_n_per_allocation returns top-N per allocation with single DP build
"""

from collections import Counter
from math import comb
from unittest.mock import patch

import pytest

from place_layers.core.types import LayerPlacement
from place_layers.evaluator.mock_supernet_evaluator import (
    MockSupernetEvaluator,
    MockSupernetEvaluatorConfig,
)
from place_layers.mrf import ChainMRFModel, ExactMRFSolver


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_evaluator() -> MockSupernetEvaluator:
    """Create a small evaluator for exhaustive verification (N=4, K=2)."""
    config = MockSupernetEvaluatorConfig(
        num_layers=4,
        mixer_names=["m0", "m1"],
        seed=42,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        noise_sigma=0.0,  # No noise for deterministic tests
        noise_seed_shift=0,
        eval_delay_seconds=0.0,
        pairwise_decay=0.5,
        pairwise_max_range=1,
        triplet_scale=0.0,
    )
    return MockSupernetEvaluator(config)


@pytest.fixture
def medium_evaluator() -> MockSupernetEvaluator:
    """Create a medium evaluator (N=6, K=3)."""
    config = MockSupernetEvaluatorConfig(
        num_layers=6,
        mixer_names=["m0", "m1", "m2"],
        seed=123,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        noise_sigma=0.0,
        noise_seed_shift=0,
        eval_delay_seconds=0.0,
        pairwise_decay=0.5,
        pairwise_max_range=1,
        triplet_scale=0.0,
    )
    return MockSupernetEvaluator(config)


@pytest.fixture
def noisy_evaluator() -> MockSupernetEvaluator:
    """Create an evaluator with noise for regret tests."""
    config = MockSupernetEvaluatorConfig(
        num_layers=10,
        mixer_names=["m0", "m1"],
        seed=42,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        noise_sigma=0.5,  # Significant noise
        noise_seed_shift=0,
        eval_delay_seconds=0.0,
        pairwise_decay=0.5,
        pairwise_max_range=1,
        triplet_scale=0.0,
    )
    return MockSupernetEvaluator(config)


# =============================================================================
# Test ChainMRFModel
# =============================================================================


class TestChainMRFModel:
    """Tests for ChainMRFModel."""

    def test_mrf_model_from_evaluator(self, small_evaluator: MockSupernetEvaluator) -> None:
        """Test accessing MRF model via evaluator property."""
        model = small_evaluator.mrf_model

        assert model.num_layers == 4
        assert model.num_mixers == 2
        assert model.mixer_names == ("m0", "m1")
        assert len(model.layer_needs) == 4
        assert len(model.pairwise_weights) == 3  # num_layers - 1
        assert model.unary_bonus == 1.0
        assert model.unary_penalty == 0.5
        assert model.pairwise_scale == 0.3

    @pytest.mark.asyncio
    async def test_true_score_matches_evaluator_no_noise(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """Test that true_score matches evaluator score when noise_sigma=0."""
        model = small_evaluator.mrf_model

        # Test several placements
        placements = [
            LayerPlacement(assignments=("m0", "m0", "m0", "m0")),
            LayerPlacement(assignments=("m1", "m1", "m1", "m1")),
            LayerPlacement(assignments=("m0", "m1", "m0", "m1")),
            LayerPlacement(assignments=("m1", "m0", "m1", "m0")),
        ]

        for placement in placements:
            true_score = model.true_score(placement)
            # With noise_sigma=0, evaluator should return same score
            evaluator_score = await small_evaluator.evaluate(placement, noise_sigma=0.0)
            assert abs(true_score - evaluator_score) < 1e-10

    def test_construction(self) -> None:
        """Test direct construction produces consistent results."""
        model = ChainMRFModel(
            mixer_names=("m0", "m1"),
            seed=42,
            num_layers=4,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
        )
        assert model.num_layers == 4
        assert model.num_mixers == 2
        assert all(name in ("m0", "m1") for name in model.layer_needs)

    def test_construction_matches_evaluator(self, small_evaluator: MockSupernetEvaluator) -> None:
        """Test direct construction with same params produces same model as evaluator."""
        model_direct = ChainMRFModel(
            mixer_names=("m0", "m1"),
            seed=42,
            num_layers=4,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
        )
        model_from_eval = small_evaluator.mrf_model

        assert model_direct.layer_needs == model_from_eval.layer_needs
        assert model_direct.pairwise_weights == model_from_eval.pairwise_weights


# =============================================================================
# Test ExactMRFSolver - Unconstrained
# =============================================================================


class TestExactMRFSolverUnconstrained:
    """Tests for unconstrained optimization."""

    def test_matches_exhaustive_small(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify DP matches exhaustive enumeration for small instance (N=4, K=2)."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        # Get DP solution
        dp_solution = solver.solve_unconstrained()

        # Get exhaustive solution
        all_placements = solver.enumerate_all()
        best_exhaustive = all_placements[0]  # Already sorted by score descending

        assert abs(dp_solution.true_score - best_exhaustive[1]) < 1e-10
        assert dp_solution.placement == best_exhaustive[0]

    def test_matches_exhaustive_medium(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify DP matches exhaustive enumeration for medium instance (N=6, K=3)."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        # Get DP solution
        dp_solution = solver.solve_unconstrained()

        # Get exhaustive solution
        all_placements = solver.enumerate_all()
        best_exhaustive = all_placements[0]

        assert abs(dp_solution.true_score - best_exhaustive[1]) < 1e-10
        assert dp_solution.placement == best_exhaustive[0]

    def test_solution_score_is_correct(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify that the solution's score matches recomputed score."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        solution = solver.solve_unconstrained()

        # Recompute score
        recomputed_score = model.true_score(solution.placement)
        assert abs(solution.true_score - recomputed_score) < 1e-10


# =============================================================================
# Test ExactMRFSolver - Constrained
# =============================================================================


class TestExactMRFSolverConstrained:
    """Tests for constrained optimization."""

    def test_matches_exhaustive_constrained(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify constrained DP matches exhaustive enumeration."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        allocation = {"m0": 2, "m1": 2}

        # Get DP solution
        dp_solution = solver.solve_constrained(allocation)
        assert dp_solution is not None

        # Get exhaustive solution
        all_constrained = solver.enumerate_constrained(allocation)
        best_exhaustive = all_constrained[0]

        assert abs(dp_solution.true_score - best_exhaustive[1]) < 1e-10
        assert dp_solution.placement == best_exhaustive[0]

    def test_matches_exhaustive_medium_constrained(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify constrained DP for medium instance (N=6, K=3)."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        allocation = {"m0": 2, "m1": 2, "m2": 2}

        # Get DP solution
        dp_solution = solver.solve_constrained(allocation)
        assert dp_solution is not None

        # Get exhaustive solution
        all_constrained = solver.enumerate_constrained(allocation)
        best_exhaustive = all_constrained[0]

        assert abs(dp_solution.true_score - best_exhaustive[1]) < 1e-10
        assert dp_solution.placement == best_exhaustive[0]

    def test_respects_allocation(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify solution respects allocation constraints."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        allocation = {"m0": 1, "m1": 3, "m2": 2}

        solution = solver.solve_constrained(allocation)
        assert solution is not None

        # Count mixers in solution
        counts = {"m0": 0, "m1": 0, "m2": 0}
        for mixer_name in solution.placement.assignments:
            counts[mixer_name] += 1

        assert counts == allocation

    def test_infeasible_allocation(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify None returned for impossible allocation."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        # Allocation sums to wrong total
        with pytest.raises(ValueError, match="must equal num_layers"):
            solver.solve_constrained({"m0": 1, "m1": 1})  # Sum = 2, need 4

    def test_various_allocations(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """Test solver with various allocation patterns."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        allocations = [
            {"m0": 0, "m1": 0, "m2": 6},  # All one mixer
            {"m0": 6, "m1": 0, "m2": 0},  # All another mixer
            {"m0": 3, "m1": 2, "m2": 1},  # Uneven split
            {"m0": 1, "m1": 1, "m2": 4},  # Mostly one mixer
        ]

        for allocation in allocations:
            dp_solution = solver.solve_constrained(allocation)
            assert dp_solution is not None

            # Verify via exhaustive
            all_constrained = solver.enumerate_constrained(allocation)
            if all_constrained:  # Some allocations might have no valid placements
                best_exhaustive = all_constrained[0]
                assert abs(dp_solution.true_score - best_exhaustive[1]) < 1e-10


# =============================================================================
# Test True Score vs Noisy Score
# =============================================================================


class TestTrueVsNoisyScore:
    """Tests comparing true score to noisy evaluator score."""

    @pytest.mark.asyncio
    async def test_true_score_differs_from_noisy(
        self, noisy_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify true score differs from noisy evaluator score."""
        placement = LayerPlacement(
            assignments=("m0", "m0", "m1", "m1", "m0", "m0", "m1", "m1", "m0", "m0")
        )

        true_score = await noisy_evaluator.evaluate(placement, noise_sigma=0.0)
        noisy_score = await noisy_evaluator.evaluate(placement)

        # With noise_sigma=0.5, scores should differ
        # (deterministic noise means they'll be consistently different)
        assert true_score != noisy_score

    @pytest.mark.asyncio
    async def test_optimal_has_best_true_score(
        self, noisy_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify optimal placement has highest true score."""
        model = noisy_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        optimal = solver.solve_unconstrained()

        # Sample some random placements
        import random

        rng = random.Random(42)
        for _ in range(10):
            assignments = tuple(
                model.mixer_names[rng.randint(0, 1)] for _ in range(10)
            )
            placement = LayerPlacement(assignments=assignments)
            score = model.true_score(placement)
            # Use tolerance for floating-point comparison
            assert score <= optimal.true_score + 1e-9


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_layer(self) -> None:
        """Test with single layer."""
        config = MockSupernetEvaluatorConfig(
            num_layers=1,
            mixer_names=["m0", "m1"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.0,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        evaluator = MockSupernetEvaluator(config)
        model = evaluator.mrf_model
        solver = ExactMRFSolver(model)

        solution = solver.solve_unconstrained()
        assert len(solution.placement.assignments) == 1

    def test_single_mixer(self) -> None:
        """Test with single mixer (trivial case)."""
        config = MockSupernetEvaluatorConfig(
            num_layers=5,
            mixer_names=["m0"],
            seed=42,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            noise_sigma=0.0,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        evaluator = MockSupernetEvaluator(config)
        model = evaluator.mrf_model
        solver = ExactMRFSolver(model)

        solution = solver.solve_unconstrained()
        assert all(m == "m0" for m in solution.placement.assignments)

    def test_all_same_preference(self) -> None:
        """Test when all layers prefer the same mixer."""
        # Create evaluator where all layers prefer m0
        config = MockSupernetEvaluatorConfig(
            num_layers=5,
            mixer_names=["m0", "m1"],
            seed=0,  # Seed that happens to give same preferences
            unary_bonus=10.0,  # Very strong unary preference
            unary_penalty=0.0,
            pairwise_scale=0.0,  # No pairwise interaction
            noise_sigma=0.0,
            noise_seed_shift=0,
            eval_delay_seconds=0.0,
            pairwise_decay=0.5,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        evaluator = MockSupernetEvaluator(config)
        model = evaluator.mrf_model
        solver = ExactMRFSolver(model)

        solution = solver.solve_unconstrained()

        # Solution should match layer needs exactly (both are now strings)
        for i, mixer_name in enumerate(solution.placement.assignments):
            assert mixer_name == model.layer_needs[i]


# =============================================================================
# Extended Model Fixtures
# =============================================================================


@pytest.fixture
def skip_pairwise_model() -> ChainMRFModel:
    """Create a model with skip-pairwise interactions (N=4, K=2, range=2)."""
    return ChainMRFModel(
        mixer_names=("m0", "m1"),
        seed=42,
        num_layers=4,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        pairwise_max_range=2,
    )


@pytest.fixture
def triplet_model() -> ChainMRFModel:
    """Create a model with triplet interactions (N=4, K=2)."""
    return ChainMRFModel(
        mixer_names=("m0", "m1"),
        seed=42,
        num_layers=4,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        triplet_scale=0.2,
    )


@pytest.fixture
def combined_model() -> ChainMRFModel:
    """Create a model with both skip-pairwise and triplet interactions (N=4, K=2)."""
    return ChainMRFModel(
        mixer_names=("m0", "m1"),
        seed=42,
        num_layers=4,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
        pairwise_max_range=2,
        triplet_scale=0.2,
    )


# =============================================================================
# Test Extended ChainMRFModel
# =============================================================================


class TestExtendedChainMRFModel:
    """Tests for extended ChainMRFModel with higher-order interactions."""

    def test_memory_baseline(self) -> None:
        """Baseline model (no extensions) has memory=1."""
        model = ChainMRFModel(
            mixer_names=("m0", "m1"),
            seed=42,
            num_layers=4,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
        )
        assert model.memory == 1

    def test_memory_skip_pairwise(self, skip_pairwise_model: ChainMRFModel) -> None:
        """Skip-pairwise range=2 gives memory=2."""
        assert skip_pairwise_model.memory == 2

    def test_memory_triplet(self, triplet_model: ChainMRFModel) -> None:
        """Triplet interactions give memory=2."""
        assert triplet_model.memory == 2

    def test_memory_combined(self, combined_model: ChainMRFModel) -> None:
        """Combined model has memory=2."""
        assert combined_model.memory == 2

    def test_memory_large_range(self) -> None:
        """Skip-pairwise range=3 gives memory=3."""
        model = ChainMRFModel(
            mixer_names=("m0", "m1"),
            seed=42,
            num_layers=6,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            pairwise_max_range=3,
        )
        assert model.memory == 3

    def test_baseline_defaults(self) -> None:
        """Default construction without extensions matches baseline behavior."""
        baseline = ChainMRFModel(
            mixer_names=("m0", "m1"),
            seed=42,
            num_layers=4,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
        )
        extended = ChainMRFModel(
            mixer_names=("m0", "m1"),
            seed=42,
            num_layers=4,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            pairwise_max_range=1,
            triplet_scale=0.0,
        )
        assert baseline.layer_needs == extended.layer_needs
        assert baseline.pairwise_weights == extended.pairwise_weights
        assert extended.pairwise_max_range == 1
        assert extended.triplet_scale == 0.0

    def test_skip_pairwise_nonzero(self, skip_pairwise_model: ChainMRFModel) -> None:
        """Skip-pairwise potentials contribute nonzero values."""
        p1 = LayerPlacement(assignments=("m0", "m0", "m0", "m0"))
        p2 = LayerPlacement(assignments=("m0", "m0", "m1", "m0"))
        # These differ only in layer 2; skip-pairwise between layer 0 and 2 should differ
        s1 = skip_pairwise_model.true_score(p1)
        s2 = skip_pairwise_model.true_score(p2)
        assert s1 != s2

    def test_triplet_nonzero(self, triplet_model: ChainMRFModel) -> None:
        """Triplet potentials contribute nonzero values."""
        p1 = LayerPlacement(assignments=("m0", "m0", "m0", "m0"))
        p2 = LayerPlacement(assignments=("m0", "m0", "m0", "m1"))
        # Changing layer 3 affects triplet (1,2,3) but not (0,1,2)
        s1 = triplet_model.true_score(p1)
        s2 = triplet_model.true_score(p2)
        assert s1 != s2

    def test_true_score_includes_all_terms(self, combined_model: ChainMRFModel) -> None:
        """true_score includes unary + pairwise + skip-pairwise + triplet terms."""
        placement = LayerPlacement(assignments=("m0", "m1", "m0", "m1"))
        score = combined_model.true_score(placement)

        # Compute expected from individual terms
        expected = 0.0
        for i in range(4):
            expected += combined_model.unary_potential(i, placement.assignments[i])
        for i in range(3):
            expected += combined_model.pairwise_potential(
                i, placement.assignments[i], placement.assignments[i + 1]
            )
        for r in range(2, combined_model.pairwise_max_range + 1):
            for i in range(4 - r):
                expected += combined_model.skip_pairwise_potential(
                    i, r, placement.assignments[i], placement.assignments[i + r]
                )
        for i in range(2):
            expected += combined_model.triplet_potential(
                i,
                placement.assignments[i],
                placement.assignments[i + 1],
                placement.assignments[i + 2],
            )

        assert abs(score - expected) < 1e-10


# =============================================================================
# Test Extended Solver - Unconstrained
# =============================================================================


class TestExtendedSolverUnconstrained:
    """Tests for unconstrained optimization with extended models."""

    def test_skip_pairwise_matches_exhaustive(
        self, skip_pairwise_model: ChainMRFModel
    ) -> None:
        """DP matches exhaustive for skip-pairwise model."""
        solver = ExactMRFSolver(skip_pairwise_model)
        dp = solver.solve_unconstrained()
        exhaustive = solver.enumerate_all()
        assert abs(dp.true_score - exhaustive[0][1]) < 1e-10
        assert dp.placement == exhaustive[0][0]

    def test_triplet_matches_exhaustive(
        self, triplet_model: ChainMRFModel
    ) -> None:
        """DP matches exhaustive for triplet model."""
        solver = ExactMRFSolver(triplet_model)
        dp = solver.solve_unconstrained()
        exhaustive = solver.enumerate_all()
        assert abs(dp.true_score - exhaustive[0][1]) < 1e-10
        assert dp.placement == exhaustive[0][0]

    def test_combined_matches_exhaustive(
        self, combined_model: ChainMRFModel
    ) -> None:
        """DP matches exhaustive for combined model."""
        solver = ExactMRFSolver(combined_model)
        dp = solver.solve_unconstrained()
        exhaustive = solver.enumerate_all()
        assert abs(dp.true_score - exhaustive[0][1]) < 1e-10
        assert dp.placement == exhaustive[0][0]

    def test_skip_pairwise_score_recomputed(
        self, skip_pairwise_model: ChainMRFModel
    ) -> None:
        """Solution score matches recomputed score for skip-pairwise model."""
        solver = ExactMRFSolver(skip_pairwise_model)
        dp = solver.solve_unconstrained()
        recomputed = skip_pairwise_model.true_score(dp.placement)
        assert abs(dp.true_score - recomputed) < 1e-10

    def test_triplet_score_recomputed(
        self, triplet_model: ChainMRFModel
    ) -> None:
        """Solution score matches recomputed score for triplet model."""
        solver = ExactMRFSolver(triplet_model)
        dp = solver.solve_unconstrained()
        recomputed = triplet_model.true_score(dp.placement)
        assert abs(dp.true_score - recomputed) < 1e-10

    def test_combined_score_recomputed(
        self, combined_model: ChainMRFModel
    ) -> None:
        """Solution score matches recomputed score for combined model."""
        solver = ExactMRFSolver(combined_model)
        dp = solver.solve_unconstrained()
        recomputed = combined_model.true_score(dp.placement)
        assert abs(dp.true_score - recomputed) < 1e-10


# =============================================================================
# Test Extended Solver - Constrained
# =============================================================================


class TestExtendedSolverConstrained:
    """Tests for constrained optimization with extended models."""

    def test_skip_pairwise_matches_exhaustive(
        self, skip_pairwise_model: ChainMRFModel
    ) -> None:
        """Constrained DP matches exhaustive for skip-pairwise model."""
        solver = ExactMRFSolver(skip_pairwise_model)
        allocation = {"m0": 2, "m1": 2}
        dp = solver.solve_constrained(allocation)
        assert dp is not None
        exhaustive = solver.enumerate_constrained(allocation)
        assert abs(dp.true_score - exhaustive[0][1]) < 1e-10
        assert dp.placement == exhaustive[0][0]

    def test_triplet_matches_exhaustive(
        self, triplet_model: ChainMRFModel
    ) -> None:
        """Constrained DP matches exhaustive for triplet model."""
        solver = ExactMRFSolver(triplet_model)
        allocation = {"m0": 2, "m1": 2}
        dp = solver.solve_constrained(allocation)
        assert dp is not None
        exhaustive = solver.enumerate_constrained(allocation)
        assert abs(dp.true_score - exhaustive[0][1]) < 1e-10
        assert dp.placement == exhaustive[0][0]

    def test_combined_matches_exhaustive(
        self, combined_model: ChainMRFModel
    ) -> None:
        """Constrained DP matches exhaustive for combined model."""
        solver = ExactMRFSolver(combined_model)
        allocation = {"m0": 2, "m1": 2}
        dp = solver.solve_constrained(allocation)
        assert dp is not None
        exhaustive = solver.enumerate_constrained(allocation)
        assert abs(dp.true_score - exhaustive[0][1]) < 1e-10
        assert dp.placement == exhaustive[0][0]

    def test_skip_pairwise_respects_allocation(
        self, skip_pairwise_model: ChainMRFModel
    ) -> None:
        """Solution respects allocation for skip-pairwise model."""
        solver = ExactMRFSolver(skip_pairwise_model)
        allocation = {"m0": 1, "m1": 3}
        dp = solver.solve_constrained(allocation)
        assert dp is not None
        counts = {"m0": 0, "m1": 0}
        for name in dp.placement.assignments:
            counts[name] += 1
        assert counts == allocation

    def test_triplet_respects_allocation(
        self, triplet_model: ChainMRFModel
    ) -> None:
        """Solution respects allocation for triplet model."""
        solver = ExactMRFSolver(triplet_model)
        allocation = {"m0": 3, "m1": 1}
        dp = solver.solve_constrained(allocation)
        assert dp is not None
        counts = {"m0": 0, "m1": 0}
        for name in dp.placement.assignments:
            counts[name] += 1
        assert counts == allocation

    def test_combined_respects_allocation(
        self, combined_model: ChainMRFModel
    ) -> None:
        """Solution respects allocation for combined model."""
        solver = ExactMRFSolver(combined_model)
        allocation = {"m0": 1, "m1": 3}
        dp = solver.solve_constrained(allocation)
        assert dp is not None
        counts = {"m0": 0, "m1": 0}
        for name in dp.placement.assignments:
            counts[name] += 1
        assert counts == allocation


# =============================================================================
# Test solve_top_n
# =============================================================================


class TestSolveTopN:
    """Tests for globally top-N placement solver."""

    def test_baseline_top1_matches_exhaustive(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """solve_top_n top-1 matches exhaustive optimum (N=4, K=2)."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        N = model.num_layers
        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]

        # Exhaustive best
        all_scored = []
        for alloc in all_allocs:
            all_scored.extend(solver.enumerate_constrained(alloc))
        all_scored.sort(key=lambda x: x[1], reverse=True)

        top_n = solver.solve_top_n(all_allocs, n=10)

        # Top-1 must be the global optimum
        assert len(top_n) > 0
        assert abs(top_n[0].true_score - all_scored[0][1]) < 1e-10
        assert top_n[0].placement == all_scored[0][0]

        # Each returned placement's score must match its true_score
        for sol in top_n:
            assert abs(sol.true_score - model.true_score(sol.placement)) < 1e-10

    def test_baseline_matches_exhaustive_medium(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """solve_top_n top-1 matches exhaustive for medium model (N=6, K=3).

        With K=3, m=1: up to 3 history states per allocation, more diversity.
        """
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        all_allocs = [
            {"m0": a, "m1": b, "m2": 6 - a - b}
            for a in range(7) for b in range(7 - a)
        ]

        # Exhaustive best
        all_scored = []
        for alloc in all_allocs:
            all_scored.extend(solver.enumerate_constrained(alloc))
        all_scored.sort(key=lambda x: x[1], reverse=True)

        top_n = solver.solve_top_n(all_allocs, n=20)
        assert len(top_n) == 20

        # Top-1 must match
        assert abs(top_n[0].true_score - all_scored[0][1]) < 1e-10
        assert top_n[0].placement == all_scored[0][0]

        # All returned scores must be correct
        for sol in top_n:
            assert abs(sol.true_score - model.true_score(sol.placement)) < 1e-10

    def test_skip_pairwise_top1_matches(
        self, skip_pairwise_model: ChainMRFModel
    ) -> None:
        """solve_top_n top-1 matches exhaustive for skip-pairwise model."""
        solver = ExactMRFSolver(skip_pairwise_model)
        N = skip_pairwise_model.num_layers
        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]

        all_scored = []
        for alloc in all_allocs:
            all_scored.extend(solver.enumerate_constrained(alloc))
        all_scored.sort(key=lambda x: x[1], reverse=True)

        top_n = solver.solve_top_n(all_allocs, n=10)
        assert len(top_n) > 0
        assert abs(top_n[0].true_score - all_scored[0][1]) < 1e-10

        for sol in top_n:
            assert abs(sol.true_score - skip_pairwise_model.true_score(sol.placement)) < 1e-10

    def test_triplet_top1_matches(
        self, triplet_model: ChainMRFModel
    ) -> None:
        """solve_top_n top-1 matches exhaustive for triplet model."""
        solver = ExactMRFSolver(triplet_model)
        N = triplet_model.num_layers
        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]

        all_scored = []
        for alloc in all_allocs:
            all_scored.extend(solver.enumerate_constrained(alloc))
        all_scored.sort(key=lambda x: x[1], reverse=True)

        top_n = solver.solve_top_n(all_allocs, n=10)
        assert len(top_n) > 0
        assert abs(top_n[0].true_score - all_scored[0][1]) < 1e-10

        for sol in top_n:
            assert abs(sol.true_score - triplet_model.true_score(sol.placement)) < 1e-10

    def test_combined_top1_matches(
        self, combined_model: ChainMRFModel
    ) -> None:
        """solve_top_n top-1 matches exhaustive for combined model."""
        solver = ExactMRFSolver(combined_model)
        N = combined_model.num_layers
        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]

        all_scored = []
        for alloc in all_allocs:
            all_scored.extend(solver.enumerate_constrained(alloc))
        all_scored.sort(key=lambda x: x[1], reverse=True)

        top_n = solver.solve_top_n(all_allocs, n=10)
        assert len(top_n) > 0
        assert abs(top_n[0].true_score - all_scored[0][1]) < 1e-10

        for sol in top_n:
            assert abs(sol.true_score - combined_model.true_score(sol.placement)) < 1e-10

    def test_intra_allocation_diversity(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify some allocations appear multiple times with different arrangements."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        # Single allocation with many possible arrangements
        allocation = {"m0": 2, "m1": 2, "m2": 2}
        top_n = solver.solve_top_n([allocation], n=20)

        # All should have the same allocation
        for sol in top_n:
            counts = dict(Counter(sol.placement.assignments))
            assert counts == allocation

        # But different arrangements
        unique = {sol.placement.assignments for sol in top_n}
        assert len(unique) == len(top_n), "All returned solutions should be distinct"
        assert len(unique) > 1, "Should have multiple arrangements for one allocation"

    def test_score_matches_true_score(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """Each returned solution's score matches model.true_score(placement)."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        N = model.num_layers
        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]

        top_n = solver.solve_top_n(all_allocs, n=10)
        for sol in top_n:
            recomputed = model.true_score(sol.placement)
            assert abs(sol.true_score - recomputed) < 1e-10

    def test_sorted_descending(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """Results should be sorted by score descending."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        N = model.num_layers
        all_allocs = [
            {"m0": a, "m1": b, "m2": N - a - b}
            for a in range(N + 1) for b in range(N + 1 - a)
        ]

        top_n = solver.solve_top_n(all_allocs, n=30)
        for i in range(len(top_n) - 1):
            assert top_n[i].true_score >= top_n[i + 1].true_score

    def test_empty_feasible(self) -> None:
        """Empty feasible allocations returns empty list."""
        model = ChainMRFModel(
            mixer_names=("m0", "m1"),
            seed=42,
            num_layers=4,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
        )
        solver = ExactMRFSolver(model)
        assert solver.solve_top_n([], n=10) == []

    def test_n_larger_than_candidates(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """When n > total candidates, return all available."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        # Only one allocation
        top_n = solver.solve_top_n([{"m0": 2, "m1": 2}], n=1000)
        # N=4, K=2, allocation (2,2) has C(4,2)=6 placements, with m=1 up to K=2 history states
        assert len(top_n) <= 1000
        assert len(top_n) > 0

    def test_subset_of_allocations(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify that restricting feasible allocations filters correctly."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)

        restricted = [{"m0": 3, "m1": 3, "m2": 0}]
        top_n = solver.solve_top_n(restricted, n=10)

        for sol in top_n:
            counts = dict(Counter(sol.placement.assignments))
            # m2 might not appear in Counter — fill in zeros
            for name in model.mixer_names:
                counts.setdefault(name, 0)
            assert counts == {"m0": 3, "m1": 3, "m2": 0}


# =============================================================================
# Test solve_top_n_per_allocation
# =============================================================================


class TestSolveTopNPerAllocation:
    """Tests for per-allocation top-N solver (single DP build)."""

    def _all_allocs_small(self, model: ChainMRFModel) -> list[dict[str, int]]:
        """All valid allocations for a 2-mixer model."""
        N = model.num_layers
        return [{"m0": i, "m1": N - i} for i in range(N + 1)]

    def _all_allocs_medium(self, model: ChainMRFModel) -> list[dict[str, int]]:
        """All valid allocations for a 3-mixer model."""
        N = model.num_layers
        return [
            {"m0": a, "m1": b, "m2": N - a - b}
            for a in range(N + 1)
            for b in range(N + 1 - a)
        ]

    def test_per_allocation_equivalence_small(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """Per-allocation scores match those from solve_top_n([alloc], n) loop."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_small(model)
        names = model.mixer_names
        n = 3

        # Single-pass result
        per_alloc = solver.solve_top_n_per_allocation(all_allocs, n_per_allocation=n)

        # Loop reference
        for alloc in all_allocs:
            counts_tuple = tuple(alloc.get(name, 0) for name in names)
            loop_sols = solver.solve_top_n([alloc], n=n)
            loop_scores = sorted([s.true_score for s in loop_sols], reverse=True)
            pa_sols = per_alloc.get(counts_tuple, [])
            pa_scores = [s.true_score for s in pa_sols]
            assert len(pa_scores) == len(loop_scores)
            for a, b in zip(pa_scores, loop_scores):
                assert abs(a - b) < 1e-10

    def test_per_allocation_equivalence_medium(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """Per-allocation scores match loop reference for 3-mixer model."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_medium(model)
        names = model.mixer_names
        n = 3

        per_alloc = solver.solve_top_n_per_allocation(all_allocs, n_per_allocation=n)

        for alloc in all_allocs:
            counts_tuple = tuple(alloc.get(name, 0) for name in names)
            loop_sols = solver.solve_top_n([alloc], n=n)
            loop_scores = sorted([s.true_score for s in loop_sols], reverse=True)
            pa_sols = per_alloc.get(counts_tuple, [])
            pa_scores = [s.true_score for s in pa_sols]
            assert len(pa_scores) == len(loop_scores)
            for a, b in zip(pa_scores, loop_scores):
                assert abs(a - b) < 1e-10

    def test_score_correctness(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """Every returned solution's true_score matches model.true_score."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_small(model)

        per_alloc = solver.solve_top_n_per_allocation(all_allocs, n_per_allocation=5)

        for sols in per_alloc.values():
            for sol in sols:
                recomputed = model.true_score(sol.placement)
                assert abs(sol.true_score - recomputed) < 1e-10

    def test_allocation_correctness(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """Every solution's actual mixer counts match its counts-key."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_medium(model)
        names = model.mixer_names

        per_alloc = solver.solve_top_n_per_allocation(all_allocs, n_per_allocation=3)

        for counts_tuple, sols in per_alloc.items():
            for sol in sols:
                actual_counts = tuple(
                    sol.placement.assignments.count(name) for name in names
                )
                assert actual_counts == counts_tuple

    def test_per_allocation_cap(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """No group exceeds n_per_allocation."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_medium(model)
        n = 2

        per_alloc = solver.solve_top_n_per_allocation(
            all_allocs, n_per_allocation=n
        )

        for sols in per_alloc.values():
            assert len(sols) <= n

    def test_n_per_allocation_one(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """n_per_allocation=1 returns only the best per allocation."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_small(model)
        names = model.mixer_names

        per_alloc = solver.solve_top_n_per_allocation(
            all_allocs, n_per_allocation=1
        )

        for counts_tuple, sols in per_alloc.items():
            assert len(sols) == 1
            # Verify it really is the best for this allocation
            alloc = {names[k]: counts_tuple[k] for k in range(len(names))}
            exhaustive = solver.enumerate_constrained(alloc)
            assert abs(sols[0].true_score - exhaustive[0][1]) < 1e-10

    def test_fewer_candidates_than_n(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """When allocation has fewer candidates than n, return all available."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        # All-m0 allocation: only 1 possible placement
        alloc = [{"m0": model.num_layers, "m1": 0}]

        per_alloc = solver.solve_top_n_per_allocation(alloc, n_per_allocation=100)
        names = model.mixer_names
        counts_tuple = tuple(alloc[0].get(name, 0) for name in names)
        sols = per_alloc.get(counts_tuple, [])
        # Only 1 possible placement for all-m0
        assert len(sols) == 1

    def test_sorted_descending_per_allocation(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """Results within each allocation are sorted by score descending."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_medium(model)

        per_alloc = solver.solve_top_n_per_allocation(
            all_allocs, n_per_allocation=10
        )

        for sols in per_alloc.values():
            for i in range(len(sols) - 1):
                assert sols[i].true_score >= sols[i + 1].true_score

    def test_single_dp_build(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """Verify _build_constrained_tables is called exactly once."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_small(model)

        with patch.object(
            solver, "_build_constrained_tables", wraps=solver._build_constrained_tables
        ) as mock_build:
            solver.solve_top_n_per_allocation(all_allocs, n_per_allocation=3)
            assert mock_build.call_count == 1

    def test_early_abort_fallback(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """With max_states_per_layer=0, fallback produces correct results."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_small(model)
        names = model.mixer_names
        n = 3

        # Force fallback by setting threshold to 0
        per_alloc = solver.solve_top_n_per_allocation(
            all_allocs, n_per_allocation=n, max_states_per_layer=0
        )

        # Verify correctness against loop reference
        for alloc in all_allocs:
            counts_tuple = tuple(alloc.get(name, 0) for name in names)
            loop_sols = solver.solve_top_n([alloc], n=n)
            loop_scores = sorted([s.true_score for s in loop_sols], reverse=True)
            pa_sols = per_alloc.get(counts_tuple, [])
            pa_scores = [s.true_score for s in pa_sols]
            assert len(pa_scores) == len(loop_scores)
            for a, b in zip(pa_scores, loop_scores):
                assert abs(a - b) < 1e-10

    def test_early_abort_returns_none(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """_build_constrained_tables returns None when threshold is 0."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_small(model)

        result = solver._build_constrained_tables(
            all_allocs, max_states_per_layer=0
        )
        assert result is None

    def test_existing_solve_top_n_unchanged(
        self, medium_evaluator: MockSupernetEvaluator
    ) -> None:
        """solve_top_n still works correctly after refactoring."""
        model = medium_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        all_allocs = self._all_allocs_medium(model)

        # Exhaustive reference
        all_scored = []
        for alloc in all_allocs:
            all_scored.extend(solver.enumerate_constrained(alloc))
        all_scored.sort(key=lambda x: x[1], reverse=True)

        top_n = solver.solve_top_n(all_allocs, n=20)
        assert len(top_n) == 20
        assert abs(top_n[0].true_score - all_scored[0][1]) < 1e-10
        for sol in top_n:
            assert abs(sol.true_score - model.true_score(sol.placement)) < 1e-10

    def test_skip_pairwise_per_allocation(
        self, skip_pairwise_model: ChainMRFModel
    ) -> None:
        """Per-allocation works with skip-pairwise model."""
        solver = ExactMRFSolver(skip_pairwise_model)
        N = skip_pairwise_model.num_layers
        names = skip_pairwise_model.mixer_names
        all_allocs = [{"m0": i, "m1": N - i} for i in range(N + 1)]
        n = 3

        per_alloc = solver.solve_top_n_per_allocation(all_allocs, n_per_allocation=n)

        for counts_tuple, sols in per_alloc.items():
            for sol in sols:
                recomputed = skip_pairwise_model.true_score(sol.placement)
                assert abs(sol.true_score - recomputed) < 1e-10
                actual_counts = tuple(
                    sol.placement.assignments.count(name) for name in names
                )
                assert actual_counts == counts_tuple

    def test_empty_allocations(self) -> None:
        """Empty allocations returns empty dict."""
        model = ChainMRFModel(
            mixer_names=("m0", "m1"),
            seed=42,
            num_layers=4,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
        )
        solver = ExactMRFSolver(model)
        result = solver.solve_top_n_per_allocation([], n_per_allocation=5)
        assert result == {}

    def test_duplicate_allocations_collapse(
        self, small_evaluator: MockSupernetEvaluator
    ) -> None:
        """Duplicate allocations in input are collapsed to same counts key."""
        model = small_evaluator.mrf_model
        solver = ExactMRFSolver(model)
        # Same allocation listed twice
        allocs = [{"m0": 2, "m1": 2}, {"m0": 2, "m1": 2}]

        per_alloc = solver.solve_top_n_per_allocation(allocs, n_per_allocation=5)
        assert (2, 2) in per_alloc
        # Should have the same results as a single allocation
        single = solver.solve_top_n_per_allocation(
            [{"m0": 2, "m1": 2}], n_per_allocation=5
        )
        assert len(per_alloc[(2, 2)]) == len(single[(2, 2)])


# =============================================================================
# K=4 Bisection Tests
# =============================================================================


@pytest.fixture
def k4_model() -> ChainMRFModel:
    """Create a K=4 model (N=8, K=4) for bisection tests.

    Small enough for exhaustive verification (4^8 = 65K placements),
    generates 165 unique allocations.
    """
    return ChainMRFModel(
        mixer_names=("m0", "m1", "m2", "m3"),
        seed=42,
        num_layers=8,
        unary_bonus=1.0,
        unary_penalty=0.5,
        pairwise_scale=0.3,
    )


def _all_allocs_k4(model: ChainMRFModel) -> list[dict[str, int]]:
    """All valid allocations for a 4-mixer, 8-layer model."""
    N = model.num_layers
    names = model.mixer_names
    allocs = []
    for a in range(N + 1):
        for b in range(N + 1 - a):
            for c in range(N + 1 - a - b):
                d = N - a - b - c
                allocs.append({names[0]: a, names[1]: b, names[2]: c, names[3]: d})
    return allocs


class TestK4Bisection:
    """Tests for K=4 bisection fallback in solve_top_n_per_allocation."""

    def test_k4_bisection_matches_loop(self, k4_model: ChainMRFModel) -> None:
        """Force bisection (max_states_per_layer=1), verify per-allocation
        scores match solve_top_n([alloc], n) for all allocations."""
        solver = ExactMRFSolver(k4_model)
        all_allocs = _all_allocs_k4(k4_model)
        names = k4_model.mixer_names
        n = 1

        # Force bisection by setting threshold to 1
        per_alloc = solver.solve_top_n_per_allocation(
            all_allocs, n_per_allocation=n, max_states_per_layer=1
        )

        # Verify against per-allocation loop reference
        for alloc in all_allocs:
            counts_tuple = tuple(alloc.get(name, 0) for name in names)
            loop_sols = solver.solve_top_n([alloc], n=n)
            loop_scores = sorted([s.true_score for s in loop_sols], reverse=True)
            pa_sols = per_alloc.get(counts_tuple, [])
            pa_scores = [s.true_score for s in pa_sols]
            assert len(pa_scores) == len(loop_scores), (
                f"Mismatch for {counts_tuple}: got {len(pa_scores)}, expected {len(loop_scores)}"
            )
            for a_score, b_score in zip(pa_scores, loop_scores):
                assert abs(a_score - b_score) < 1e-10, (
                    f"Score mismatch for {counts_tuple}: {a_score} != {b_score}"
                )

    def test_k4_bisection_splits(self, k4_model: ChainMRFModel) -> None:
        """Moderate threshold: verify _build_constrained_tables called >1 but <165 times."""
        solver = ExactMRFSolver(k4_model)
        all_allocs = _all_allocs_k4(k4_model)

        with patch.object(
            solver, "_build_constrained_tables", wraps=solver._build_constrained_tables
        ) as mock_build:
            solver.solve_top_n_per_allocation(
                all_allocs, n_per_allocation=1, max_states_per_layer=100
            )
            # Should split into batches: more than 1 call (full batch fails)
            # but fewer than 165 (one per allocation)
            assert mock_build.call_count > 1
            assert mock_build.call_count < len(all_allocs)

    def test_k4_all_identical_allocations(self, k4_model: ChainMRFModel) -> None:
        """10 duplicate allocations, forced bisection, correct dedup."""
        solver = ExactMRFSolver(k4_model)
        alloc = {"m0": 2, "m1": 2, "m2": 2, "m3": 2}
        allocs = [alloc] * 10

        per_alloc = solver.solve_top_n_per_allocation(
            allocs, n_per_allocation=1, max_states_per_layer=1
        )

        # Should have exactly one counts key
        assert len(per_alloc) == 1
        counts_tuple = (2, 2, 2, 2)
        assert counts_tuple in per_alloc

        # Verify score matches direct solve
        reference = solver.solve_top_n([alloc], n=1)
        assert len(per_alloc[counts_tuple]) == 1
        assert abs(per_alloc[counts_tuple][0].true_score - reference[0].true_score) < 1e-10

    def test_k4_single_allocation_bisection(self, k4_model: ChainMRFModel) -> None:
        """Single allocation hits base case correctly under forced bisection."""
        solver = ExactMRFSolver(k4_model)
        alloc = {"m0": 3, "m1": 2, "m2": 2, "m3": 1}
        names = k4_model.mixer_names
        counts_tuple = tuple(alloc.get(name, 0) for name in names)

        per_alloc = solver.solve_top_n_per_allocation(
            [alloc], n_per_allocation=1, max_states_per_layer=1
        )

        assert counts_tuple in per_alloc
        reference = solver.solve_top_n([alloc], n=1)
        assert abs(per_alloc[counts_tuple][0].true_score - reference[0].true_score) < 1e-10


# =============================================================================
# Test _estimate_peak_states
# =============================================================================


class TestEstimatePeakStates:
    """Tests for the inclusion-exclusion peak state estimator."""

    def test_uncapped_matches_stars_and_bars(self) -> None:
        """With max_counts=[N]*K, estimate equals K^m × C(N+K-1, K-1)."""
        K, m, N = 4, 3, 48
        max_counts = [N] * K
        estimate = ExactMRFSolver._estimate_peak_states(K, m, N, max_counts)
        # Stars-and-bars: peak at last layer where compositions = C(N+K-1, K-1)
        expected = (K ** m) * comb(N + K - 1, K - 1)
        assert estimate == expected

    def test_caps_reduce_count(self) -> None:
        """Tight caps produce a strictly lower estimate than uncapped."""
        K, m, N = 4, 3, 16
        uncapped = [N] * K
        capped = [6, 6, 6, 6]  # each mixer capped at 6, sum of caps = 24 > N
        est_uncapped = ExactMRFSolver._estimate_peak_states(K, m, N, uncapped)
        est_capped = ExactMRFSolver._estimate_peak_states(K, m, N, capped)
        assert est_capped < est_uncapped

    def test_estimate_is_upper_bound(self, k4_model: ChainMRFModel) -> None:
        """Estimate >= actual peak states from a real DP build."""
        solver = ExactMRFSolver(k4_model)
        N = k4_model.num_layers
        names = k4_model.mixer_names
        all_allocs = _all_allocs_k4(k4_model)

        # Build tables with no threshold
        tables = solver._build_constrained_tables(all_allocs)
        assert tables is not None

        actual_peak = max(len(v) for v in tables.V)

        # Compute max_counts the same way _build_constrained_tables does
        K = k4_model.num_mixers
        m = k4_model.memory
        feasible_set: set[tuple[int, ...]] = set()
        for alloc in all_allocs:
            target = tuple(alloc.get(name, 0) for name in names)
            if sum(target) == N:
                feasible_set.add(target)
        max_counts = [0] * K
        for target in feasible_set:
            for k in range(K):
                if target[k] > max_counts[k]:
                    max_counts[k] = target[k]

        estimate = ExactMRFSolver._estimate_peak_states(K, m, N, max_counts)
        assert estimate >= actual_peak

    def test_estimate_is_upper_bound_high_memory(self) -> None:
        """Estimate >= actual peak states with m=3 (K^m = 64)."""
        model = ChainMRFModel(
            mixer_names=("m0", "m1", "m2", "m3"),
            seed=42,
            num_layers=16,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            pairwise_max_range=3,
        )
        solver = ExactMRFSolver(model)
        N, K, m = model.num_layers, model.num_mixers, model.memory
        names = model.mixer_names

        all_allocs = []
        for a in range(N + 1):
            for b in range(N + 1 - a):
                for c in range(N + 1 - a - b):
                    d = N - a - b - c
                    all_allocs.append(
                        {names[0]: a, names[1]: b, names[2]: c, names[3]: d}
                    )

        tables = solver._build_constrained_tables(all_allocs)
        assert tables is not None

        actual_peak = max(len(v) for v in tables.V)

        feasible_set: set[tuple[int, ...]] = set()
        for alloc in all_allocs:
            target = tuple(alloc.get(name, 0) for name in names)
            if sum(target) == N:
                feasible_set.add(target)
        max_counts = [0] * K
        for target in feasible_set:
            for k in range(K):
                if target[k] > max_counts[k]:
                    max_counts[k] = target[k]

        estimate = ExactMRFSolver._estimate_peak_states(K, m, N, max_counts)
        assert estimate >= actual_peak

    def test_pre_check_prevents_dp_computation(self) -> None:
        """Threshold just below estimate → None; just above → succeeds."""
        model = ChainMRFModel(
            mixer_names=("m0", "m1", "m2", "m3"),
            seed=42,
            num_layers=16,
            unary_bonus=1.0,
            unary_penalty=0.5,
            pairwise_scale=0.3,
            pairwise_max_range=3,
        )
        solver = ExactMRFSolver(model)
        N = model.num_layers
        names = model.mixer_names
        K = model.num_mixers
        m = model.memory

        all_allocs = []
        for a in range(N + 1):
            for b in range(N + 1 - a):
                for c in range(N + 1 - a - b):
                    d = N - a - b - c
                    all_allocs.append(
                        {names[0]: a, names[1]: b, names[2]: c, names[3]: d}
                    )

        # Compute max_counts
        feasible_set: set[tuple[int, ...]] = set()
        for alloc in all_allocs:
            target = tuple(alloc.get(name, 0) for name in names)
            if sum(target) == N:
                feasible_set.add(target)
        max_counts = [0] * K
        for target in feasible_set:
            for k in range(K):
                if target[k] > max_counts[k]:
                    max_counts[k] = target[k]

        estimate = ExactMRFSolver._estimate_peak_states(K, m, N, max_counts)
        assert estimate > 0

        # Threshold just below estimate → should return None (pre-check blocks)
        result = solver._build_constrained_tables(
            all_allocs, max_states_per_layer=estimate - 1
        )
        assert result is None

        # Threshold at estimate → should succeed (pre-check passes)
        result = solver._build_constrained_tables(
            all_allocs, max_states_per_layer=estimate
        )
        assert result is not None
