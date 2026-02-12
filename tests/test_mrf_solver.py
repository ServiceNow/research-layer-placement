"""Tests for MRF solver correctness.

The core guarantees of the MRF solver are:
1. Unconstrained solver finds the global optimum (verified via exhaustive search)
2. Constrained solver finds the global optimum within allocation constraints
3. True score computation matches the evaluator's score without noise
"""

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
