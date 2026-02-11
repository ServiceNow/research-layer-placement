"""Unit tests for the annealing module building blocks."""

import random
from collections import Counter
from unittest.mock import MagicMock

import numpy as np
import pytest

from place_layers.annealing.events import (
    AnnealingReheatEvent,
    AnnealingRestartEvent,
    AnnealingStepEvent,
)
from place_layers.annealing.moves import random_assignment, type_mutation
from place_layers.annealing.sa_core import (
    ReheatConfig,
    SAConfig,
    TopKTracker,
    metropolis_accept,
    pick_restart_allocation,
    run_multi_start_sa,
)
from place_layers.cost_model.linear_cost_model import LinearCostModelConfig
from place_layers.surrogate.forest_surrogate import ForestSurrogate


pytestmark = pytest.mark.asyncio(loop_scope="function")


# =============================================================================
# metropolis_accept
# =============================================================================


class TestMetropolisAccept:
    def test_improving_always_accepted(self) -> None:
        """Improving moves (delta > 0) are always accepted."""
        rng = random.Random(42)
        for _ in range(100):
            assert metropolis_accept(0.1, 1.0, rng) is True
            assert metropolis_accept(10.0, 0.001, rng) is True

    def test_worsening_rejected_at_zero_temp(self) -> None:
        """Worsening moves (delta < 0) at T=0 are always rejected."""
        rng = random.Random(42)
        for _ in range(100):
            assert metropolis_accept(-0.1, 0.0, rng) is False
            assert metropolis_accept(-10.0, 0.0, rng) is False

    def test_worsening_probabilistic(self) -> None:
        """Worsening moves at T>0 should be accepted stochastically."""
        rng = random.Random(42)
        delta = -0.1
        temperature = 1.0
        # At T=1.0, delta=-0.1 -> P(accept) = exp(-0.1) ~ 0.905
        n_trials = 10000
        accepted = sum(
            metropolis_accept(delta, temperature, rng) for _ in range(n_trials)
        )
        acceptance_rate = accepted / n_trials
        # Should be close to exp(-0.1) ~ 0.905
        assert 0.88 < acceptance_rate < 0.93

    def test_large_delta_low_probability(self) -> None:
        """Large worsening + low T should almost never be accepted."""
        rng = random.Random(42)
        delta = -10.0
        temperature = 0.01
        # P(accept) = exp(-10/0.01) = exp(-1000) ~ 0
        n_trials = 10000
        accepted = sum(
            metropolis_accept(delta, temperature, rng) for _ in range(n_trials)
        )
        assert accepted == 0


# =============================================================================
# TopKTracker
# =============================================================================


class TestTopKTracker:
    def test_add_and_retrieve(self) -> None:
        """Basic add/retrieve flow."""
        tracker = TopKTracker(max_size=5)
        tracker.add(("a", "b"), 1.0)
        tracker.add(("c", "d"), 2.0)
        assert len(tracker) == 2
        candidates = tracker.top_candidates()
        assert len(candidates) == 2
        assert candidates[0] == (("c", "d"), 2.0)  # Higher score first
        assert candidates[1] == (("a", "b"), 1.0)

    def test_bounded_size(self) -> None:
        """Tracker never exceeds max_size."""
        tracker = TopKTracker(max_size=3)
        for i in range(10):
            tracker.add((str(i),), float(i))
        assert len(tracker) == 3

    def test_deduplication(self) -> None:
        """Same candidate is not added twice."""
        tracker = TopKTracker(max_size=5)
        tracker.add(("a", "b"), 1.0)
        tracker.add(("a", "b"), 2.0)  # Same candidate, different score
        assert len(tracker) == 1

    def test_evicts_worst(self) -> None:
        """Worst score is evicted when full."""
        tracker = TopKTracker(max_size=3)
        tracker.add(("a",), 1.0)
        tracker.add(("b",), 2.0)
        tracker.add(("c",), 3.0)
        # Now full with scores 1, 2, 3
        tracker.add(("d",), 4.0)
        # Should have 2, 3, 4 (evicted 1)
        assert len(tracker) == 3
        candidates = tracker.top_candidates()
        candidate_tuples = [c for c, _s in candidates]
        assert ("a",) not in candidate_tuples
        assert ("d",) in candidate_tuples

    def test_top_candidates_sorted_desc(self) -> None:
        """top_candidates returns descending order by score."""
        tracker = TopKTracker(max_size=5)
        tracker.add(("c",), 3.0)
        tracker.add(("a",), 1.0)
        tracker.add(("b",), 2.0)
        candidates = tracker.top_candidates()
        assert candidates == [(("c",), 3.0), (("b",), 2.0), (("a",), 1.0)]


# =============================================================================
# type_mutation (moved from test_surrogate_search.py)
# =============================================================================


class TestTypeMutation:
    def test_one_position_changed(self) -> None:
        """type_mutation should change exactly one position to a different type."""
        rng = random.Random(42)
        assignments = ["m0", "m1", "m0", "m1", "m0"]
        mixer_names = ["m0", "m1"]
        for _ in range(20):
            proposed = type_mutation(rng, assignments, mixer_names)
            assert len(proposed) == len(assignments)
            diffs = [i for i in range(len(assignments)) if proposed[i] != assignments[i]]
            assert len(diffs) == 1
            assert proposed[diffs[0]] != assignments[diffs[0]]
            assert proposed[diffs[0]] in mixer_names

    def test_preserves_length(self) -> None:
        """Result should have the same length as input."""
        rng = random.Random(123)
        assignments = ["a", "b", "c", "a", "b"]
        mixer_names = ["a", "b", "c"]
        proposed = type_mutation(rng, assignments, mixer_names)
        assert len(proposed) == 5

    def test_single_mixer_no_change(self) -> None:
        """With only one mixer type, should return a copy unchanged."""
        rng = random.Random(42)
        assignments = ["m0", "m0", "m0"]
        proposed = type_mutation(rng, assignments, ["m0"])
        assert proposed == assignments


# =============================================================================
# run_multi_start_sa
# =============================================================================


def _make_sa_config(**overrides: object) -> SAConfig:
    """Create an SAConfig with sensible test defaults."""
    defaults = dict(
        num_restarts=3,
        steps_per_restart=500,
        initial_temperature=0.1,
        cooling_rate=0.999,
        min_temperature=1e-6,
        mutation_prob=0.3,
        max_ruin_fraction=0.5,
        contiguous_ruin_prob=0.3,
        top_k=50,
        no_op_retries=0,
        reheat=None,
        seed=None,
    )
    defaults.update(overrides)
    return SAConfig(**defaults)


class TestRunMultiStartSA:
    async def test_run_multi_start_sa(self) -> None:
        """run_multi_start_sa should return distinct feasible candidates."""
        rng = random.Random(42)
        mixer_names = ["m0", "m1"]
        num_layers = 8

        allocs = [
            {"m0": 6, "m1": 2},
            {"m0": 4, "m1": 4},
            {"m0": 2, "m1": 6},
        ]
        mixer_costs = {"m0": 1.0, "m1": 0.5}
        cost_model = LinearCostModelConfig(
            cost_model_type="linear", mixer_costs=mixer_costs
        ).create_cost_model()
        cost_budget = 7.0  # allows allocations with cost <= 7.0

        # Filter feasible
        feasible = [a for a in allocs if cost_model.allocation_cost(a) <= cost_budget]
        assert len(feasible) > 0

        # Generate training data
        assignments_list = []
        scores = []
        for _ in range(100):
            alloc = rng.choice(feasible)
            assignment = random_assignment(rng, alloc, num_layers)
            assignments_list.append(tuple(assignment))
            # Simple scoring: count m1 in even positions
            scores.append(sum(1.0 for i, m in enumerate(assignment) if i % 2 == 0 and m == "m1"))

        surrogate = ForestSurrogate(
            n_estimators=20, max_depth=None, seed=42, mixer_names=mixer_names
        )
        surrogate.fit(assignments_list, np.array(scores))

        sa_config = _make_sa_config()

        # Wrap sync surrogate in async
        async def score_fn(assignment: tuple[str, ...]) -> float:
            return surrogate.predict_mean_one(assignment)

        # Run SA
        result = await run_multi_start_sa(
            score_fn=score_fn,
            allocation_cost=cost_model.allocation_cost,
            feasible_allocations=feasible,
            num_layers=num_layers,
            mixer_names=mixer_names,
            cost_budget=cost_budget,
            config=sa_config,
            rng=rng,
        )

        candidates = [c for c, _s in result.candidates]

        # Verify results
        assert len(candidates) > 0
        assert len(candidates) <= 50

        # All candidates should be distinct
        assert len(set(candidates)) == len(candidates)

        # All candidates should have correct length
        for c in candidates:
            assert len(c) == num_layers
            assert all(m in mixer_names for m in c)

        # All candidates should be cost-feasible
        for c in candidates:
            alloc = dict(Counter(c))
            assert cost_model.allocation_cost(alloc) <= cost_budget + 1e-9

    async def test_sa_without_budget(self) -> None:
        """run_multi_start_sa with cost_budget=None should still work."""
        rng = random.Random(42)
        mixer_names = ["m0", "m1"]
        num_layers = 6

        allocs = [{"m0": 3, "m1": 3}, {"m0": 4, "m1": 2}]
        mixer_costs = {"m0": 1.0, "m1": 0.5}
        cost_model = LinearCostModelConfig(
            cost_model_type="linear", mixer_costs=mixer_costs
        ).create_cost_model()

        assignments_list = []
        scores = []
        for _ in range(50):
            alloc = rng.choice(allocs)
            assignment = random_assignment(rng, alloc, num_layers)
            assignments_list.append(tuple(assignment))
            scores.append(rng.random())

        surrogate = ForestSurrogate(
            n_estimators=10, max_depth=None, seed=42, mixer_names=mixer_names
        )
        surrogate.fit(assignments_list, np.array(scores))

        sa_config = _make_sa_config(num_restarts=2, steps_per_restart=200, top_k=30)

        async def score_fn(assignment: tuple[str, ...]) -> float:
            return surrogate.predict_mean_one(assignment)

        result = await run_multi_start_sa(
            score_fn=score_fn,
            allocation_cost=cost_model.allocation_cost,
            feasible_allocations=allocs,
            num_layers=num_layers,
            mixer_names=mixer_names,
            cost_budget=None,
            config=sa_config,
            rng=rng,
        )

        candidates = [c for c, _s in result.candidates]
        assert len(candidates) > 0
        assert len(set(candidates)) == len(candidates)


# =============================================================================
# Reheating
# =============================================================================


class TestReheating:
    async def test_reheating_triggers(self) -> None:
        """SA with ReheatConfig should trigger reheats when acceptance rate drops."""
        rng = random.Random(42)
        num_layers = 6
        allocs = [{"m0": 3, "m1": 3}]

        # Peaked landscape: one target assignment scores high, all others low.
        # Most moves away from the peak are worsening (delta ~ -10).
        # As temperature drops, these get rejected → acceptance rate falls → reheat.
        target = ("m0", "m1", "m0", "m1", "m0", "m1")

        async def peaked_score(assignment: tuple[str, ...]) -> float:
            matches = sum(1 for a, t in zip(assignment, target) if a == t)
            return float(matches) * 2.0  # range 0..12

        sa_config = _make_sa_config(
            num_restarts=1,
            steps_per_restart=500,
            initial_temperature=10.0,
            cooling_rate=0.98,
            min_temperature=1e-6,
            mutation_prob=0.0,
            max_ruin_fraction=0.5,
            no_op_retries=4,
            reheat=ReheatConfig(fraction=0.3, threshold=0.15, window=30),
        )

        result = await run_multi_start_sa(
            score_fn=peaked_score,
            allocation_cost=lambda a: 1.0,
            feasible_allocations=allocs,
            num_layers=num_layers,
            mixer_names=["m0", "m1"],
            cost_budget=None,
            config=sa_config,
            rng=rng,
        )

        # With aggressive cooling on a peaked landscape, reheating should trigger
        assert result.total_reheats > 0


# =============================================================================
# SA Observer
# =============================================================================


class TestSAObserver:
    async def test_observer_called(self) -> None:
        """Observers should receive step, restart, and reheat events."""
        rng = random.Random(42)
        num_layers = 6
        allocs = [{"m0": 3, "m1": 3}]

        async def dummy_score(assignment: tuple[str, ...]) -> float:
            return rng.random()

        observer = MagicMock()
        observer.on_annealing_step = MagicMock()
        observer.on_annealing_restart = MagicMock()
        observer.on_annealing_reheat = MagicMock()

        sa_config = _make_sa_config(
            num_restarts=2,
            steps_per_restart=20,
            initial_temperature=1.0,
            cooling_rate=0.95,
            min_temperature=1e-6,
            mutation_prob=0.0,
            no_op_retries=4,
        )

        await run_multi_start_sa(
            score_fn=dummy_score,
            allocation_cost=lambda a: 1.0,
            feasible_allocations=allocs,
            num_layers=num_layers,
            mixer_names=["m0", "m1"],
            cost_budget=None,
            config=sa_config,
            rng=rng,
            observers=[observer],
        )

        # Restart events: one per restart
        assert observer.on_annealing_restart.call_count == 2
        for call in observer.on_annealing_restart.call_args_list:
            event = call[0][0]
            assert isinstance(event, AnnealingRestartEvent)

        # Step events: should have received some
        assert observer.on_annealing_step.call_count > 0
        for call in observer.on_annealing_step.call_args_list:
            event = call[0][0]
            assert isinstance(event, AnnealingStepEvent)


# =============================================================================
# pick_restart_allocation
# =============================================================================


class TestPickRestartAllocation:
    def test_basic(self) -> None:
        """pick_restart_allocation should return a valid allocation."""
        rng = random.Random(42)
        allocs = [{"m0": 3, "m1": 3}, {"m0": 4, "m1": 2}, {"m0": 5, "m1": 1}]
        mixer_costs = {"m0": 1.0, "m1": 0.5}
        cost_model = LinearCostModelConfig(
            cost_model_type="linear", mixer_costs=mixer_costs
        ).create_cost_model()
        cost_budget = 5.0

        for _ in range(20):
            alloc = pick_restart_allocation(
                allocs, cost_model.allocation_cost, cost_budget, rng
            )
            assert alloc in allocs

    def test_no_budget(self) -> None:
        """Without budget, should return a uniform random allocation."""
        rng = random.Random(42)
        allocs = [{"m0": 3, "m1": 3}, {"m0": 4, "m1": 2}]

        def dummy_cost(a: dict[str, int]) -> float:
            return 0.0

        alloc = pick_restart_allocation(allocs, dummy_cost, None, rng)
        assert alloc in allocs
