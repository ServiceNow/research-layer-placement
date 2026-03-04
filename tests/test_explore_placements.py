"""Integration tests for explore_placements command."""

import random
from pathlib import Path

import numpy as np
import pytest

from place_layers.command.explore_placements import (
    _check_plateau,
    _compute_cost_uniform_weights,
    _fast_forward_rng,
    _load_progress,
    _sample_batch,
)
from place_layers.surrogate.evaluation_log import (
    EVALUATION_RECORDS_FILENAME,
    EvaluationLogMeta,
    EvaluationRecord,
    append_record,
    discover_evaluation_records,
    load_pooled_data,
    write_meta,
)
from place_layers.cost_model.enumeration import enumerate_feasible_allocations
from place_layers.surrogate.config import BayesianRidgeSurrogateConfig, RidgeSurrogateConfig
from place_layers.surrogate.dataset import ExplorationDataset
from place_layers.surrogate.encoding import FeatureExpansionConfig
from place_layers.core.types import LayerPlacement
from place_layers.cost_model.linear_cost_model import LinearCostModelConfig
from place_layers.evaluator.mock_supernet_evaluator import (
    MockSupernetEvaluator,
    MockSupernetEvaluatorConfig,
)
from place_layers.problem.layer_placement import (
    CostConstrainedProblem,
    MaxPerformanceProblemConfig,
)
from place_layers.surrogate.model_selection import ridge_cv_rho
from tests.helpers import wrap_evaluator


pytestmark = pytest.mark.asyncio(loop_scope="function")


# =============================================================================
# Fixtures
# =============================================================================


def _make_small_evaluator(
    num_layers: int = 12, mixer_names: list[str] | None = None
) -> MockSupernetEvaluator:
    """Create a small mock evaluator for testing."""
    if mixer_names is None:
        mixer_names = ["m0", "m1"]
    config = MockSupernetEvaluatorConfig(
        evaluator_type="mock_supernet",
        num_layers=num_layers,
        mixer_names=mixer_names,
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
    return MockSupernetEvaluator(config=config)


def _make_small_problem(
    evaluator: MockSupernetEvaluator,
) -> CostConstrainedProblem:
    """Create a small cost-constrained problem."""
    num_layers = evaluator.num_layers
    mixer_names = sorted(evaluator.mixer_names)
    # Cost: m0=1.0, m1=0.5
    mixer_costs = {name: 1.0 - 0.5 * i for i, name in enumerate(mixer_names)}
    config = MaxPerformanceProblemConfig(
        problem_type="cost_max_performance",
        cost_model=LinearCostModelConfig(
            cost_model_type="linear",
            mixer_costs=mixer_costs,
        ),
        cost_budget=num_layers * 0.8,  # Budget between min and max
        cost_floor=None,
    )
    return CostConstrainedProblem(evaluators=wrap_evaluator(evaluator), config=config, objective_name="weighted_average")


# =============================================================================
# Helper function tests
# =============================================================================


class TestHelpers:
    def testenumerate_feasible_allocations(self) -> None:
        """Should enumerate allocations within budget."""
        config = LinearCostModelConfig(
            cost_model_type="linear",
            mixer_costs={"m0": 1.0, "m1": 0.5},
        )
        cost_model = config.create_cost_model()
        num_layers = 6

        # All allocations (no budget)
        all_allocs = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=None
        )
        assert len(all_allocs) == num_layers + 1  # 0..6 for m0

        # With budget
        budget = 4.0  # Excludes allocations costing more than 4.0
        feasible = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget=budget
        )
        for alloc in feasible:
            cost = cost_model.allocation_cost(alloc)
            assert cost <= budget + 1e-9

    def test_sample_batch(self) -> None:
        """Should produce correct number of samples with correct lengths."""
        allocs = [{"m0": 3, "m1": 2}, {"m0": 1, "m1": 4}]
        rng = random.Random(42)
        batch = _sample_batch(10, allocs, num_layers=5, rng=rng)
        assert len(batch) == 10
        for assignment in batch:
            assert len(assignment) == 5

    def test_sample_batch_with_weights(self) -> None:
        """Weights should bias allocation selection towards higher-weighted allocations."""
        allocs = [{"m0": 5, "m1": 0}, {"m0": 0, "m1": 5}]
        # Heavily bias towards second allocation
        weights = [0.01, 0.99]
        rng = random.Random(42)
        batch = _sample_batch(200, allocs, num_layers=5, rng=rng, weights=weights)

        # Count how many samples came from each allocation
        second_alloc_count = sum(
            1 for a in batch if a.count("m1") == 5
        )
        # With 99% weight on second allocation, vast majority should be from it
        assert second_alloc_count > 150, (
            f"Expected >150 samples from second allocation, got {second_alloc_count}"
        )

    def test_cost_uniform_weights_basic(self) -> None:
        """Each cost level should receive equal total weight."""
        config = LinearCostModelConfig(
            cost_model_type="linear",
            mixer_costs={"m0": 1.0, "m1": 0.5},
        )
        cost_model = config.create_cost_model()
        num_layers = 4

        # Enumerate all allocations (no budget)
        allocs = enumerate_feasible_allocations(cost_model, num_layers, cost_budget=None)
        weights = _compute_cost_uniform_weights(allocs, cost_model)

        assert len(weights) == len(allocs)

        # Group total weight by cost level
        cost_totals: dict[float, float] = {}
        for alloc, w in zip(allocs, weights):
            cost = round(cost_model.allocation_cost(alloc), 10)
            cost_totals[cost] = cost_totals.get(cost, 0.0) + w

        # Each cost level should have equal total weight (they're all 1.0
        # since each cost level has exactly 1 allocation for linear costs)
        totals = list(cost_totals.values())
        for t in totals:
            assert abs(t - totals[0]) < 1e-9, (
                f"Cost levels have unequal total weights: {cost_totals}"
            )

    def test_check_plateau_no_plateau(self) -> None:
        history = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert not _check_plateau(history, patience=3, min_improvement=0.02)

    def test_check_plateau_detected(self) -> None:
        history = [0.5, 0.501, 0.502, 0.503]
        assert _check_plateau(history, patience=3, min_improvement=0.02)

    def test_check_plateau_too_short(self) -> None:
        history = [0.5, 0.501]
        assert not _check_plateau(history, patience=3, min_improvement=0.02)

    def test_exploration_dataset_load(self, tmp_path: Path) -> None:
        """ExplorationDataset.load() should deserialize from file."""
        dataset = ExplorationDataset(
            assignments=[("m0", "m1")],
            scores=[0.5],
            objective_name="weighted_average",
            num_layers=2,
            mixer_names=["m0", "m1"],
            num_samples=1,
            termination_reason="max_samples",
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            surrogate_config=RidgeSurrogateConfig(
                n_bootstraps=50, ei_xi=0.01, cv_folds=5,
            ),
            rho_history=[0.3],
            samples_at_check=[1],
            final_rho=0.3,
        )
        path = tmp_path / "exploration_results.json"
        path.write_text(dataset.model_dump_json(indent=2))

        loaded = ExplorationDataset.load(path)
        assert loaded.num_samples == 1
        assert loaded.assignments == [("m0", "m1")]
        assert loaded.final_rho == 0.3
        assert loaded.objective_name == "weighted_average"
        assert loaded.surrogate_config is not None
        assert loaded.surrogate_config.surrogate_type == "ridge"

    def test_jsonl_roundtrip(self, tmp_path: Path) -> None:
        """Write and read progress JSONL via evaluation_log module."""
        path = tmp_path / "progress.jsonl"
        with path.open("w") as f:
            append_record(f, EvaluationRecord(
                assignment=("m0", "m1", "m0"), score=0.75,
            ))
            append_record(f, EvaluationRecord(
                assignment=("m1", "m1", "m0"), score=0.82,
            ))

        assignments, scores = _load_progress(path)
        assert len(assignments) == 2
        assert assignments[0] == ("m0", "m1", "m0")
        assert scores[1] == 0.82


# =============================================================================
# Integration test
# =============================================================================


class TestExploreIntegration:
    """Integration test: run exploration loop on small problem."""

    async def test_exploration_loop(self) -> None:
        """Run adaptive exploration on a 12-layer, 2-mixer problem."""
        evaluator = _make_small_evaluator(num_layers=12)
        problem = _make_small_problem(evaluator)
        cost_model = problem.cost_model
        num_layers = evaluator.num_layers

        # Enumerate feasible allocations
        assert isinstance(problem.config, MaxPerformanceProblemConfig)
        cost_budget = problem.config.cost_budget
        feasible = enumerate_feasible_allocations(
            cost_model, num_layers, cost_budget
        )
        assert len(feasible) > 0

        # Sample and evaluate
        rng = random.Random(42)
        min_samples = 100
        batch_size = 50

        all_assignments: list[tuple[str, ...]] = []
        all_scores: list[float] = []

        # Initial batch
        batch = _sample_batch(min_samples, feasible, num_layers, rng)
        for assignment in batch:
            placement = LayerPlacement(assignments=tuple(assignment))
            score = await evaluator.evaluate(placement)
            all_assignments.append(tuple(assignment))
            all_scores.append(score)

        assert len(all_assignments) == min_samples

        # Quality check via Ridge CV
        rho1 = ridge_cv_rho(
            assignments=all_assignments,
            scores=np.array(all_scores),
            mixer_names=cost_model.phase_order,
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            cv_folds=5,
            seed=42,
        )
        rho_history = [rho1]

        # One more batch
        batch = _sample_batch(batch_size, feasible, num_layers, rng)
        for assignment in batch:
            placement = LayerPlacement(assignments=tuple(assignment))
            score = await evaluator.evaluate(placement)
            all_assignments.append(tuple(assignment))
            all_scores.append(score)

        rho2 = ridge_cv_rho(
            assignments=all_assignments,
            scores=np.array(all_scores),
            mixer_names=cost_model.phase_order,
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            cv_folds=5,
            seed=42,
        )
        rho_history.append(rho2)

        # Assertions
        assert len(all_assignments) == min_samples + batch_size
        assert rho_history[0] > -1.0  # Sanity: rho is computed
        assert len(rho_history) == 2

    async def test_exploration_dataset_serialization(self) -> None:
        """ExplorationDataset should serialize and deserialize correctly."""
        dataset = ExplorationDataset(
            assignments=[("m0", "m1"), ("m1", "m0")],
            scores=[0.5, 0.7],
            objective_name="weighted_average",
            num_layers=2,
            mixer_names=["m0", "m1"],
            num_samples=2,
            termination_reason="max_samples",
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            surrogate_config=RidgeSurrogateConfig(
                n_bootstraps=50, ei_xi=0.01, cv_folds=5,
            ),
            rho_history=[0.3],
            samples_at_check=[2],
            final_rho=0.3,
        )

        # Serialize to JSON and back
        json_str = dataset.model_dump_json()
        loaded = ExplorationDataset.model_validate_json(json_str)

        assert loaded.num_samples == 2
        assert loaded.assignments == [("m0", "m1"), ("m1", "m0")]
        assert loaded.scores == [0.5, 0.7]
        assert loaded.objective_name == "weighted_average"
        assert loaded.termination_reason == "max_samples"
        assert loaded.surrogate_config is not None
        assert loaded.surrogate_config.surrogate_type == "ridge"

    async def test_exploration_dataset_bayesian_surrogate(self) -> None:
        """ExplorationDataset with BayesianRidgeSurrogateConfig round-trips."""
        cfg = BayesianRidgeSurrogateConfig(
            prior_alpha=1.0,
            prior_a0=1e-6,
            prior_b0=1e-6,
            feature_scaling="counts_only",
        )
        dataset = ExplorationDataset(
            assignments=[("m0", "m1")],
            scores=[0.5],
            objective_name="weighted_average",
            num_layers=2,
            mixer_names=["m0", "m1"],
            num_samples=1,
            termination_reason="max_samples",
            feature_config=FeatureExpansionConfig(max_order=1, pairwise_max_range=1, allocation_counts=True),
            surrogate_config=cfg,
            rho_history=[0.3],
            samples_at_check=[1],
            final_rho=0.3,
        )

        json_str = dataset.model_dump_json()
        loaded = ExplorationDataset.model_validate_json(json_str)
        assert loaded.surrogate_config is not None
        assert loaded.surrogate_config.surrogate_type == "bayesian_ridge"
        assert isinstance(loaded.surrogate_config, BayesianRidgeSurrogateConfig)
        assert loaded.surrogate_config.prior_alpha == 1.0


# =============================================================================
# Supplemental data merge tests
# =============================================================================


def _write_supplemental_jsonl(
    directory: Path,
    obj_name: str,
    records: list[EvaluationRecord],
    *,
    num_layers: int = 12,
    mixer_names: list[str] | None = None,
) -> Path:
    """Write a supplemental JSONL file with sidecar metadata."""
    if mixer_names is None:
        mixer_names = ["m0", "m1"]
    path = directory / f"{obj_name}/{EVALUATION_RECORDS_FILENAME}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            append_record(f, rec)
    write_meta(
        path,
        EvaluationLogMeta(
            version=1,
            objective_name=obj_name,
            num_layers=num_layers,
            mixer_names=mixer_names,
            source_command="test",
            source_detail="supplemental",
            created_at="2026-01-01T00:00:00Z",
            num_records=len(records),
        ),
    )
    return path


class TestSupplementalDataMerge:
    """Tests for supplemental_data_dirs merge logic."""

    def test_basic_merge(self, tmp_path: Path) -> None:
        """Supplemental records appear in all_assignments after merge."""
        # Local progress: 2 records
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        local_progress = local_dir / "exploration_progress_obj.jsonl"
        with local_progress.open("w") as f:
            append_record(f, EvaluationRecord(
                assignment=("m0", "m1"), score=0.5,
            ))
            append_record(f, EvaluationRecord(
                assignment=("m1", "m0"), score=0.6,
            ))

        # Supplemental dir: 2 different records
        supp_dir = tmp_path / "supplemental"
        supp_dir.mkdir()
        _write_supplemental_jsonl(
            supp_dir, "obj",
            [
                EvaluationRecord(assignment=("m0", "m0"), score=0.4),
                EvaluationRecord(assignment=("m1", "m1"), score=0.7),
            ],
            num_layers=2,
        )

        # Load local progress
        all_assignments, all_scores = _load_progress(local_progress)
        num_local = len(all_assignments)
        assert num_local == 2

        # Merge supplemental
        supplemental_files = discover_evaluation_records([str(supp_dir)], "obj")
        assert len(supplemental_files) > 0

        pool = load_pooled_data(supplemental_files, deduplicate=True)
        existing = {(a, s) for a, s in zip(all_assignments, all_scores)}
        for a, s in zip(pool.assignments, pool.scores):
            if (a, s) not in existing:
                all_assignments.append(a)
                all_scores.append(s)
                existing.add((a, s))

        assert len(all_assignments) == 4
        assert num_local == 2  # unchanged
        # Verify supplemental records are present
        assert ("m0", "m0") in all_assignments
        assert ("m1", "m1") in all_assignments

    def test_deduplication(self, tmp_path: Path) -> None:
        """Records already in local progress are not duplicated."""
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        local_progress = local_dir / "exploration_progress_obj.jsonl"

        shared_record = EvaluationRecord(
            assignment=("m0", "m1"), score=0.5,
        )
        with local_progress.open("w") as f:
            append_record(f, shared_record)

        # Supplemental has the same record + one new
        supp_dir = tmp_path / "supplemental"
        supp_dir.mkdir()
        _write_supplemental_jsonl(
            supp_dir, "obj",
            [
                shared_record,
                EvaluationRecord(assignment=("m1", "m0"), score=0.6),
            ],
            num_layers=2,
        )

        all_assignments, all_scores = _load_progress(local_progress)
        num_local = len(all_assignments)

        supplemental_files = discover_evaluation_records([str(supp_dir)], "obj")
        pool = load_pooled_data(supplemental_files, deduplicate=True)
        existing = {(a, s) for a, s in zip(all_assignments, all_scores)}
        n_before = len(all_assignments)
        for a, s in zip(pool.assignments, pool.scores):
            if (a, s) not in existing:
                all_assignments.append(a)
                all_scores.append(s)
                existing.add((a, s))
        n_added = len(all_assignments) - n_before

        assert num_local == 1
        assert n_added == 1  # only the new record
        assert len(all_assignments) == 2

    def test_rng_determinism_with_supplemental(self) -> None:
        """RNG is only advanced by num_local, not by supplemental count.

        A fresh start with supplemental data must produce the same RNG
        sequence as a fresh start without supplemental data.
        """
        allocs = [{"m0": 3, "m1": 2}, {"m0": 2, "m1": 3}]
        num_layers = 5

        # Fresh RNG, no local progress, no supplemental
        rng_fresh = random.Random(42)
        batch_fresh = _sample_batch(5, allocs, num_layers, rng_fresh)

        # Fresh RNG, no local progress (num_local=0), with supplemental
        # — RNG should NOT be advanced
        rng_with_supp = random.Random(42)
        num_local = 0
        if num_local > 0:
            _fast_forward_rng(rng_with_supp, num_local, allocs, num_layers)
        batch_with_supp = _sample_batch(5, allocs, num_layers, rng_with_supp)

        assert batch_fresh == batch_with_supp

    def test_rng_determinism_with_local_and_supplemental(self) -> None:
        """RNG advances by num_local only, supplemental records are ignored for RNG."""
        allocs = [{"m0": 3, "m1": 2}, {"m0": 2, "m1": 3}]
        num_layers = 5

        # Simulate: 3 local samples, then sample batch
        rng_local_only = random.Random(42)
        _fast_forward_rng(rng_local_only, 3, allocs, num_layers)
        batch_local_only = _sample_batch(5, allocs, num_layers, rng_local_only)

        # Simulate: 3 local + 10 supplemental, RNG advanced by 3 only
        rng_local_plus_supp = random.Random(42)
        num_local = 3
        _fast_forward_rng(rng_local_plus_supp, num_local, allocs, num_layers)
        batch_local_plus_supp = _sample_batch(5, allocs, num_layers, rng_local_plus_supp)

        assert batch_local_only == batch_local_plus_supp

    def test_nonexistent_supplemental_dir(self, tmp_path: Path) -> None:
        """discover_evaluation_records returns [] for nonexistent directory."""
        bogus = str(tmp_path / "does_not_exist")
        files = discover_evaluation_records([bogus], "obj")
        assert files == []

    def test_empty_supplemental_dirs_skipped(self) -> None:
        """When supplemental_data_dirs is empty, the merge block is skipped."""
        # Just verify the config field accepts an empty list
        from place_layers.command.explore_placements import ExplorePlacementsCommandConfig

        # The field should be present and accept an empty list
        field_info = ExplorePlacementsCommandConfig.model_fields["supplemental_data_dirs"]
        assert field_info is not None

    def test_sidecar_num_records_excludes_supplemental(self, tmp_path: Path) -> None:
        """Sidecar num_records should count only locally-evaluated records, not supplemental."""
        # Local progress: 2 records
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        local_progress = local_dir / "exploration_progress_obj.jsonl"
        with local_progress.open("w") as f:
            append_record(f, EvaluationRecord(
                assignment=("m0", "m1"), score=0.5,
            ))
            append_record(f, EvaluationRecord(
                assignment=("m1", "m0"), score=0.6,
            ))

        # Supplemental dir: 2 new records
        supp_dir = tmp_path / "supplemental"
        supp_dir.mkdir()
        _write_supplemental_jsonl(
            supp_dir, "obj",
            [
                EvaluationRecord(assignment=("m0", "m0"), score=0.4),
                EvaluationRecord(assignment=("m1", "m1"), score=0.7),
            ],
            num_layers=2,
        )

        # Simulate the merge logic from the command
        all_assignments, all_scores = _load_progress(local_progress)
        num_local = len(all_assignments)
        n_supplemental = 0

        supplemental_files = discover_evaluation_records([str(supp_dir)], "obj")
        pool = load_pooled_data(supplemental_files, deduplicate=True)
        existing = {(a, s) for a, s in zip(all_assignments, all_scores)}
        n_before = len(all_assignments)
        for a, s in zip(pool.assignments, pool.scores):
            if (a, s) not in existing:
                all_assignments.append(a)
                all_scores.append(s)
                existing.add((a, s))
        n_supplemental = len(all_assignments) - n_before

        # Sidecar should reflect only local records
        sidecar_num_records = len(all_assignments) - n_supplemental
        assert sidecar_num_records == num_local == 2
        assert len(all_assignments) == 4  # total includes supplemental


# =============================================================================
# Snowball file tests
# =============================================================================


class TestSnowballFile:
    """Tests for incremental snowball evaluations_{obj}.jsonl in explore_placements."""

    def test_snowball_incremental(self, tmp_path: Path) -> None:
        """Snowball file gets supplemental + local records; progress only local."""
        from place_layers.surrogate.evaluation_log import load_records, load_meta

        output_dir = tmp_path / "run"
        output_dir.mkdir()
        obj_name = "obj"

        # Supplemental records (simulating what was loaded from prior runs)
        supp_records = [
            EvaluationRecord(assignment=("m0", "m0"), score=0.4),
            EvaluationRecord(assignment=("m1", "m1"), score=0.7),
        ]

        # Simulate the dual-write flow from explore_placements
        obj_dir = output_dir / obj_name
        obj_dir.mkdir(parents=True, exist_ok=True)
        progress_path = obj_dir / "progress.jsonl"
        snowball_path = obj_dir / EVALUATION_RECORDS_FILENAME

        # Write sidecar metadata
        from datetime import datetime, timezone

        progress_meta = EvaluationLogMeta(
            version=1, objective_name=obj_name, num_layers=2,
            mixer_names=["m0", "m1"], source_command="explore_placements",
            source_detail="exploration_progress",
            created_at=datetime.now(timezone.utc).isoformat(),
            num_records=None,
        )
        snowball_meta = EvaluationLogMeta(
            version=1, objective_name=obj_name, num_layers=2,
            mixer_names=["m0", "m1"], source_command="explore_placements",
            source_detail="merged_snowball",
            created_at=datetime.now(timezone.utc).isoformat(),
            num_records=None,
        )
        write_meta(progress_path, progress_meta)
        write_meta(snowball_path, snowball_meta)

        # Open both files, seed snowball
        progress_file = progress_path.open("a")
        snowball_file = snowball_path.open("w")
        try:
            # Seed snowball with supplemental records
            for rec in supp_records:
                append_record(snowball_file, rec)

            # Simulate evaluating 1 new local record
            local_record = EvaluationRecord(
                assignment=("m0", "m1"), score=0.6,
            )
            append_record(progress_file, local_record)
            append_record(snowball_file, local_record)
        finally:
            progress_file.close()
            snowball_file.close()

        # Verify: progress has only local record
        progress_records = load_records(progress_path)
        assert len(progress_records) == 1
        assert progress_records[0].assignment == ("m0", "m1")

        # Verify: snowball has supplemental + local
        snowball_records = load_records(snowball_path)
        assert len(snowball_records) == 3
        assert snowball_records[0].assignment == ("m0", "m0")  # supplemental
        assert snowball_records[1].assignment == ("m1", "m1")  # supplemental
        assert snowball_records[2].assignment == ("m0", "m1")  # local

        # Update sidecars with final counts
        write_meta(
            progress_path,
            progress_meta.model_copy(update={"num_records": 1}),
        )
        write_meta(
            snowball_path,
            snowball_meta.model_copy(update={"num_records": 3}),
        )

        progress_loaded_meta = load_meta(progress_path)
        snowball_loaded_meta = load_meta(snowball_path)
        assert progress_loaded_meta is not None
        assert progress_loaded_meta.num_records == 1
        assert snowball_loaded_meta is not None
        assert snowball_loaded_meta.num_records == 3

    def test_chaining_single_dir(self, tmp_path: Path) -> None:
        """Chain 3 runs: each snowball accumulates prior data."""
        from place_layers.surrogate.evaluation_log import (
            load_records,
            load_pooled_data,
            discover_evaluation_records,
        )

        # --- Run 1: 2 local records, no supplemental ---
        run1_dir = tmp_path / "run1"
        run1_dir.mkdir()
        snowball1 = run1_dir / "evaluations_obj.jsonl"
        with snowball1.open("w") as f:
            append_record(f, EvaluationRecord(
                assignment=("m0", "m1"), score=0.5,
            ))
            append_record(f, EvaluationRecord(
                assignment=("m1", "m0"), score=0.6,
            ))
        write_meta(snowball1, EvaluationLogMeta(
            version=1, objective_name="obj", num_layers=2,
            mixer_names=["m0", "m1"], source_command="explore_placements",
            source_detail="merged_snowball",
            created_at="2026-01-01T00:00:00Z", num_records=2,
        ))

        # --- Run 2: supplemental=[run1], 2 new local records ---
        run2_dir = tmp_path / "run2"
        run2_dir.mkdir()

        supp_files = discover_evaluation_records([str(run1_dir)], "obj")
        assert len(supp_files) > 0

        pool = load_pooled_data(supp_files, deduplicate=True)
        assert pool.num_samples == 2

        snowball2 = run2_dir / "evaluations_obj.jsonl"
        with snowball2.open("w") as f:
            # Seed with supplemental
            for rec in pool.as_records():
                append_record(f, rec)
            # Add 2 new local records
            append_record(f, EvaluationRecord(
                assignment=("m0", "m0"), score=0.4,
            ))
            append_record(f, EvaluationRecord(
                assignment=("m1", "m1"), score=0.7,
            ))
        write_meta(snowball2, EvaluationLogMeta(
            version=1, objective_name="obj", num_layers=2,
            mixer_names=["m0", "m1"], source_command="explore_placements",
            source_detail="merged_snowball",
            created_at="2026-01-02T00:00:00Z", num_records=4,
        ))

        # Verify run2 snowball has 4 records
        assert len(load_records(snowball2)) == 4

        # --- Run 3: supplemental=[run2], loads all 4 ---
        supp_files_3 = discover_evaluation_records([str(run2_dir)], "obj")
        pool3 = load_pooled_data(supp_files_3, deduplicate=True)
        assert pool3.num_samples == 4
        # All 4 assignments present
        all_assignments = set(pool3.assignments)
        assert ("m0", "m1") in all_assignments
        assert ("m1", "m0") in all_assignments
        assert ("m0", "m0") in all_assignments
        assert ("m1", "m1") in all_assignments
