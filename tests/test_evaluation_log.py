"""Unit tests for surrogate/evaluation_log module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from place_layers.surrogate.evaluation_log import (
    EvaluationLogMeta,
    EvaluationRecord,
    PooledEvaluationData,
    append_record,
    discover_evaluation_files,
    load_meta,
    load_pooled_data,
    load_records,
    meta_path_for,
    write_meta,
)


# =============================================================================
# EvaluationRecord
# =============================================================================


class TestEvaluationRecord:
    """Tests for EvaluationRecord model."""

    def test_round_trip_json(self):
        """model_dump_json -> model_validate_json preserves fields."""
        rec = EvaluationRecord(
            assignment=("attn", "gdn", "attn"),
            score=0.75,
            allocation={"attn": 2, "gdn": 1},
        )
        json_str = rec.model_dump_json()
        loaded = EvaluationRecord.model_validate_json(json_str)
        assert loaded.assignment == ("attn", "gdn", "attn")
        assert loaded.score == 0.75
        assert loaded.allocation == {"attn": 2, "gdn": 1}

    def test_frozen(self):
        """EvaluationRecord is frozen."""
        rec = EvaluationRecord(
            assignment=("attn",), score=0.5, allocation={"attn": 1}
        )
        with pytest.raises(Exception):
            rec.score = 0.9  # type: ignore[misc]

    def test_backward_compat_with_explore_format(self):
        """Records can be loaded from explore_placements JSONL format."""
        line = json.dumps({
            "assignment": ["attn", "gdn", "attn"],
            "score": 0.8,
            "allocation": {"attn": 2, "gdn": 1},
        })
        rec = EvaluationRecord.model_validate_json(line)
        assert rec.assignment == ("attn", "gdn", "attn")
        assert rec.score == 0.8


# =============================================================================
# JSONL I/O
# =============================================================================


class TestJsonlIO:
    """Tests for append_record / load_records."""

    def test_write_and_read_round_trip(self, tmp_path: Path):
        """Write multiple records, read them back identically."""
        records = [
            EvaluationRecord(
                assignment=("attn", "gdn"), score=0.5, allocation={"attn": 1, "gdn": 1}
            ),
            EvaluationRecord(
                assignment=("gdn", "gdn"), score=0.6, allocation={"gdn": 2}
            ),
            EvaluationRecord(
                assignment=("attn", "attn"), score=0.7, allocation={"attn": 2}
            ),
        ]

        jsonl_path = tmp_path / "evaluations_test.jsonl"
        with jsonl_path.open("w") as f:
            for rec in records:
                append_record(f, rec)

        loaded = load_records(jsonl_path)
        assert len(loaded) == 3
        for orig, loaded_rec in zip(records, loaded):
            assert loaded_rec.assignment == orig.assignment
            assert loaded_rec.score == orig.score
            assert loaded_rec.allocation == orig.allocation

    def test_load_empty_file(self, tmp_path: Path):
        """Loading an empty file returns empty list."""
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")
        assert load_records(jsonl_path) == []

    def test_load_skips_blank_lines(self, tmp_path: Path):
        """Blank lines in JSONL are skipped."""
        rec = EvaluationRecord(
            assignment=("attn",), score=0.5, allocation={"attn": 1}
        )
        jsonl_path = tmp_path / "with_blanks.jsonl"
        jsonl_path.write_text(
            "\n" + rec.model_dump_json() + "\n\n" + rec.model_dump_json() + "\n\n"
        )
        loaded = load_records(jsonl_path)
        assert len(loaded) == 2


# =============================================================================
# Sidecar metadata
# =============================================================================


class TestMetadata:
    """Tests for EvaluationLogMeta sidecar files."""

    def test_meta_path_for(self):
        """meta_path_for maps .jsonl to .meta.yaml correctly."""
        p = meta_path_for(Path("/foo/bar/evaluations_gsm8k.jsonl"))
        assert p == Path("/foo/bar/evaluations_gsm8k.meta.yaml")

    def test_write_and_load_meta(self, tmp_path: Path):
        """Write and load sidecar metadata round-trip."""
        jsonl_path = tmp_path / "evaluations_test.jsonl"
        jsonl_path.write_text("")  # create the file

        meta = EvaluationLogMeta(
            version=1,
            objective_name="gsm8k",
            num_layers=24,
            mixer_names=["attn", "gdn"],
            source_command="validate_landscape",
            source_detail="band_003",
            created_at="2026-01-01T00:00:00+00:00",
            num_records=42,
        )
        write_meta(jsonl_path, meta)

        loaded = load_meta(jsonl_path)
        assert loaded is not None
        assert loaded.version == 1
        assert loaded.objective_name == "gsm8k"
        assert loaded.num_layers == 24
        assert loaded.mixer_names == ["attn", "gdn"]
        assert loaded.num_records == 42

    def test_load_meta_missing(self, tmp_path: Path):
        """load_meta returns None when no sidecar exists."""
        jsonl_path = tmp_path / "evaluations_test.jsonl"
        assert load_meta(jsonl_path) is None


# =============================================================================
# Pooled data loading
# =============================================================================


class TestPooledData:
    """Tests for load_pooled_data and PooledEvaluationData."""

    def _write_jsonl(self, path: Path, records: list[EvaluationRecord]) -> None:
        with path.open("w") as f:
            for rec in records:
                append_record(f, rec)

    def _write_exploration_json(self, path: Path, assignments, scores, allocations) -> None:
        """Write a minimal ExplorationDataset JSON file."""
        from place_layers.surrogate.dataset import ExplorationDataset
        from place_layers.surrogate.encoding import FeatureExpansionConfig

        ds = ExplorationDataset(
            assignments=assignments,
            scores=scores,
            allocations=allocations,
            objective_name="test",
            num_layers=len(assignments[0]),
            mixer_names=["attn", "gdn"],
            num_samples=len(assignments),
            termination_reason="max_samples",
            feature_config=FeatureExpansionConfig(
                max_order=1, pairwise_max_range=1, allocation_counts=True
            ),
            rho_history=[0.9],
            samples_at_check=[len(assignments)],
            final_rho=0.9,
        )
        path.write_text(ds.model_dump_json())

    def test_load_jsonl_files(self, tmp_path: Path):
        """load_pooled_data loads JSONL files correctly."""
        records = [
            EvaluationRecord(assignment=("attn", "gdn"), score=0.5, allocation={"attn": 1, "gdn": 1}),
            EvaluationRecord(assignment=("gdn", "gdn"), score=0.6, allocation={"gdn": 2}),
        ]
        jsonl_path = tmp_path / "evaluations_test.jsonl"
        self._write_jsonl(jsonl_path, records)

        # Write sidecar for metadata
        meta = EvaluationLogMeta(
            version=1, objective_name="test", num_layers=2,
            mixer_names=["attn", "gdn"], source_command="test",
            source_detail="test", created_at="2026-01-01T00:00:00+00:00",
            num_records=2,
        )
        write_meta(jsonl_path, meta)

        pool = load_pooled_data([jsonl_path])
        assert pool.num_samples == 2
        assert pool.num_layers == 2
        assert pool.mixer_names == ["attn", "gdn"]

    def test_load_exploration_json(self, tmp_path: Path):
        """load_pooled_data loads ExplorationDataset .json files."""
        assignments = [("attn", "gdn"), ("gdn", "attn")]
        scores = [0.5, 0.6]
        allocations = [{"attn": 1, "gdn": 1}, {"attn": 1, "gdn": 1}]

        json_path = tmp_path / "exploration_results_test.json"
        self._write_exploration_json(json_path, assignments, scores, allocations)

        pool = load_pooled_data([json_path])
        assert pool.num_samples == 2
        assert pool.num_layers == 2

    def test_multi_file_loading(self, tmp_path: Path):
        """load_pooled_data merges data from JSON + JSONL sources."""
        # Write exploration JSON
        assignments = [("attn", "gdn"), ("gdn", "attn")]
        scores = [0.5, 0.6]
        allocations = [{"attn": 1, "gdn": 1}, {"attn": 1, "gdn": 1}]
        json_path = tmp_path / "exploration_results_test.json"
        self._write_exploration_json(json_path, assignments, scores, allocations)

        # Write JSONL with different data
        records = [
            EvaluationRecord(assignment=("gdn", "gdn"), score=0.7, allocation={"gdn": 2}),
        ]
        jsonl_path = tmp_path / "evaluations_test.jsonl"
        self._write_jsonl(jsonl_path, records)

        pool = load_pooled_data([json_path, jsonl_path], deduplicate=False)
        assert pool.num_samples == 3

    def test_deduplication(self, tmp_path: Path):
        """Exact (assignment, score) duplicates are removed."""
        rec = EvaluationRecord(assignment=("attn", "gdn"), score=0.5, allocation={"attn": 1, "gdn": 1})
        jsonl1 = tmp_path / "eval1.jsonl"
        jsonl2 = tmp_path / "eval2.jsonl"
        self._write_jsonl(jsonl1, [rec])
        self._write_jsonl(jsonl2, [rec])

        # Write sidecar for metadata on first file
        meta = EvaluationLogMeta(
            version=1, objective_name="test", num_layers=2,
            mixer_names=["attn", "gdn"], source_command="test",
            source_detail="test", created_at="2026-01-01T00:00:00+00:00",
            num_records=1,
        )
        write_meta(jsonl1, meta)

        pool_dedup = load_pooled_data([jsonl1, jsonl2], deduplicate=True)
        assert pool_dedup.num_samples == 1

        pool_no_dedup = load_pooled_data([jsonl1, jsonl2], deduplicate=False)
        assert pool_no_dedup.num_samples == 2

    def test_missing_file_skipped(self, tmp_path: Path):
        """Missing files are warned about but don't cause errors."""
        # Write one valid JSONL with sidecar
        rec = EvaluationRecord(assignment=("attn",), score=0.5, allocation={"attn": 1})
        jsonl_path = tmp_path / "evaluations_test.jsonl"
        self._write_jsonl(jsonl_path, [rec])
        meta = EvaluationLogMeta(
            version=1, objective_name="test", num_layers=1,
            mixer_names=["attn"], source_command="test",
            source_detail="test", created_at="2026-01-01T00:00:00+00:00",
            num_records=1,
        )
        write_meta(jsonl_path, meta)

        pool = load_pooled_data([
            tmp_path / "nonexistent.jsonl",
            jsonl_path,
        ])
        assert pool.num_samples == 1

    def test_no_valid_files_raises(self, tmp_path: Path):
        """ValueError when no file provides num_layers."""
        with pytest.raises(ValueError, match="Could not determine num_layers"):
            load_pooled_data([tmp_path / "nonexistent.jsonl"])

    def test_num_samples_property(self):
        """PooledEvaluationData.num_samples returns correct count."""
        pool = PooledEvaluationData(
            assignments=[("a", "b"), ("c", "d")],
            scores=[0.1, 0.2],
            allocations=[{"a": 1, "b": 1}, {"c": 1, "d": 1}],
            num_layers=2,
            mixer_names=["a", "b", "c", "d"],
        )
        assert pool.num_samples == 2


# =============================================================================
# discover_evaluation_files
# =============================================================================


class TestDiscoverEvaluationFiles:
    """Tests for discover_evaluation_files."""

    def test_discovers_exploration_json(self, tmp_path: Path):
        """Finds exploration_results_{obj}.json at top level."""
        (tmp_path / "exploration_results_gsm8k.json").write_text("{}")
        files = discover_evaluation_files(tmp_path, "gsm8k")
        assert len(files) == 1
        assert files[0].name == "exploration_results_gsm8k.json"

    def test_discovers_nested_jsonl(self, tmp_path: Path):
        """Finds evaluations_{obj}.jsonl recursively."""
        band_dir = tmp_path / "bands_gsm8k" / "band_001"
        band_dir.mkdir(parents=True)
        (band_dir / "evaluations_gsm8k.jsonl").write_text("")
        files = discover_evaluation_files(tmp_path, "gsm8k")
        assert len(files) == 1
        assert files[0].name == "evaluations_gsm8k.jsonl"

    def test_discovers_both_types(self, tmp_path: Path):
        """Finds both JSON and JSONL files."""
        (tmp_path / "exploration_results_gsm8k.json").write_text("{}")
        band_dir = tmp_path / "bands_gsm8k" / "band_001"
        band_dir.mkdir(parents=True)
        (band_dir / "evaluations_gsm8k.jsonl").write_text("")
        files = discover_evaluation_files(tmp_path, "gsm8k")
        assert len(files) == 2
        # JSON comes first
        assert files[0].suffix == ".json"
        assert files[1].suffix == ".jsonl"

    def test_discovers_exploration_progress_jsonl(self, tmp_path: Path):
        """Finds exploration_progress_{obj}.jsonl at top level."""
        (tmp_path / "exploration_progress_gsm8k.jsonl").write_text("")
        files = discover_evaluation_files(tmp_path, "gsm8k")
        assert len(files) == 1
        assert files[0].name == "exploration_progress_gsm8k.jsonl"

    def test_discovers_all_three_types(self, tmp_path: Path):
        """Finds JSON, progress JSONL, and evaluation JSONL in correct order."""
        (tmp_path / "exploration_results_gsm8k.json").write_text("{}")
        (tmp_path / "exploration_progress_gsm8k.jsonl").write_text("")
        band_dir = tmp_path / "bands_gsm8k" / "band_001"
        band_dir.mkdir(parents=True)
        (band_dir / "evaluations_gsm8k.jsonl").write_text("")
        files = discover_evaluation_files(tmp_path, "gsm8k")
        assert len(files) == 3
        assert files[0].suffix == ".json"
        assert files[1].name == "exploration_progress_gsm8k.jsonl"
        assert files[2].name == "evaluations_gsm8k.jsonl"

    def test_ignores_wrong_objective(self, tmp_path: Path):
        """Files for a different objective are not discovered."""
        (tmp_path / "exploration_results_mmlu.json").write_text("{}")
        files = discover_evaluation_files(tmp_path, "gsm8k")
        assert len(files) == 0

    def test_empty_directory(self, tmp_path: Path):
        """Empty directory returns no files."""
        files = discover_evaluation_files(tmp_path, "gsm8k")
        assert len(files) == 0
