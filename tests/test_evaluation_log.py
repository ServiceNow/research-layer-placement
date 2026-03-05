"""Unit tests for surrogate/evaluation_log module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from place_layers.surrogate.evaluation_log import (
    EVALUATION_RECORDS_FILENAME,
    EvaluationLogMeta,
    EvaluationRecord,
    PooledEvaluationData,
    append_record,
    discover_evaluation_records,
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
        )
        json_str = rec.model_dump_json()
        loaded = EvaluationRecord.model_validate_json(json_str)
        assert loaded.assignment == ("attn", "gdn", "attn")
        assert loaded.score == 0.75
        assert loaded.allocation == {"attn": 2, "gdn": 1}

    def test_frozen(self):
        """EvaluationRecord is frozen."""
        rec = EvaluationRecord(
            assignment=("attn",), score=0.5,
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
        # allocation is computed, stored value is ignored
        assert rec.allocation == {"attn": 2, "gdn": 1}

    def test_allocation_is_computed(self):
        """allocation is computed from assignment, not stored."""
        rec = EvaluationRecord(
            assignment=("attn", "gdn", "gdn", "attn"),
            score=0.5,
        )
        assert rec.allocation == {"attn": 2, "gdn": 2}

    def test_allocation_in_dump(self):
        """allocation appears in model_dump_json for backward compat."""
        rec = EvaluationRecord(
            assignment=("attn", "gdn"),
            score=0.5,
        )
        data = json.loads(rec.model_dump_json())
        assert "allocation" in data
        assert data["allocation"] == {"attn": 1, "gdn": 1}


# =============================================================================
# JSONL I/O
# =============================================================================


class TestJsonlIO:
    """Tests for append_record / load_records."""

    def test_write_and_read_round_trip(self, tmp_path: Path):
        """Write multiple records, read them back identically."""
        records = [
            EvaluationRecord(
                assignment=("attn", "gdn"), score=0.5,
            ),
            EvaluationRecord(
                assignment=("gdn", "gdn"), score=0.6,
            ),
            EvaluationRecord(
                assignment=("attn", "attn"), score=0.7,
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
            assignment=("attn",), score=0.5,
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

    def test_meta_path_for_new_convention(self):
        """meta_path_for maps evaluation_records.jsonl correctly."""
        p = meta_path_for(Path("/foo/bar/evaluation_records.jsonl"))
        assert p == Path("/foo/bar/evaluation_records.meta.yaml")

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

    def test_meta_with_band_fields(self, tmp_path: Path):
        """EvaluationLogMeta supports cost_floor, cost_ceiling, band_index."""
        jsonl_path = tmp_path / EVALUATION_RECORDS_FILENAME
        jsonl_path.write_text("")

        meta = EvaluationLogMeta(
            version=1,
            objective_name="gsm8k",
            num_layers=24,
            mixer_names=["attn", "gdn"],
            source_command="validate_landscape",
            source_detail="per_band_search",
            created_at="2026-01-01T00:00:00+00:00",
            num_records=10,
            cost_floor=0.5,
            cost_ceiling=0.8,
            band_index=3,
        )
        write_meta(jsonl_path, meta)

        loaded = load_meta(jsonl_path)
        assert loaded is not None
        assert loaded.cost_floor == 0.5
        assert loaded.cost_ceiling == 0.8
        assert loaded.band_index == 3

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

    def _write_exploration_json(self, path: Path, assignments, scores) -> None:
        """Write a minimal ExplorationDataset JSON file."""
        from place_layers.surrogate.dataset import ExplorationDataset
        from place_layers.surrogate.encoding import FeatureExpansionConfig

        ds = ExplorationDataset(
            assignments=assignments,
            scores=scores,
            objective_name="test",
            num_layers=len(assignments[0]),
            mixer_names=["attn", "gdn"],
            num_samples=len(assignments),
            termination_reason="max_samples",
            feature_config=FeatureExpansionConfig(
                max_order=1, pairwise_max_range=1, allocation_counts=True
            ),
            quality_history=[0.9],
            samples_at_check=[len(assignments)],
            final_quality=0.9,
        )
        path.write_text(ds.model_dump_json())

    def test_load_jsonl_files(self, tmp_path: Path):
        """load_pooled_data loads JSONL files correctly."""
        records = [
            EvaluationRecord(assignment=("attn", "gdn"), score=0.5),
            EvaluationRecord(assignment=("gdn", "gdn"), score=0.6),
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

        json_path = tmp_path / "exploration_results_test.json"
        self._write_exploration_json(json_path, assignments, scores)

        pool = load_pooled_data([json_path])
        assert pool.num_samples == 2
        assert pool.num_layers == 2

    def test_multi_file_loading(self, tmp_path: Path):
        """load_pooled_data merges data from JSON + JSONL sources."""
        # Write exploration JSON
        assignments = [("attn", "gdn"), ("gdn", "attn")]
        scores = [0.5, 0.6]
        json_path = tmp_path / "exploration_results_test.json"
        self._write_exploration_json(json_path, assignments, scores)

        # Write JSONL with different data
        records = [
            EvaluationRecord(assignment=("gdn", "gdn"), score=0.7),
        ]
        jsonl_path = tmp_path / "evaluations_test.jsonl"
        self._write_jsonl(jsonl_path, records)

        pool = load_pooled_data([json_path, jsonl_path], deduplicate=False)
        assert pool.num_samples == 3

    def test_deduplication(self, tmp_path: Path):
        """Exact (assignment, score) duplicates are removed."""
        rec = EvaluationRecord(assignment=("attn", "gdn"), score=0.5)
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
        rec = EvaluationRecord(assignment=("attn",), score=0.5)
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
            num_layers=2,
            mixer_names=["a", "b", "c", "d"],
        )
        assert pool.num_samples == 2

    def test_as_records(self):
        """as_records() converts parallel lists to EvaluationRecord list."""
        pool = PooledEvaluationData(
            assignments=[("attn", "gdn"), ("gdn", "gdn")],
            scores=[0.5, 0.6],
            num_layers=2,
            mixer_names=["attn", "gdn"],
        )
        records = pool.as_records()
        assert len(records) == 2
        assert records[0].assignment == ("attn", "gdn")
        assert records[0].score == 0.5
        assert records[0].allocation == {"attn": 1, "gdn": 1}
        assert records[1].assignment == ("gdn", "gdn")
        assert records[1].score == 0.6
        assert records[1].allocation == {"gdn": 2}

    def test_as_records_empty(self):
        """as_records() returns empty list for empty pool."""
        pool = PooledEvaluationData(
            assignments=[], scores=[],
            num_layers=2, mixer_names=["attn", "gdn"],
        )
        assert pool.as_records() == []


# =============================================================================
# discover_evaluation_records
# =============================================================================


class TestDiscoverEvaluationRecords:
    """Tests for discover_evaluation_records."""

    def _write_sidecar(self, path: Path, objective_name: str) -> None:
        """Write a sidecar for the given path."""
        meta = EvaluationLogMeta(
            version=1, objective_name=objective_name, num_layers=24,
            mixer_names=["attn", "gdn"], source_command="test",
            source_detail="test", created_at="2026-01-01T00:00:00+00:00",
            num_records=1,
        )
        write_meta(path, meta)

    def test_finds_legacy_snowball(self, tmp_path: Path):
        """Finds evaluations_{obj}.jsonl at top level (legacy)."""
        (tmp_path / "evaluations_gsm8k.jsonl").write_text("")
        files = discover_evaluation_records([tmp_path], "gsm8k")
        assert len(files) == 1
        assert files[0].name == "evaluations_gsm8k.jsonl"

    def test_finds_new_convention(self, tmp_path: Path):
        """Finds evaluation_records.jsonl with matching sidecar."""
        obj_dir = tmp_path / "gsm8k"
        obj_dir.mkdir()
        eval_path = obj_dir / EVALUATION_RECORDS_FILENAME
        eval_path.write_text("")
        self._write_sidecar(eval_path, "gsm8k")

        files = discover_evaluation_records([tmp_path], "gsm8k")
        assert len(files) == 1
        assert files[0].name == EVALUATION_RECORDS_FILENAME

    def test_finds_nested_snowball(self, tmp_path: Path):
        """Finds evaluations_{obj}.jsonl recursively."""
        band_dir = tmp_path / "bands_gsm8k" / "band_001"
        band_dir.mkdir(parents=True)
        (band_dir / "evaluations_gsm8k.jsonl").write_text("")
        files = discover_evaluation_records([tmp_path], "gsm8k")
        assert len(files) == 1
        assert files[0].name == "evaluations_gsm8k.jsonl"

    def test_ignores_wrong_objective_sidecar(self, tmp_path: Path):
        """evaluation_records.jsonl with wrong objective sidecar is skipped."""
        eval_path = tmp_path / EVALUATION_RECORDS_FILENAME
        eval_path.write_text("")
        self._write_sidecar(eval_path, "mmlu")  # wrong objective

        files = discover_evaluation_records([tmp_path], "gsm8k")
        assert len(files) == 0

    def test_ignores_no_sidecar(self, tmp_path: Path):
        """evaluation_records.jsonl without sidecar is skipped."""
        eval_path = tmp_path / EVALUATION_RECORDS_FILENAME
        eval_path.write_text("")

        files = discover_evaluation_records([tmp_path], "gsm8k")
        assert len(files) == 0

    def test_deduplicates_across_dirs(self, tmp_path: Path):
        """Same file referenced from two dirs appears once."""
        (tmp_path / "evaluations_gsm8k.jsonl").write_text("")
        files = discover_evaluation_records([tmp_path, tmp_path], "gsm8k")
        assert len(files) == 1

    def test_empty_directory(self, tmp_path: Path):
        """Empty directory returns no files."""
        files = discover_evaluation_records([tmp_path], "gsm8k")
        assert len(files) == 0

    def test_ignores_wrong_objective(self, tmp_path: Path):
        """Files for a different objective are not discovered."""
        (tmp_path / "evaluations_mmlu.jsonl").write_text("")
        files = discover_evaluation_records([tmp_path], "gsm8k")
        assert len(files) == 0

    def test_cost_bound_filtering(self, tmp_path: Path):
        """Cost bound constraints filter by sidecar metadata."""
        band_dir = tmp_path / "band_001"
        band_dir.mkdir()
        eval_path = band_dir / EVALUATION_RECORDS_FILENAME
        eval_path.write_text("")
        meta = EvaluationLogMeta(
            version=1, objective_name="gsm8k", num_layers=24,
            mixer_names=["attn", "gdn"], source_command="test",
            source_detail="test", created_at="2026-01-01T00:00:00+00:00",
            num_records=1, cost_floor=0.3, cost_ceiling=0.6,
        )
        write_meta(eval_path, meta)

        # Within range: should find
        files = discover_evaluation_records(
            [tmp_path], "gsm8k", cost_floor=0.2, cost_ceiling=0.7,
        )
        assert len(files) == 1

        # Outside range (ceiling too low): should not find
        files = discover_evaluation_records(
            [tmp_path], "gsm8k", cost_floor=0.2, cost_ceiling=0.5,
        )
        assert len(files) == 0

    def test_nonexistent_directory(self, tmp_path: Path):
        """Nonexistent directory is handled gracefully."""
        files = discover_evaluation_records(
            [tmp_path / "does_not_exist"], "gsm8k"
        )
        assert len(files) == 0
