"""Evaluation data logging and pooling for surrogate training.

Provides:
- ``EvaluationRecord`` / ``EvaluationLogMeta``: Pydantic models for JSONL
  evaluation logs and their sidecar metadata.
- ``append_record`` / ``load_records``: write/read individual JSONL files.
- ``PooledEvaluationData`` / ``load_pooled_data``: merge data from multiple
  evaluation sources (exploration JSON + evaluation JSONL).
- ``resolve_supplemental_dirs``: find per-objective snowball JSONL files in
  a set of directories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import IO, Sequence

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ============================================================================
# JSONL record model
# ============================================================================


class EvaluationRecord(BaseModel):
    """Single evaluated placement. One per JSONL line.

    Field names match the existing explore_placements progress format
    (``assignment``, ``score``, ``allocation``) for backward compatibility.
    """

    model_config = {"frozen": True}

    assignment: tuple[str, ...]
    score: float
    allocation: dict[str, int]


# ============================================================================
# Sidecar metadata
# ============================================================================


class EvaluationLogMeta(BaseModel):
    """Sidecar metadata for an evaluation JSONL file."""

    model_config = {"frozen": True}

    version: int
    objective_name: str
    num_layers: int
    mixer_names: list[str]
    source_command: str
    source_detail: str
    created_at: str
    num_records: int | None


# ============================================================================
# Record I/O
# ============================================================================


def append_record(f: IO[str], record: EvaluationRecord) -> None:
    """Append a single record to an open JSONL file handle and flush."""
    f.write(record.model_dump_json() + "\n")
    f.flush()


def load_records(path: Path | str) -> list[EvaluationRecord]:
    """Load all records from a JSONL evaluation log file."""
    path = Path(path)
    records: list[EvaluationRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(EvaluationRecord.model_validate_json(line))
    return records


# ============================================================================
# Sidecar I/O
# ============================================================================


def meta_path_for(jsonl_path: Path) -> Path:
    """Public helper: ``evaluations_foo.jsonl`` → ``evaluations_foo.meta.yaml``."""
    stem = jsonl_path.name
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    return jsonl_path.with_name(stem + ".meta.yaml")


def write_meta(jsonl_path: Path, meta: EvaluationLogMeta) -> None:
    """Write sidecar metadata YAML alongside a JSONL file."""
    mp = meta_path_for(jsonl_path)
    mp.write_text(yaml.safe_dump(meta.model_dump(), sort_keys=False))


def load_meta(jsonl_path: Path) -> EvaluationLogMeta | None:
    """Load sidecar metadata if it exists, else None."""
    mp = meta_path_for(jsonl_path)
    if not mp.exists():
        return None
    return EvaluationLogMeta.model_validate(yaml.safe_load(mp.read_text()))


# ============================================================================
# Pooled data
# ============================================================================


class PooledEvaluationData(BaseModel):
    """Merged data from multiple evaluation sources. Immutable view."""

    model_config = {"frozen": True}

    assignments: list[tuple[str, ...]]
    scores: list[float]
    allocations: list[dict[str, int]]
    num_layers: int
    mixer_names: list[str]

    @property
    def num_samples(self) -> int:
        return len(self.assignments)

    def as_records(self) -> list[EvaluationRecord]:
        """Convert to list of EvaluationRecord for serialization."""
        return [
            EvaluationRecord(assignment=a, score=s, allocation=alloc)
            for a, s, alloc in zip(self.assignments, self.scores, self.allocations)
        ]


def load_pooled_data(
    paths: Sequence[Path | str],
    *,
    deduplicate: bool = True,
) -> PooledEvaluationData:
    """Load and merge evaluation data from multiple files.

    Accepts:
    - ExplorationDataset ``.json`` files (reads via ``ExplorationDataset.load()``)
    - Evaluation ``.jsonl`` files (reads records via ``load_records()``)

    When *deduplicate* is True, removes exact ``(assignment, score)`` duplicates
    that arise when the same data is loaded from multiple sources.
    """
    from place_layers.surrogate.dataset import ExplorationDataset

    all_assignments: list[tuple[str, ...]] = []
    all_scores: list[float] = []
    all_allocations: list[dict[str, int]] = []
    num_layers: int | None = None
    mixer_names: list[str] | None = None

    for raw_path in paths:
        p = Path(raw_path)
        if not p.exists():
            logger.warning("Skipping missing evaluation data file: %s", p)
            continue

        if p.suffix == ".json":
            ds = ExplorationDataset.load(p)
            all_assignments.extend(ds.assignments)
            all_scores.extend(ds.scores)
            all_allocations.extend(ds.allocations)
            if num_layers is None:
                num_layers = ds.num_layers
                mixer_names = ds.mixer_names
        elif p.suffix == ".jsonl":
            records = load_records(p)
            for rec in records:
                all_assignments.append(rec.assignment)
                all_scores.append(rec.score)
                all_allocations.append(rec.allocation)
            # Try to get metadata from sidecar
            if num_layers is None:
                meta = load_meta(p)
                if meta is not None:
                    num_layers = meta.num_layers
                    mixer_names = meta.mixer_names
        else:
            logger.warning("Skipping unrecognized file type: %s", p)
            continue

    if num_layers is None:
        raise ValueError(
            "Could not determine num_layers from any loaded file. "
            f"Paths: {[str(p) for p in paths]}"
        )
    assert mixer_names is not None

    if deduplicate:
        seen: set[tuple[tuple[str, ...], float]] = set()
        deduped_assignments: list[tuple[str, ...]] = []
        deduped_scores: list[float] = []
        deduped_allocations: list[dict[str, int]] = []
        for a, s, alloc in zip(all_assignments, all_scores, all_allocations):
            key = (a, s)
            if key not in seen:
                seen.add(key)
                deduped_assignments.append(a)
                deduped_scores.append(s)
                deduped_allocations.append(alloc)
        n_removed = len(all_assignments) - len(deduped_assignments)
        if n_removed > 0:
            logger.info(
                "Deduplication removed %d / %d records",
                n_removed,
                len(all_assignments),
            )
        all_assignments = deduped_assignments
        all_scores = deduped_scores
        all_allocations = deduped_allocations

    return PooledEvaluationData(
        assignments=all_assignments,
        scores=all_scores,
        allocations=all_allocations,
        num_layers=num_layers,
        mixer_names=mixer_names,
    )


def resolve_supplemental_dirs(
    dirs: Sequence[str | Path],
    objective_name: str,
) -> list[Path]:
    """Resolve supplemental directories to per-objective snowball JSONL paths.

    For each directory, recursively finds ``evaluations_{objective_name}.jsonl``
    snowball files (sorted by modification time within each directory).
    Deduplicates by absolute path across directories.
    """
    files: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        for f in sorted(
            Path(d).rglob(f"evaluations_{objective_name}.jsonl"),
            key=lambda p: p.stat().st_mtime,
        ):
            key = str(f)
            if key not in seen:
                files.append(f)
                seen.add(key)
    return files
