"""Evaluation data logging and pooling for surrogate training.

Provides:
- ``EvaluationRecord`` / ``EvaluationLogMeta``: Pydantic models for JSONL
  evaluation logs and their sidecar metadata.
- ``append_record`` / ``load_records``: write/read individual JSONL files.
- ``PooledEvaluationData``: merge data from multiple JSONL evaluation logs
  (use ``from_jsonl`` classmethod to load, ``merge`` to combine).
- ``discover_evaluation_records``: find per-objective evaluation record files
  across directories (supports both new and legacy naming conventions).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import IO, TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from place_layers.surrogate.dataset import ExplorationDataset

import yaml
from pydantic import BaseModel, computed_field, model_validator

logger = logging.getLogger(__name__)

EVALUATION_RECORDS_FILENAME = "evaluation_records.jsonl"


# ============================================================================
# JSONL record model
# ============================================================================


class EvaluationRecord(BaseModel):
    """Single evaluated placement. One per JSONL line.

    ``allocation`` is computed from ``assignment`` — no need to store it
    explicitly.  Old JSONL files with a stored ``allocation`` field are
    silently accepted (the stored value is ignored in favor of the computed
    one).
    """

    model_config = {"frozen": True}

    assignment: tuple[str, ...]
    score: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allocation(self) -> dict[str, int]:
        return dict(Counter(self.assignment))


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
    cost_floor: float | None = None
    cost_ceiling: float | None = None
    band_index: int | None = None


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
    """Public helper: ``evaluation_records.jsonl`` → ``evaluation_records.meta.yaml``."""
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
    """Merged data from multiple evaluation sources. Immutable view.

    Construct via:
    - ``from_jsonl(paths)`` — load and deduplicate from JSONL files
    - ``from_exploration(dataset)`` — lift an ``ExplorationDataset``
    - Direct construction for tests
    """

    model_config = {"frozen": True}

    assignments: list[tuple[str, ...]]
    scores: list[float]
    num_layers: int
    mixer_names: list[str]

    @model_validator(mode="before")
    @classmethod
    def _enforce_dedup(cls, data: dict) -> dict:
        assignments = data.get("assignments", [])
        scores = data.get("scores", [])
        if not assignments:
            return data
        seen: set[tuple[tuple[str, ...], float]] = set()
        deduped_a: list = []
        deduped_s: list = []
        for a, s in zip(assignments, scores):
            key = (tuple(a), s)
            if key not in seen:
                seen.add(key)
                deduped_a.append(a)
                deduped_s.append(s)
        n_removed = len(assignments) - len(deduped_a)
        if n_removed > 0:
            logger.info("Deduplication removed %d / %d records", n_removed, len(assignments))
        data["assignments"] = deduped_a
        data["scores"] = deduped_s
        return data

    @property
    def allocations(self) -> list[dict[str, int]]:
        """Compute allocations from assignments (never stored)."""
        return [dict(Counter(a)) for a in self.assignments]

    @property
    def num_samples(self) -> int:
        return len(self.assignments)

    def as_records(self) -> list[EvaluationRecord]:
        """Convert to list of EvaluationRecord for serialization."""
        return [
            EvaluationRecord(assignment=a, score=s)
            for a, s in zip(self.assignments, self.scores, strict=True)
        ]

    @classmethod
    def from_jsonl(cls, paths: Sequence[Path | str]) -> PooledEvaluationData:
        """Load and merge evaluation data from multiple JSONL files.

        Reads ``.jsonl`` evaluation log files via ``load_records()``.  Metadata
        (``num_layers``, ``mixer_names``) is obtained from ``.meta.yaml`` sidecars.
        At least one file must have a sidecar with this information.

        Deduplicates exact ``(assignment, score)`` pairs automatically.
        """
        all_assignments: list[tuple[str, ...]] = []
        all_scores: list[float] = []
        num_layers: int | None = None
        mixer_names: list[str] | None = None

        for raw_path in paths:
            p = Path(raw_path)
            if not p.exists():
                logger.warning("Skipping missing evaluation data file: %s", p)
                continue

            if p.suffix == ".jsonl":
                records = load_records(p)
                for rec in records:
                    all_assignments.append(rec.assignment)
                    all_scores.append(rec.score)
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
        if mixer_names is None:
            raise ValueError(
                "Could not determine mixer_names from any loaded file. "
                f"Paths: {[str(p) for p in paths]}"
            )

        return cls(
            assignments=all_assignments,
            scores=all_scores,
            num_layers=num_layers,
            mixer_names=mixer_names,
        )

    @classmethod
    def from_exploration(
        cls, dataset: ExplorationDataset,
    ) -> PooledEvaluationData:
        """Create from an ExplorationDataset."""
        return cls(
            assignments=list(dataset.assignments),
            scores=list(dataset.scores),
            num_layers=dataset.num_layers,
            mixer_names=list(dataset.mixer_names),
        )

    def merge(self, other: PooledEvaluationData) -> PooledEvaluationData:
        """Merge two pools, deduplicating (assignment, score) pairs.

        Self's entries take priority (appear first, preserved on collision).
        Returns a new frozen instance.
        """
        return PooledEvaluationData(
            assignments=list(self.assignments) + list(other.assignments),
            scores=list(self.scores) + list(other.scores),
            num_layers=self.num_layers,
            mixer_names=self.mixer_names,
        )


# ============================================================================
# Discovery
# ============================================================================


def discover_evaluation_records(
    dirs: Sequence[str | Path],
    objective_name: str,
    *,
    num_layers: int | None = None,
    mixer_names: list[str] | None = None,
    cost_floor: float | None = None,
    cost_ceiling: float | None = None,
) -> list[Path]:
    """Find evaluation record files matching constraints (recursive).

    For new-convention files (evaluation_records.jsonl): reads companion
    .meta.yaml sidecar, matches by objective_name and optional constraints.
    For legacy files (evaluations_{obj}.jsonl): matches by filename.

    Constraint semantics:
    - objective_name: exact match (required)
    - num_layers: exact match (if provided)
    - mixer_names: set equality (if provided)
    - cost_floor: meta.cost_floor >= this  (files with None cost_floor pass)
    - cost_ceiling: meta.cost_ceiling <= this  (files with None cost_ceiling pass)

    Cost bound semantics: files without cost bounds (exploration data) are
    ALWAYS included — they're unscoped.  Files with cost bounds are only
    included if their range is fully within the requested range.

    Deduplicates by absolute path.  Sorted by modification time.
    """
    files: list[Path] = []
    seen: set[str] = set()

    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue

        candidates: list[Path] = []

        # New convention: evaluation_records.jsonl with sidecar
        for f in d.rglob(EVALUATION_RECORDS_FILENAME):
            candidates.append(f)

        # Legacy convention: evaluations_{objective_name}.jsonl
        for f in d.rglob(f"evaluations_{objective_name}.jsonl"):
            candidates.append(f)

        # Sort by modification time within each directory
        candidates.sort(key=lambda p: p.stat().st_mtime)

        for f in candidates:
            abs_key = str(f.resolve())
            if abs_key in seen:
                continue

            # For new-convention files, check sidecar metadata
            if f.name == EVALUATION_RECORDS_FILENAME:
                meta = load_meta(f)
                if meta is None:
                    continue
                if meta.objective_name != objective_name:
                    continue
                if num_layers is not None and meta.num_layers != num_layers:
                    continue
                if mixer_names is not None and set(meta.mixer_names) != set(mixer_names):
                    continue
                # Cost bound filtering
                if cost_floor is not None and meta.cost_floor is not None:
                    if meta.cost_floor < cost_floor:
                        continue
                if cost_ceiling is not None and meta.cost_ceiling is not None:
                    if meta.cost_ceiling > cost_ceiling:
                        continue

            # Legacy files match by filename (already filtered by rglob pattern)

            files.append(f)
            seen.add(abs_key)

    return files
