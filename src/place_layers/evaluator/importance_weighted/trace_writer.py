"""Trace writing and state reconstruction for precomputation.

Key design principle: JSONL trace files are the single source of truth.
State (which problems are complete) is always reconstructed by scanning
the trace files. This eliminates the possibility of state divergence.

Directory structure:
    output_dir/
        metadata.json       # Config hash, teacher model, samples_per_problem
        gsm8k.jsonl         # Traces for gsm8k task
        triviaqa.jsonl      # Traces for triviaqa task
        ...
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from place_layers.evaluator.importance_weighted.traces import (
    SkippedProblem,
    Trace,
    TraceRecord,
    TraceRecordAdapter,
)

logger = logging.getLogger(__name__)


class PrecomputeMetadata(BaseModel):
    """Metadata for a trace precomputation run.

    This is the only persisted state besides the traces themselves.
    It's used for:
    1. Validating config hash on resumption
    2. Providing context when loading traces
    """

    model_config = {"frozen": True}

    config_hash: str = Field(description="Hash of config for resumption validation")
    teacher_model: str = Field(description="Teacher model identifier")
    samples_per_problem: int = Field(description="Target samples per problem")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this run was started",
    )


class TraceWriter:
    """Appends traces to per-task JSONL files.

    This is intentionally simple - it just writes traces.
    State reconstruction is handled separately by scanning the files.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def append(self, task_name: str, records: list[TraceRecord]) -> None:
        """Append trace records to task's JSONL file."""
        path = self.output_dir / f"{task_name}.jsonl"
        with path.open("a") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")


def compute_config_hash(config: dict) -> str:
    """Compute a hash of configuration for resumption validation."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def load_metadata(output_dir: Path) -> PrecomputeMetadata | None:
    """Load metadata from output directory, if it exists."""
    path = output_dir / "metadata.json"
    if not path.exists():
        return None
    return PrecomputeMetadata.model_validate_json(path.read_text())


def save_metadata(output_dir: Path, metadata: PrecomputeMetadata) -> None:
    """Save metadata to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata.json"
    path.write_text(metadata.model_dump_json(indent=2))


def _extract_problem_idx(problem_id: str) -> int:
    """Extract problem index from problem_id like 'gsm8k:42'."""
    match = re.search(r":(\d+)$", problem_id)
    if not match:
        raise ValueError(f"Cannot extract problem index from '{problem_id}'")
    return int(match.group(1))


def get_completed_problems(output_dir: Path, task_name: str) -> set[int]:
    """Scan JSONL to find completed problem indices.

    This is the authoritative way to determine what work has been done.
    The traces on disk are the single source of truth. Both successful
    traces and skipped problems count as "completed" (no retry needed).

    Args:
        output_dir: Directory containing trace files.
        task_name: Name of the task to check.

    Returns:
        Set of problem indices that have been processed (traced or skipped).
    """
    jsonl_path = output_dir / f"{task_name}.jsonl"
    if not jsonl_path.exists():
        return set()

    completed: set[int] = set()
    malformed_lines = 0

    with jsonl_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = TraceRecordAdapter.validate_json(line)
                match record:
                    case Trace(problem_id=pid) | SkippedProblem(problem_id=pid):
                        problem_idx = _extract_problem_idx(pid)
                        completed.add(problem_idx)
            except Exception as e:
                malformed_lines += 1
                logger.warning(f"Malformed line {line_num} in {jsonl_path}: {e}")

    if malformed_lines > 0:
        logger.warning(f"Task '{task_name}': {malformed_lines} malformed lines ignored")

    return completed


def get_all_completed_problems(output_dir: Path) -> dict[str, set[int]]:
    """Scan all JSONL files to find completed problems per task.

    Args:
        output_dir: Directory containing trace files.

    Returns:
        Dict mapping task_name -> set of completed problem indices.
    """
    completed: dict[str, set[int]] = {}

    for jsonl_path in output_dir.glob("*.jsonl"):
        task_name = jsonl_path.stem
        completed[task_name] = get_completed_problems(output_dir, task_name)
        if completed[task_name]:
            logger.info(
                f"Reconstructed task '{task_name}': "
                f"{len(completed[task_name])} problems completed"
            )

    return completed
