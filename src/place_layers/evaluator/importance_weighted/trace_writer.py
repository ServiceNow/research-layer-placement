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
    BaseGeneration,
    SkippedProblem,
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
    """Appends traces to validation.jsonl and skipped problems to skipped.jsonl.

    This structure enables direct loading via HuggingFace datasets:
        load_dataset(output_dir, split='validation')

    State reconstruction scans both files, filtering by task_name field.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def append(self, records: list[TraceRecord]) -> None:
        """Append records to appropriate files based on record type."""
        # Traces and Generations (BaseGeneration subclasses) go to validation.jsonl
        valid = [r for r in records if isinstance(r, BaseGeneration)]
        skipped = [r for r in records if isinstance(r, SkippedProblem)]

        if valid:
            with (self.output_dir / "validation.jsonl").open("a") as f:
                for record in valid:
                    f.write(record.model_dump_json() + "\n")

        if skipped:
            with (self.output_dir / "skipped.jsonl").open("a") as f:
                for record in skipped:
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


def _scan_completed_problems(output_dir: Path) -> tuple[dict[str, set[int]], int]:
    """Scan trace files and return completed problems grouped by task.

    Internal helper that does the actual file scanning once.

    Returns:
        Tuple of (dict mapping task_name -> set of problem indices, malformed line count).
    """
    completed: dict[str, set[int]] = {}
    malformed_lines = 0

    for jsonl_path in [output_dir / "validation.jsonl", output_dir / "skipped.jsonl"]:
        if not jsonl_path.exists():
            continue

        with jsonl_path.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = TraceRecordAdapter.validate_json(line)
                    match record:
                        # BaseGeneration covers both Trace and Generation
                        case BaseGeneration(task_name=tn, problem_id=pid) | SkippedProblem(task_name=tn, problem_id=pid):
                            if tn not in completed:
                                completed[tn] = set()
                            problem_idx = _extract_problem_idx(pid)
                            completed[tn].add(problem_idx)
                except Exception as e:
                    malformed_lines += 1
                    logger.warning(f"Malformed line {line_num} in {jsonl_path}: {e}")

    return completed, malformed_lines


def get_completed_problems(output_dir: Path, task_name: str) -> set[int]:
    """Scan validation.jsonl and skipped.jsonl to find completed problem indices.

    This is the authoritative way to determine what work has been done.
    The traces on disk are the single source of truth. Both successful
    traces and skipped problems count as "completed" (no retry needed).

    Args:
        output_dir: Directory containing trace files.
        task_name: Name of the task to check.

    Returns:
        Set of problem indices that have been processed (traced or skipped).
    """
    completed, malformed_lines = _scan_completed_problems(output_dir)

    if malformed_lines > 0:
        logger.warning(f"Task '{task_name}': {malformed_lines} malformed lines ignored")

    return completed.get(task_name, set())


def get_all_completed_problems(output_dir: Path) -> dict[str, set[int]]:
    """Scan validation.jsonl and skipped.jsonl to find completed problems per task.

    Args:
        output_dir: Directory containing trace files.

    Returns:
        Dict mapping task_name -> set of completed problem indices.
    """
    completed, malformed_lines = _scan_completed_problems(output_dir)

    if malformed_lines > 0:
        logger.warning(f"{malformed_lines} malformed lines ignored")

    for task_name, indices in completed.items():
        logger.info(f"Reconstructed task '{task_name}': {len(indices)} problems completed")

    return completed


# =============================================================================
# Generation Summary
# =============================================================================


class TaskStats(BaseModel):
    """Per-task statistics for generation evaluation."""

    model_config = {"frozen": True}

    num_problems: int = Field(description="Number of unique problems")
    num_samples: int = Field(description="Total number of samples")
    problems_correct: int = Field(description="Problems with at least one correct sample")
    samples_correct: int = Field(description="Number of correct samples")

    @property
    def problem_accuracy(self) -> float:
        """Pass@k accuracy for this task."""
        return self.problems_correct / self.num_problems if self.num_problems > 0 else 0.0

    @property
    def sample_accuracy(self) -> float:
        """Per-sample accuracy for this task."""
        return self.samples_correct / self.num_samples if self.num_samples > 0 else 0.0


class GenerationSummary(BaseModel):
    """Summary statistics for generation evaluation.

    Computed from Generation records on disk - the single source of truth.
    """

    model_config = {"frozen": True}

    # Per-task statistics
    task_stats: dict[str, TaskStats] = Field(description="Statistics per task")

    # Aggregate statistics
    total_problems: int = Field(description="Total problems across all tasks")
    total_samples: int = Field(description="Total samples across all tasks")
    problems_correct: int = Field(description="Problems with at least one correct sample")
    samples_correct: int = Field(description="Total correct samples")

    @property
    def problem_accuracy(self) -> float:
        """Pass@k accuracy: fraction of problems with any correct sample."""
        return self.problems_correct / self.total_problems if self.total_problems > 0 else 0.0

    @property
    def sample_accuracy(self) -> float:
        """Per-sample accuracy: fraction of samples that are correct."""
        return self.samples_correct / self.total_samples if self.total_samples > 0 else 0.0

    def __str__(self) -> str:
        lines = ["Generation Summary:"]
        for task_name, stats in sorted(self.task_stats.items()):
            lines.append(
                f"  {task_name}: {stats.problems_correct}/{stats.num_problems} problems "
                f"({stats.problem_accuracy:.1%}), "
                f"{stats.samples_correct}/{stats.num_samples} samples "
                f"({stats.sample_accuracy:.1%})"
            )
        lines.append(
            f"  TOTAL: {self.problems_correct}/{self.total_problems} problems "
            f"({self.problem_accuracy:.1%}), "
            f"{self.samples_correct}/{self.total_samples} samples "
            f"({self.sample_accuracy:.1%})"
        )
        return "\n".join(lines)


def summarize_generations(output_dir: Path) -> GenerationSummary:
    """Compute summary statistics from Generation records on disk.

    This is the authoritative way to compute evaluation metrics.
    The validation.jsonl file is the single source of truth.

    Args:
        output_dir: Directory containing validation.jsonl.

    Returns:
        GenerationSummary with per-task and aggregate statistics.
    """
    from collections import defaultdict

    validation_path = output_dir / "validation.jsonl"
    if not validation_path.exists():
        raise FileNotFoundError(f"No validation.jsonl found in {output_dir}")

    # Group records by task and problem
    # task_name -> problem_id -> list of correctness values
    task_problems: dict[str, dict[str, list[bool | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    malformed = 0

    with validation_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = TraceRecordAdapter.validate_json(line)
                if isinstance(record, BaseGeneration):
                    task_problems[record.task_name][record.problem_id].append(
                        record.correct
                    )
            except Exception as e:
                malformed += 1
                logger.warning(f"Malformed line {line_num}: {e}")

    if malformed > 0:
        logger.warning(f"{malformed} malformed lines ignored in summary")

    # Compute per-task statistics
    task_stats: dict[str, TaskStats] = {}
    total_problems = 0
    total_samples = 0
    total_problems_correct = 0
    total_samples_correct = 0

    for task_name, problems in task_problems.items():
        num_problems = len(problems)
        num_samples = sum(len(samples) for samples in problems.values())
        problems_correct = sum(
            1 for samples in problems.values()
            if any(s is True for s in samples)
        )
        samples_correct = sum(
            sum(1 for s in samples if s is True)
            for samples in problems.values()
        )

        task_stats[task_name] = TaskStats(
            num_problems=num_problems,
            num_samples=num_samples,
            problems_correct=problems_correct,
            samples_correct=samples_correct,
        )

        total_problems += num_problems
        total_samples += num_samples
        total_problems_correct += problems_correct
        total_samples_correct += samples_correct

    return GenerationSummary(
        task_stats=task_stats,
        total_problems=total_problems,
        total_samples=total_samples,
        problems_correct=total_problems_correct,
        samples_correct=total_samples_correct,
    )
