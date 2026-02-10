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
from typing import Sequence

from pydantic import BaseModel, Field

from place_layers.evaluator.traces.types import (
    BaseGeneration,
    ChatLikelihoodRecord,
    ISProblemRecord,
    SkippedProblem,
    TextLikelihoodRecord,
    TraceRecord,
    TraceRecordAdapter,
)

# Tuple for isinstance checks (union types don't work with isinstance)
_LikelihoodRecordTypes = (ChatLikelihoodRecord, TextLikelihoodRecord)

logger = logging.getLogger(__name__)


class OutputMetadata(BaseModel):
    """Metadata for trace output directories.

    Required fields ensure proper provenance tracking and resumption support.
    TraceWriter validates that required fields are provided.

    Used for:
    1. Validating config hash on resumption (prevents mixing incompatible runs)
    2. Providing provenance (model, placement) when loading traces
    3. Recording placement for supernet optimization
    """

    model_config = {"frozen": True}

    # Timing
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this output directory was created",
    )

    # Model identification (required)
    model_name: str = Field(
        description="Model name or path (required for provenance)",
    )
    model_revision: str | None = Field(
        default=None,
        description="Model revision (git commit, HF revision, etc.)",
    )

    # Placement (None for non-supernet models)
    placement: list[str] | None = Field(
        description="Layer placement (mixer at each layer). "
        "None for non-supernet models.",
    )

    # Resumption (required)
    config_hash: str = Field(
        description="Hash of config for resumption validation",
    )

    # Precomputation specific (optional)
    samples_per_problem: int | None = Field(
        default=None,
        description="Number of samples generated per problem",
    )


class TraceWriter:
    """Appends traces to validation.jsonl and skipped problems to skipped.jsonl.

    This structure enables direct loading via HuggingFace datasets:
        load_dataset(output_dir, split='validation')

    Owns metadata lifecycle:
    - Loads existing metadata on init (for resumption)
    - Merges with provided metadata
    - Saves merged metadata to disk

    State reconstruction scans both files, filtering by task_name field.

    """

    def __init__(
        self,
        output_dir: Path,
        metadata: OutputMetadata,
    ) -> None:
        """Initialize TraceWriter with required metadata.

        Args:
            output_dir: Directory for trace files (will be created if needed).
            metadata: Required metadata for this output directory. Must include
                model_name, placement, and config_hash.

        Raises:
            ValueError: If resuming with different config_hash.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata (if resuming)
        existing = self._load_metadata()

        # Validate config hash on resumption
        if existing is not None and existing.config_hash != metadata.config_hash:
            raise ValueError(
                f"Config hash mismatch: existing={existing.config_hash}, "
                f"current={metadata.config_hash}. Cannot resume with different config."
            )

        # Always use provided metadata (it's from current config)
        # For resumption, we've validated config_hash matches
        self._metadata = metadata
        self._save_metadata()

        self._is_resuming = existing is not None

        # Cache for completed problems (lazily populated on first access)
        # Maps task_name -> set of completed problem indices
        # Invalidated when new records are appended
        # Cache lifetime = TraceWriter lifetime = single evaluation
        # This optimization is critical for multi-task benchmarks like MMLU (50+ tasks):
        # without caching, checking completion status takes ~5sec (0.1sec * 50 tasks)
        # with caching, it takes ~0.1sec (one scan, then instant lookups)
        self._completed_cache: dict[str, set[int]] | None = None

    @property
    def is_resuming(self) -> bool:
        """Whether this writer is resuming from existing data."""
        return self._is_resuming

    @property
    def metadata(self) -> OutputMetadata | None:
        """Current metadata for this output directory."""
        return self._metadata

    def _load_metadata(self) -> OutputMetadata | None:
        """Load metadata from disk if it exists."""
        path = self.output_dir / "metadata.json"
        if not path.exists():
            return None
        return OutputMetadata.model_validate_json(path.read_text())

    def _save_metadata(self) -> None:
        """Save current metadata to disk."""
        if self._metadata is not None:
            path = self.output_dir / "metadata.json"
            path.write_text(self._metadata.model_dump_json(indent=2))

    def append(self, records: Sequence[TraceRecord]) -> None:
        """Append records to appropriate files based on record type."""
        # Traces, Generations, LikelihoodRecords, and ISProblemRecords go to validation.jsonl
        valid = [
            r for r in records
            if isinstance(r, (BaseGeneration, *_LikelihoodRecordTypes, ISProblemRecord))
        ]
        skipped = [r for r in records if isinstance(r, SkippedProblem)]

        if valid:
            with (self.output_dir / "validation.jsonl").open("a") as f:
                for record in valid:
                    f.write(record.model_dump_json() + "\n")

        if skipped:
            with (self.output_dir / "skipped.jsonl").open("a") as f:
                for record in skipped:
                    f.write(record.model_dump_json() + "\n")

        # Invalidate cache when new records are written
        if valid or skipped:
            self._completed_cache = None

    def get_completed_problems(self, task_name: str) -> set[int]:
        """Scan validation.jsonl and skipped.jsonl to find completed problem indices.

        This is the authoritative way to determine what work has been done.
        The traces on disk are the single source of truth. Both successful
        traces and skipped problems count as "completed" (no retry needed).

        Uses a cache to avoid re-scanning files for multiple tasks. The cache
        is populated on first access by scanning files once and extracting
        completion data for ALL tasks. Subsequent calls for different tasks
        use the cached data (instant lookup). The cache is invalidated when
        new records are appended.

        Args:
            task_name: Name of the task to check.

        Returns:
            Set of problem indices that have been processed (traced or skipped).
        """
        # Populate cache on first access (or after invalidation)
        if self._completed_cache is None:
            completed, malformed_lines = _scan_completed_problems(self.output_dir)
            self._completed_cache = completed

            if malformed_lines > 0:
                logger.warning(f"{malformed_lines} malformed lines ignored in trace files")

        return self._completed_cache.get(task_name, set())


def compute_config_hash(config: dict) -> str:
    """Compute a hash of configuration for resumption validation."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


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
                        # ChatLikelihoodRecord/TextLikelihoodRecord for multiple-choice
                        # ISProblemRecord for importance-weighted evaluation
                        case (
                            BaseGeneration(task_name=tn, problem_id=pid)
                            | SkippedProblem(task_name=tn, problem_id=pid)
                            | ChatLikelihoodRecord(task_name=tn, problem_id=pid)
                            | TextLikelihoodRecord(task_name=tn, problem_id=pid)
                            | ISProblemRecord(task_name=tn, problem_id=pid)
                        ):
                            if tn not in completed:
                                completed[tn] = set()
                            problem_idx = _extract_problem_idx(pid)
                            completed[tn].add(problem_idx)
                except Exception as e:
                    malformed_lines += 1
                    logger.warning(f"Malformed line {line_num} in {jsonl_path}: {e}")

    return completed, malformed_lines


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


class LikelihoodTaskStats(BaseModel):
    """Per-task statistics for likelihood (multiple-choice) evaluation."""

    model_config = {"frozen": True}

    num_problems: int = Field(description="Number of problems")
    num_correct: int = Field(description="Number correct (raw)")
    num_correct_norm: int = Field(description="Number correct (length-normalized)")

    @property
    def accuracy(self) -> float:
        """Raw accuracy."""
        return self.num_correct / self.num_problems if self.num_problems > 0 else 0.0

    @property
    def accuracy_norm(self) -> float:
        """Length-normalized accuracy."""
        return self.num_correct_norm / self.num_problems if self.num_problems > 0 else 0.0


class LikelihoodSummary(BaseModel):
    """Summary statistics for likelihood evaluation.

    Computed from LikelihoodRecord entries on disk - the single source of truth.
    """

    model_config = {"frozen": True}

    task_stats: dict[str, LikelihoodTaskStats] = Field(description="Statistics per task")
    total_problems: int = Field(description="Total problems across all tasks")
    total_correct: int = Field(description="Total correct (raw)")
    total_correct_norm: int = Field(description="Total correct (length-normalized)")

    @property
    def accuracy(self) -> float:
        """Raw accuracy across all tasks."""
        return self.total_correct / self.total_problems if self.total_problems > 0 else 0.0

    @property
    def accuracy_norm(self) -> float:
        """Length-normalized accuracy across all tasks."""
        return self.total_correct_norm / self.total_problems if self.total_problems > 0 else 0.0

    def __str__(self) -> str:
        lines = ["Likelihood Evaluation Summary:"]
        for task_name, stats in sorted(self.task_stats.items()):
            lines.append(
                f"  {task_name}: {stats.num_correct}/{stats.num_problems} "
                f"({stats.accuracy:.1%}), "
                f"norm={stats.num_correct_norm}/{stats.num_problems} "
                f"({stats.accuracy_norm:.1%})"
            )
        lines.append(
            f"  TOTAL: {self.total_correct}/{self.total_problems} "
            f"({self.accuracy:.1%}), "
            f"norm={self.total_correct_norm}/{self.total_problems} "
            f"({self.accuracy_norm:.1%})"
        )
        return "\n".join(lines)


def summarize_likelihood(output_dir: Path) -> LikelihoodSummary:
    """Compute summary statistics from LikelihoodRecord entries on disk.

    This is the authoritative way to compute likelihood evaluation metrics.
    The validation.jsonl file is the single source of truth.

    Args:
        output_dir: Directory containing validation.jsonl.

    Returns:
        LikelihoodSummary with per-task and aggregate statistics.
    """
    from collections import defaultdict

    validation_path = output_dir / "validation.jsonl"
    if not validation_path.exists():
        raise FileNotFoundError(f"No validation.jsonl found in {output_dir}")

    # Group records by task
    # task_name -> list of (correct, correct_norm) tuples
    task_results: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    malformed = 0

    with validation_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = TraceRecordAdapter.validate_json(line)
                if isinstance(record, _LikelihoodRecordTypes):
                    task_results[record.task_name].append(
                        (record.correct, record.correct_norm)
                    )
            except Exception as e:
                malformed += 1
                logger.warning(f"Malformed line {line_num}: {e}")

    if malformed > 0:
        logger.warning(f"{malformed} malformed lines ignored in likelihood summary")

    # Compute per-task statistics
    task_stats: dict[str, LikelihoodTaskStats] = {}
    total_problems = 0
    total_correct = 0
    total_correct_norm = 0

    for task_name, results in task_results.items():
        num_problems = len(results)
        num_correct = sum(1 for c, _ in results if c)
        num_correct_norm = sum(1 for _, cn in results if cn)

        task_stats[task_name] = LikelihoodTaskStats(
            num_problems=num_problems,
            num_correct=num_correct,
            num_correct_norm=num_correct_norm,
        )

        total_problems += num_problems
        total_correct += num_correct
        total_correct_norm += num_correct_norm

    return LikelihoodSummary(
        task_stats=task_stats,
        total_problems=total_problems,
        total_correct=total_correct,
        total_correct_norm=total_correct_norm,
    )


class ISTaskStats(BaseModel):
    """Per-task statistics for importance-weighted evaluation."""

    model_config = {"frozen": True}

    num_problems: int = Field(description="Number of problems")
    num_samples: int = Field(description="Total samples across all problems")
    is_accuracy: float = Field(description="Mean IS estimate of P(correct)")
    avg_ess: float = Field(description="Mean effective sample size")


class ISSummary(BaseModel):
    """Summary statistics for importance-weighted evaluation.

    Computed from ISProblemRecord entries on disk - the single source of truth.
    Grouped by task_name for proper weighted combination.
    """

    model_config = {"frozen": True}

    task_stats: dict[str, ISTaskStats] = Field(description="Statistics per task")
    total_problems: int = Field(description="Total problems across all tasks")
    total_samples: int = Field(description="Total samples across all tasks")

    def weighted_accuracy(self, weights: dict[str, float]) -> float:
        """Compute weighted accuracy using provided task weights.

        Args:
            weights: Dict mapping task_name -> weight (should sum to 1).

        Returns:
            Weighted accuracy across tasks.
        """
        return sum(
            self.task_stats[name].is_accuracy * weight
            for name, weight in weights.items()
            if name in self.task_stats
        )

    def __str__(self) -> str:
        lines = ["Importance-Weighted Evaluation Summary:"]
        for task_name, stats in sorted(self.task_stats.items()):
            lines.append(
                f"  {task_name}: IS_acc={stats.is_accuracy:.4f}, "
                f"ESS={stats.avg_ess:.1f}, "
                f"problems={stats.num_problems}, samples={stats.num_samples}"
            )
        lines.append(
            f"  TOTAL: {self.total_problems} problems, {self.total_samples} samples"
        )
        return "\n".join(lines)


def summarize_is(
    output_dir: Path,
    clip_min: float = 1e-300,
    clip_max: float = 1e300,
) -> ISSummary:
    """Compute summary statistics from ISProblemRecord entries on disk.

    This is the authoritative way to compute IS evaluation metrics.
    The validation.jsonl file is the single source of truth.

    IMPORTANT: This function RECOMPUTES IS estimates from raw log-probs
    stored in each record, applying the specified clipping bounds. The
    stored `is_correct_estimate` and `ess` fields in records are IGNORED
    (they are for observability only). This allows re-running with different
    clipping parameters without re-computing expensive model forward passes.

    Args:
        output_dir: Directory containing validation.jsonl.
        clip_min: Minimum importance weight (default: 1e-300, effectively no clipping).
        clip_max: Maximum importance weight (default: 1e300, effectively no clipping).

    Returns:
        ISSummary with per-task and aggregate statistics.
    """
    import math
    from collections import defaultdict

    # Local import to avoid circular dependency
    from place_layers.evaluator.importance_weighted import compute_is_estimate

    validation_path = output_dir / "validation.jsonl"
    if not validation_path.exists():
        raise FileNotFoundError(f"No validation.jsonl found in {output_dir}")

    log_clip_min = math.log(clip_min)
    log_clip_max = math.log(clip_max)

    # Group records by task_name
    # task_name -> list of (is_correct_estimate, ess, num_samples)
    task_results: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    malformed = 0

    with validation_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = TraceRecordAdapter.validate_json(line)
                if isinstance(record, ISProblemRecord):
                    # RECOMPUTE IS estimate from raw log-probs with current clipping
                    _, is_estimate, ess, _, _ = compute_is_estimate(
                        record.student_log_probs,
                        record.teacher_log_probs,
                        record.correct_flags,
                        log_clip_min,
                        log_clip_max,
                    )
                    task_results[record.task_name].append(
                        (is_estimate, ess, record.num_samples)
                    )
            except Exception as e:
                malformed += 1
                logger.warning(f"Malformed line {line_num}: {e}")

    if malformed > 0:
        logger.warning(f"{malformed} malformed lines ignored in IS summary")

    # Compute per-task statistics
    task_stats: dict[str, ISTaskStats] = {}
    total_problems = 0
    total_samples = 0

    for task_name, results in task_results.items():
        num_problems = len(results)
        num_samples = sum(ns for _, _, ns in results)
        is_accuracy = sum(ice for ice, _, _ in results) / num_problems if num_problems > 0 else 0.0
        avg_ess = sum(ess for _, ess, _ in results) / num_problems if num_problems > 0 else 0.0

        task_stats[task_name] = ISTaskStats(
            num_problems=num_problems,
            num_samples=num_samples,
            is_accuracy=is_accuracy,
            avg_ess=avg_ess,
        )

        total_problems += num_problems
        total_samples += num_samples

    return ISSummary(
        task_stats=task_stats,
        total_problems=total_problems,
        total_samples=total_samples,
    )
