"""Importance-weighted accuracy estimator for supernet evaluation.

This evaluator uses importance sampling to estimate subnet accuracy without
generation. The key insight: sample reasoning traces from teacher once,
then reweight using student log-probabilities.

Theory:
    We want E_q[correct(x)] for subnet q, but sampling from q is expensive.
    Instead, sample from teacher p and reweight:
        E_q[correct(x)] = E_p[correct(x) × q(x)/p(x)]

    Using self-normalized importance sampling per problem:
        score_problem ≈ Σᵢ [correct(xᵢ) × wᵢ] / Σᵢ wᵢ
    where wᵢ = exp(log q(xᵢ) - log p(xᵢ))

    Final accuracy is the mean of per-problem IS estimates.

Multi-Task Support:
    Traces from multiple source directories are merged by task_name.
    Each task gets its own IS estimate, then combined with optional
    user-specified weights:

        final_score = Σⱼ αⱼ × score_q(task_j)

    If task_weights is not specified, all tasks are weighted equally.

Numerical Stability:
    For long sequences, log-probabilities can be very negative (e.g., -5000).
    The log importance weight (log q - log p) is computed and clipped in
    log-space before exponentiating to avoid underflow/overflow.

Streaming & Resumption:
    Per-problem ISProblemRecord is streamed to validation.jsonl as problems
    are evaluated. This enables resumption from interruptions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Self, Sequence

from pydantic import Field, model_validator

from place_layers.core.types import LayerPlacement
from place_layers.evaluator.backend_evaluator import BackendEvaluator, BackendEvaluatorConfig
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.traces.writer import (
    get_completed_problems,
    summarize_is,
)
from place_layers.evaluator.traces.types import (
    ISProblemRecord,
    Trace,
    TraceDataset,
)
from place_layers.evaluator.progress import create_progress

logger = logging.getLogger(__name__)


@dataclass
class ProblemTraces:
    """Traces for a single problem, grouped for IS computation."""

    problem_id: str
    task_name: str
    traces: list[Trace] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return len(self.traces)


@dataclass
class MergedTraces:
    """Traces merged from all sources, grouped by task_name."""

    # task_name -> problem_id -> ProblemTraces
    by_task: dict[str, dict[str, ProblemTraces]] = field(default_factory=dict)

    def add_trace(self, trace: "Trace") -> None:
        """Add a trace to the merged collection."""
        if trace.task_name not in self.by_task:
            self.by_task[trace.task_name] = {}
        task_problems = self.by_task[trace.task_name]
        if trace.problem_id not in task_problems:
            task_problems[trace.problem_id] = ProblemTraces(
                problem_id=trace.problem_id,
                task_name=trace.task_name,
            )
        task_problems[trace.problem_id].traces.append(trace)

    @property
    def task_names(self) -> list[str]:
        """Get all task names."""
        return list(self.by_task.keys())

    def num_problems(self, task_name: str) -> int:
        """Get number of problems for a task."""
        return len(self.by_task.get(task_name, {}))

    def num_traces(self, task_name: str) -> int:
        """Get number of traces for a task."""
        return sum(
            len(p.traces) for p in self.by_task.get(task_name, {}).values()
        )


class ImportanceWeightedEvaluator(BackendEvaluator):
    """Evaluator using importance-weighted accuracy estimation.

    Loads a supernet student model and pre-computed teacher traces from
    one or more directories. Traces are merged by task_name, then for each
    placement configuration, estimates accuracy by:
    1. Setting mixer assignments on the model
    2. For each problem: compute student log-probs, IS estimate, ESS
    3. Stream ISProblemRecord to disk for each problem
    4. Aggregate per-problem estimates into final score (optionally weighted by task)

    This is much faster than generation (~10-100x) because:
    - No autoregressive decoding (just forward passes)
    - Same traces reused for all configurations
    - Batched computation of log-probabilities
    """

    config: "ImportanceWeightedEvaluatorConfig"

    def __init__(self, config: "ImportanceWeightedEvaluatorConfig") -> None:
        self._weight_clip_min = config.weight_clip_min
        self._weight_clip_max = config.weight_clip_max

        # Call parent init (sets up backend, trace writer, etc.)
        super().__init__(config)

        # Load and merge all traces by task_name
        self.merged_traces = self._load_and_merge_traces()

        # Compute normalized task weights
        self.task_weights = self._compute_task_weights()

        # Pre-compute log weight clip bounds for numerical stability
        self._log_weight_clip_min = math.log(self._weight_clip_min)
        self._log_weight_clip_max = math.log(self._weight_clip_max)

        logger.info(f"Saving IS results to: {self._output_dir}/validation.jsonl")

    def _load_and_merge_traces(self) -> MergedTraces:
        """Load traces from all sources and merge by task_name."""
        import time

        sources = self.config.trace_sources
        logger.info(f"Loading traces from {len(sources)} source(s)...")

        merged = MergedTraces()
        total_traces = 0

        for source_path in sources:
            path = Path(source_path)
            logger.info(f"Loading traces from {path}")
            start = time.perf_counter()
            dataset = TraceDataset.load_from_directory(path)
            elapsed = time.perf_counter() - start

            # Add all traces to merged collection
            for trace in dataset.traces:
                merged.add_trace(trace)

            total_traces += dataset.num_traces
            logger.info(
                f"  Loaded {dataset.num_traces} traces from {len(dataset.task_names)} task(s) "
                f"in {elapsed:.1f}s"
            )

        # Log summary by task
        logger.info(f"Merged {total_traces} traces across {len(merged.task_names)} task(s):")
        for task_name in sorted(merged.task_names):
            logger.info(
                f"  {task_name}: {merged.num_problems(task_name)} problems, "
                f"{merged.num_traces(task_name)} traces"
            )

        return merged

    def _compute_task_weights(self) -> dict[str, float]:
        """Compute normalized task weights.

        Returns a dict mapping task_name -> normalized weight.
        Handles missing/extra keys in config.task_weights with warnings.
        """
        task_names = set(self.merged_traces.task_names)
        config_weights = self.config.task_weights

        if config_weights is None:
            # Equal weights for all tasks
            weight = 1.0 / len(task_names) if task_names else 1.0
            return {task: weight for task in task_names}

        # Start with config weights, warn about issues
        weights: dict[str, float] = {}
        config_keys = set(config_weights.keys())

        # Check for missing tasks (in data but not in config)
        missing = task_names - config_keys
        if missing:
            logger.warning(
                f"task_weights missing keys for: {sorted(missing)}. Using weight=1.0 for these."
            )

        # Check for extra tasks (in config but not in data)
        extra = config_keys - task_names
        if extra:
            logger.warning(
                f"task_weights has keys not in data: {sorted(extra)}. Ignoring these."
            )

        # Build weights dict
        for task in task_names:
            weights[task] = config_weights.get(task, 1.0)

        # Normalize to sum to 1
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Sum of task weights must be positive")

        return {task: w / total for task, w in weights.items()}

    def _compute_problem_record(
        self,
        problem: ProblemTraces,
        student_log_probs: Sequence[float],
    ) -> ISProblemRecord:
        """Compute IS estimate for a single problem and create record.

        Args:
            problem: The problem's traces.
            student_log_probs: Student log-probs for each trace (same order as problem.traces).

        Returns:
            ISProblemRecord with per-sample data and problem-level aggregates.
        """
        sample_indices: list[int] = []
        teacher_log_probs: list[float] = []
        correct_flags: list[bool | None] = []
        log_importance_weights: list[float] = []

        for trace, student_lp in zip(problem.traces, student_log_probs):
            sample_indices.append(trace.sample_idx)
            teacher_log_probs.append(trace.teacher_log_prob)
            correct_flags.append(trace.correct)

            # Log importance weight: log w = log q - log p
            log_weight = student_lp - trace.teacher_log_prob

            # Clip in log-space for numerical stability
            log_weight = max(
                self._log_weight_clip_min,
                min(self._log_weight_clip_max, log_weight),
            )
            log_importance_weights.append(log_weight)

        # Compute importance weights
        weights = [math.exp(lw) for lw in log_importance_weights]

        # Compute IS estimate: sum(w * correct) / sum(w)
        weight_sum = sum(weights)
        if weight_sum == 0:
            is_correct_estimate = 0.0
        else:
            weighted_correct = sum(
                w * (1.0 if c else 0.0) for w, c in zip(weights, correct_flags)
            )
            is_correct_estimate = weighted_correct / weight_sum

        # Compute ESS: (sum(w))^2 / sum(w^2)
        weight_sq_sum = sum(w * w for w in weights)
        if weight_sq_sum == 0:
            ess = 0.0
        else:
            ess = (weight_sum * weight_sum) / weight_sq_sum

        return ISProblemRecord(
            problem_id=problem.problem_id,
            task_name=problem.task_name,
            sample_indices=tuple(sample_indices),
            student_log_probs=tuple(student_log_probs),
            teacher_log_probs=tuple(teacher_log_probs),
            correct_flags=tuple(correct_flags),
            log_importance_weights=tuple(log_importance_weights),
            is_correct_estimate=is_correct_estimate,
            ess=ess,
            weight_sum=weight_sum,
            weight_range=(min(weights), max(weights)) if weights else (0.0, 0.0),
            num_samples=len(weights),
        )

    async def _evaluate_task(
        self,
        task_name: str,
        problems: dict[str, ProblemTraces],
        completed_indices: set[int],
    ) -> None:
        """Evaluate a single task, streaming per-problem records to disk.

        Args:
            task_name: Name of the task (e.g., 'gsm8k').
            problems: Dict mapping problem_id -> ProblemTraces.
            completed_indices: Set of problem indices already completed (for resumption).
        """
        problem_ids = list(problems.keys())

        # Filter to remaining problems
        remaining_problems: list[ProblemTraces] = []
        for pid in problem_ids:
            # Extract problem index from problem_id (format: "task_name:idx")
            try:
                idx = int(pid.split(":")[-1])
                if idx not in completed_indices:
                    remaining_problems.append(problems[pid])
            except ValueError:
                # If we can't parse the index, include it (conservative)
                remaining_problems.append(problems[pid])

        if not remaining_problems:
            logger.info(f"  Task '{task_name}': all {len(problem_ids)} problems already completed")
            return

        logger.info(
            f"  Task '{task_name}': processing {len(remaining_problems)} problems "
            f"({len(problem_ids) - len(remaining_problems)} already completed)"
        )

        # Collect all traces for batch log-prob computation
        all_traces: list[Trace] = []
        problem_trace_ranges: list[tuple[int, int]] = []  # (start, end) for each problem

        for problem in remaining_problems:
            start = len(all_traces)
            all_traces.extend(problem.traces)
            end = len(all_traces)
            problem_trace_ranges.append((start, end))

        # Compute all log-probs in batch
        traces_data: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
            (trace.prompt_tokens, trace.completion_tokens) for trace in all_traces
        ]

        all_student_log_probs = await self._compute_batch_log_probs(traces_data)

        # Process each problem, create records, and stream to disk
        records: list[ISProblemRecord] = []

        with create_progress() as progress:
            progress_task = progress.add_task(
                f"[cyan]{task_name}",
                total=len(remaining_problems),
            )

            for problem, (start, end) in zip(remaining_problems, problem_trace_ranges):
                problem_log_probs = all_student_log_probs[start:end]
                record = self._compute_problem_record(problem, problem_log_probs)
                records.append(record)
                progress.advance(progress_task)

        # Stream all records to disk
        self._trace_writer.append(records)  # type: ignore[arg-type]

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate a placement using stratified importance-weighted accuracy.

        Computes per-problem IS estimates, streams records to disk, and
        combines with normalized task weights.

        Args:
            parameters: Layer placement to evaluate, or None for default config.
                       When None, no placement switching is performed (useful for
                       non-supernet models or backends that don't support placement).

        Returns:
            Estimated accuracy (higher is better, in [0, 1]).
        """
        import time

        if parameters is not None:
            logger.info(
                f"Evaluating placement: {parameters.assignments[:4]}... "
                f"({len(parameters.assignments)} layers)"
            )
            await self._set_placement(parameters)
        else:
            logger.info("No placement specified, using default configuration")

        # Reset inference duration for this evaluation
        self._last_inference_duration = 0.0

        # Evaluate each task
        for task_name in sorted(self.merged_traces.task_names):
            problems = self.merged_traces.by_task[task_name]
            logger.info(
                f"Evaluating task '{task_name}': "
                f"{len(problems)} problems, {self.merged_traces.num_traces(task_name)} traces"
            )

            # Get completed problems for resumption (not timed)
            completed = get_completed_problems(self._output_dir, task_name)

            # Time only the inference
            start = time.perf_counter()
            await self._evaluate_task(task_name, problems, completed)
            self._last_inference_duration += time.perf_counter() - start

        # Compute summary from disk (single source of truth)
        summary = summarize_is(self._output_dir)

        # Compute weighted accuracy using pre-computed task weights
        final_score = summary.weighted_accuracy(self.task_weights)

        logger.info(f"\n{summary}")
        logger.info(f"Weighted accuracy: {final_score:.4f} (in {self._last_inference_duration:.1f}s)")

        return final_score

    async def evaluate_batch(
        self, parameters_list: list[LayerPlacement]
    ) -> list[float]:
        """Evaluate multiple placements efficiently.

        For importance-weighted evaluation, most of the cost is in the forward
        passes through the model. Since we need different mixer assignments for
        each placement, we can't batch across placements. However, we still
        benefit from batching traces within each placement evaluation.

        Args:
            parameters_list: List of placements to evaluate.

        Returns:
            List of scores in the same order as input.
        """
        # For now, evaluate sequentially since each placement requires
        # different mixer assignments. Future optimization could cache
        # intermediate activations or use more sophisticated batching.
        scores = []
        for parameters in parameters_list:
            score = await self.evaluate(parameters)
            scores.append(score)
        return scores


class ImportanceWeightedEvaluatorConfig(BackendEvaluatorConfig):
    """Configuration for importance-weighted accuracy evaluator."""

    _impl_class: ClassVar[type[BaseEvaluator]] = ImportanceWeightedEvaluator

    evaluator_type: Literal["importance_weighted"] = "importance_weighted"

    # Trace sources - list of paths to directories containing JSONL files
    trace_sources: list[str] = Field(
        description="List of paths to trace directories (merged by task_name)"
    )

    # Optional per-task weights for weighted accuracy
    task_weights: dict[str, float] | None = Field(
        default=None,
        description="Per-task importance weights (task_name -> weight). "
        "If None, all tasks are weighted equally. "
        "Missing tasks default to 1.0 with a warning. "
        "Extra keys are ignored with a warning.",
    )

    # Importance sampling parameters
    weight_clip_min: float = Field(
        description="Minimum importance weight (prevents near-zero weights)",
        gt=0.0,
    )
    weight_clip_max: float = Field(
        description="Maximum importance weight (prevents extreme weights)",
        gt=0.0,
    )

    @model_validator(mode="after")
    def validate_trace_sources(self) -> Self:
        """Ensure at least one trace source is provided."""
        if not self.trace_sources:
            raise ValueError("At least one trace source is required")
        return self

    @model_validator(mode="after")
    def validate_task_weights(self) -> Self:
        """Ensure task weights are positive if provided."""
        if self.task_weights:
            for task, weight in self.task_weights.items():
                if weight <= 0:
                    raise ValueError(
                        f"Task weight must be positive, got {weight} for '{task}'"
                    )
        return self
