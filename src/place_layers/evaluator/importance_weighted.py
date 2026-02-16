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
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self, Sequence

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from place_layers.backend.base import AsyncBackend, SyncBackend

from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.trace_based_evaluator import (
    ProblemTraces,
    TraceBasedEvaluator,
    TraceBasedEvaluatorConfig,
)
from place_layers.evaluator.traces.writer import summarize_is
from place_layers.evaluator.traces.types import ISProblemRecord

logger = logging.getLogger(__name__)


def compute_is_estimate(
    student_log_probs: Sequence[float],
    teacher_log_probs: Sequence[float],
    correct_flags: Sequence[bool | None],
    log_clip_min: float,
    log_clip_max: float,
) -> tuple[tuple[float, ...], float, float, float, tuple[float, float]]:
    """Compute IS estimate from raw log-probs. Single source of truth.

    This function is the ONE codepath for IS computation, used by both:
    - ImportanceWeightedEvaluator._compute_problem_record() for initial computation
    - summarize_is() for re-aggregation with different clipping parameters

    Args:
        student_log_probs: Student model log-prob for each sample.
        teacher_log_probs: Teacher model log-prob for each sample.
        correct_flags: Correctness flag for each sample.
        log_clip_min: Minimum log importance weight (for clipping).
        log_clip_max: Maximum log importance weight (for clipping).

    Returns:
        Tuple of:
        - log_importance_weights: Raw (unclipped) log weights for observability
        - is_correct_estimate: IS estimate of P(correct)
        - ess: Effective sample size
        - weight_sum: Sum of (clipped) importance weights
        - weight_range: (min, max) of (clipped) importance weights
    """
    # Compute raw log importance weights (stored for observability)
    raw_log_weights: list[float] = []
    for s_lp, t_lp in zip(student_log_probs, teacher_log_probs):
        raw_log_weights.append(s_lp - t_lp)

    # Apply clipping for numerical stability when computing IS estimate
    clipped_log_weights = [
        max(log_clip_min, min(log_clip_max, lw)) for lw in raw_log_weights
    ]
    weights = [math.exp(lw) for lw in clipped_log_weights]

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

    weight_range = (min(weights), max(weights)) if weights else (0.0, 0.0)

    return (tuple(raw_log_weights), is_correct_estimate, ess, weight_sum, weight_range)


class ImportanceWeightedEvaluator(TraceBasedEvaluator):
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

    def __init__(
        self,
        config: "ImportanceWeightedEvaluatorConfig",
        *,
        output_dir: Path,
        shared_backend: "AsyncBackend | SyncBackend | None" = None,
    ) -> None:
        self._weight_clip_min = config.weight_clip_min
        self._weight_clip_max = config.weight_clip_max

        # Pre-compute log weight clip bounds for numerical stability
        self._log_weight_clip_min = math.log(self._weight_clip_min)
        self._log_weight_clip_max = math.log(self._weight_clip_max)

        # Call parent init (sets up backend, trace loading, task weights, etc.)
        super().__init__(config, output_dir=output_dir, shared_backend=shared_backend)
        logger.info(f"Saving IS results to: {output_dir}/validation.jsonl")

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
        # Extract data from traces
        sample_indices = [trace.sample_idx for trace in problem.traces]
        teacher_log_probs = [trace.teacher_log_prob for trace in problem.traces]
        correct_flags = [trace.correct for trace in problem.traces]

        # Use shared IS computation function
        log_importance_weights, is_correct_estimate, ess, weight_sum, weight_range = (
            compute_is_estimate(
                student_log_probs,
                teacher_log_probs,
                correct_flags,
                self._log_weight_clip_min,
                self._log_weight_clip_max,
            )
        )

        return ISProblemRecord(
            problem_id=problem.problem_id,
            task_name=problem.task_name,
            sample_indices=tuple(sample_indices),
            student_log_probs=tuple(student_log_probs),
            teacher_log_probs=tuple(teacher_log_probs),
            correct_flags=tuple(correct_flags),
            log_importance_weights=log_importance_weights,
            is_correct_estimate=is_correct_estimate,
            ess=ess,
            weight_sum=weight_sum,
            weight_range=weight_range,
            num_samples=len(sample_indices),
        )

    def _compute_summary(self) -> float:
        """Compute final IS accuracy score from validation.jsonl.

        Returns:
            Weighted IS accuracy.
        """
        # Compute summary from disk (single source of truth)
        # Pass clipping parameters so summarize_is recomputes with current settings
        summary = summarize_is(
            self._output_dir,
            clip_min=self._weight_clip_min,
            clip_max=self._weight_clip_max,
        )

        logger.info(f"\n{summary}")

        if self.config.scoring_mode == "student_log_prob":
            final_score = summary.weighted_student_log_prob(self.task_weights)
            logger.info(
                f"Mean student log-prob: {final_score:.1f}"
            )
        else:
            final_score = summary.weighted_accuracy(self.task_weights)
            logger.info(f"Weighted accuracy: {final_score:.4f}")

        return final_score


class ImportanceWeightedEvaluatorConfig(TraceBasedEvaluatorConfig):
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

    # Scoring mode
    scoring_mode: Literal["accuracy", "student_log_prob"] = Field(
        default="accuracy",
        description="What score to return from evaluate(). "
        "'accuracy': IS-weighted accuracy (higher=better, range [0,1]). "
        "'student_log_prob': mean student log-prob per problem (higher=better, "
        "negative values where less negative = better student fit).",
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
