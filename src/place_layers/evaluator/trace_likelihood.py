"""Trace-based likelihood evaluator for supernet evaluation.

This evaluator computes the average log-likelihood of pre-computed traces
under different model configurations. Unlike ImportanceWeightedEvaluator,
it does not use importance sampling - just reports mean log-likelihood.

Key Use Case:
    Measuring how well a subnet assigns probability to teacher-generated
    completions, without weighting by correctness or importance ratios.

Theory:
    For each problem with N traces, compute:
        mean_log_prob = (1/N) × Σᵢ log q(xᵢ)

    Final score is the mean across all problems (optionally weighted by task).

Normalization:
    Optionally apply linear normalization to map raw log-likelihoods to [0, 1]:
        score = (mean_log_prob - low) / (high - low)

    This is useful for:
    - Making scores interpretable (0 = baseline, 1 = target)
    - Combining with other metrics in multi-objective optimization
    - Setting meaningful optimization bounds

Comparison to ImportanceWeightedEvaluator:
    - ImportanceWeighted: Estimates accuracy via importance sampling
    - TraceLikelihood: Measures raw log-likelihood assignment

    ImportanceWeighted asks: "What's the estimated accuracy?"
    TraceLikelihood asks: "How probable are these completions under the model?"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Sequence

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from place_layers.backend.base import AsyncBackend, SyncBackend

from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.trace_based_evaluator import (
    ProblemTraces,
    TraceBasedEvaluator,
    TraceBasedEvaluatorConfig,
)
from place_layers.evaluator.traces.types import TraceLikelihoodProblemRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Normalization types
# =============================================================================


class LinearNormalization(BaseModel):
    """Linear normalization: maps [low, high] to [0, 1]."""

    type: Literal["linear"] = "linear"
    low: float = Field(
        description="Log-prob value that maps to 0.0 (baseline performance)"
    )
    high: float = Field(
        description="Log-prob value that maps to 1.0 (target performance)"
    )

    @field_validator("low", "high")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        """Ensure values are finite."""
        if not (-float("inf") < v < float("inf")):
            raise ValueError(f"Normalization bounds must be finite, got {v}")
        return v

    def normalize(self, value: float) -> float:
        """Apply linear normalization to a value.

        Args:
            value: Raw log-likelihood value.

        Returns:
            Normalized value in [0, 1] (may be outside if value is outside [low, high]).
        """
        if self.high == self.low:
            # Avoid division by zero - return midpoint
            return 0.5
        return (value - self.low) / (self.high - self.low)


class NoNormalization(BaseModel):
    """No normalization: return raw log-likelihood."""

    type: Literal["none"] = "none"

    def normalize(self, value: float) -> float:
        """Return value unchanged."""
        return value


# Discriminated union for normalization types
Normalization = Annotated[
    LinearNormalization | NoNormalization,
    Field(discriminator="type"),
]


class TraceLikelihoodTaskSummary(BaseModel):
    """Summary statistics for a single task."""

    task_name: str
    num_problems: int
    num_samples: int
    mean_log_prob: float

    def __str__(self) -> str:
        return (
            f"  {self.task_name}: "
            f"{self.num_problems} problems, "
            f"{self.num_samples} samples, "
            f"mean_log_prob={self.mean_log_prob:.4f}"
        )


class TraceLikelihoodSummary(BaseModel):
    """Summary statistics across all tasks."""

    tasks: list[TraceLikelihoodTaskSummary]

    def weighted_mean_log_prob(self, task_weights: dict[str, float]) -> float:
        """Compute weighted mean log-prob across tasks.

        Args:
            task_weights: Dict mapping task_name -> normalized weight.

        Returns:
            Weighted mean log-probability.
        """
        total = 0.0
        for task_summary in self.tasks:
            weight = task_weights.get(task_summary.task_name, 0.0)
            total += weight * task_summary.mean_log_prob
        return total

    def __str__(self) -> str:
        lines = ["Trace Likelihood Summary:"]
        lines.extend(str(task) for task in self.tasks)
        return "\n".join(lines)


def summarize_trace_likelihood(output_dir: Path) -> TraceLikelihoodSummary:
    """Summarize trace likelihood results from validation.jsonl.

    Args:
        output_dir: Directory containing validation.jsonl.

    Returns:
        TraceLikelihoodSummary with per-task statistics.
    """
    import json

    validation_file = output_dir / "validation.jsonl"
    if not validation_file.exists():
        return TraceLikelihoodSummary(tasks=[])

    # Group records by task
    by_task: dict[str, list[TraceLikelihoodProblemRecord]] = {}

    with open(validation_file) as f:
        for line in f:
            record = json.loads(line)
            # Skip metadata and skipped problems
            if "mean_log_prob" not in record:
                continue

            task_name = record["task_name"]
            if task_name not in by_task:
                by_task[task_name] = []
            by_task[task_name].append(TraceLikelihoodProblemRecord(**record))

    # Compute per-task summaries
    task_summaries = []
    for task_name in sorted(by_task.keys()):
        records = by_task[task_name]
        num_problems = len(records)
        num_samples = sum(r.num_samples for r in records)
        mean_log_prob = sum(r.mean_log_prob for r in records) / num_problems if num_problems > 0 else 0.0

        task_summaries.append(
            TraceLikelihoodTaskSummary(
                task_name=task_name,
                num_problems=num_problems,
                num_samples=num_samples,
                mean_log_prob=mean_log_prob,
            )
        )

    return TraceLikelihoodSummary(tasks=task_summaries)


class TraceLikelihoodEvaluator(TraceBasedEvaluator):
    """Evaluator using trace-based average log-likelihood.

    Loads a supernet student model and pre-computed traces from one or more
    directories. Traces are merged by task_name, then for each placement
    configuration, computes average log-likelihood by:
    1. Setting mixer assignments on the model
    2. For each problem: compute student log-probs, average them
    3. Stream TraceLikelihoodProblemRecord to disk for each problem
    4. Aggregate per-problem means into final score (optionally weighted by task)
    5. Optionally apply normalization to map to [0, 1]
    """

    config: "TraceLikelihoodEvaluatorConfig"

    def __init__(
        self,
        config: "TraceLikelihoodEvaluatorConfig",
        *,
        output_dir: Path,
        backend: "AsyncBackend | SyncBackend",
    ) -> None:
        # Store normalization config
        self._normalization = config.normalization

        # Call parent init (sets up backend, trace loading, task weights, etc.)
        super().__init__(config, output_dir=output_dir, backend=backend)

    def _compute_problem_record(
        self,
        problem: ProblemTraces,
        student_log_probs: Sequence[float],
    ) -> TraceLikelihoodProblemRecord:
        """Compute mean log-likelihood for a single problem and create record.

        Args:
            problem: The problem's traces.
            student_log_probs: Student log-probs for each trace (same order as problem.traces).

        Returns:
            TraceLikelihoodProblemRecord with per-sample data and mean log-prob.
        """
        sample_indices = [trace.sample_idx for trace in problem.traces]
        mean_log_prob = sum(student_log_probs) / len(student_log_probs) if student_log_probs else 0.0

        return TraceLikelihoodProblemRecord(
            problem_id=problem.problem_id,
            task_name=problem.task_name,
            sample_indices=tuple(sample_indices),
            log_probs=tuple(student_log_probs),
            mean_log_prob=mean_log_prob,
            num_samples=len(sample_indices),
        )

    def _compute_summary(self) -> float:
        """Compute final mean log-likelihood score from validation.jsonl.

        Returns:
            Weighted mean log-likelihood (normalized if normalization is configured).
        """
        summary = summarize_trace_likelihood(self._output_dir)

        # Compute weighted mean log-likelihood using pre-computed task weights
        raw_score = summary.weighted_mean_log_prob(self.task_weights)

        # Apply normalization if configured
        final_score = self._normalization.normalize(raw_score)

        logger.info(f"\n{summary}")
        logger.info(f"Weighted mean log-likelihood (raw): {raw_score:.4f}")
        if isinstance(self._normalization, LinearNormalization):
            logger.info(
                f"Normalization: [{self._normalization.low:.2f}, {self._normalization.high:.2f}] → [0, 1]"
            )
            logger.info(f"Normalized score: {final_score:.4f}")
        else:
            logger.info(f"Final score: {final_score:.4f}")

        return final_score


class TraceLikelihoodEvaluatorConfig(TraceBasedEvaluatorConfig):
    """Configuration for trace-based likelihood evaluator."""

    _impl_class: ClassVar[type[BaseEvaluator]] = TraceLikelihoodEvaluator

    evaluator_type: Literal["trace_likelihood"] = "trace_likelihood"

    # Normalization configuration
    normalization: Normalization = Field(
        default=NoNormalization(),
        description="Score normalization. Use 'linear' to map [low, high] to [0, 1], "
        "or 'none' for raw log-likelihoods.",
    )
