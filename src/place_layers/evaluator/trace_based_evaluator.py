"""Base class for evaluators that use pre-computed traces.

This module provides common infrastructure for evaluators that:
- Load pre-computed traces from disk
- Merge traces by task_name from multiple sources
- Score traces using backend.score_tokens()
- Stream per-problem records to disk
- Support resumption via TraceWriter
- Compute task-weighted final scores

Concrete evaluators inherit from TraceBasedEvaluator and implement:
- _compute_problem_record(): How to process student log-probs for a problem
- _compute_summary(): How to aggregate problem records into final score
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from place_layers.backend import BackendError, ScoreTokensRequest
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.backend_evaluator import BackendEvaluator, BackendEvaluatorConfig
from place_layers.evaluator.traces.types import SkippedProblem, Trace, TraceDataset
from place_layers.evaluator.progress import create_progress

if TYPE_CHECKING:
    from typing import Self, Sequence

logger = logging.getLogger(__name__)


class ProblemTraces(BaseModel):
    """Traces for a single problem, grouped for evaluation."""

    problem_id: str
    task_name: str
    traces: list[Trace] = Field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return len(self.traces)


class MergedTraces(BaseModel):
    """Traces merged from all sources, grouped by task_name."""

    # task_name -> problem_id -> ProblemTraces
    by_task: dict[str, dict[str, ProblemTraces]] = Field(default_factory=dict)

    def add_trace(self, trace: Trace) -> None:
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


class TraceBasedEvaluator(BackendEvaluator):
    """Base class for evaluators using pre-computed traces.

    Provides common infrastructure:
    - Trace loading and merging from multiple sources
    - Backend scoring via score_tokens()
    - Streaming results to disk
    - Resumption support
    - Task weighting

    Subclasses must implement:
    - _compute_problem_record(): Process student log-probs into a record
    - _compute_summary(): Aggregate records into final score
    """

    config: "TraceBasedEvaluatorConfig"

    def __init__(
        self, config: "TraceBasedEvaluatorConfig", *, output_dir: Path
    ) -> None:
        # Call parent init (sets up backend, trace writer, etc.)
        super().__init__(config, output_dir=output_dir)

        # Load and merge all traces by task_name
        self.merged_traces = self._load_and_merge_traces()

        # Compute normalized task weights
        self.task_weights = self._compute_task_weights()

        logger.info(f"Saving results to: {output_dir}/validation.jsonl")

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

    @abstractmethod
    def _compute_problem_record(
        self,
        problem: ProblemTraces,
        student_log_probs: Sequence[float],
    ) -> Any:
        """Compute record for a single problem.

        Args:
            problem: The problem's traces.
            student_log_probs: Student log-probs for each trace (same order as problem.traces).

        Returns:
            A record object (type depends on subclass).
        """
        ...

    @abstractmethod
    def _compute_summary(self) -> float:
        """Compute final score from records in validation.jsonl.

        Returns:
            Final evaluation score.
        """
        ...

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

        # Track pending traces per problem: problem_id -> (problem, num_traces, [log_prob slots])
        pending: dict[str, tuple[ProblemTraces, int, list[float | None]]] = {}

        def make_requests():
            for problem in remaining_problems:
                num_traces = len(problem.traces)
                pending[problem.problem_id] = (problem, num_traces, [None] * num_traces)
                for i, trace in enumerate(problem.traces):
                    key = f"{problem.problem_id}:{i}"
                    yield ScoreTokensRequest(
                        key=key,
                        prompt_tokens=trace.prompt_tokens,
                        completion_tokens=trace.completion_tokens,
                    )

        with create_progress() as progress:
            progress_task = progress.add_task(
                f"[cyan]{task_name}",
                total=len(remaining_problems),
            )
            # Track problems that had errors (skip entire problem if any trace fails)
            failed_problems: set[str] = set()

            async for result in self.score_tokens(make_requests()):
                # Parse key to get problem_id and trace_idx
                problem_id, trace_idx_str = result.key.rsplit(":", 1)

                # Handle backend errors - skip entire problem if any trace fails
                if isinstance(result, BackendError):
                    if problem_id not in failed_problems:
                        failed_problems.add(problem_id)
                        logger.warning(
                            f"Skipping {problem_id}: {result.error_type} - {result.message}"
                        )
                        self._trace_writer.append([
                            SkippedProblem(
                                problem_id=problem_id,
                                task_name=task_name,
                                reason=f"{result.error_type}: {result.message}",
                            )
                        ])
                        if problem_id in pending:
                            del pending[problem_id]
                        progress.advance(progress_task)
                    continue

                # Skip if this problem already failed on another trace
                if problem_id in failed_problems:
                    continue

                trace_idx = int(trace_idx_str)

                # Record result
                problem, _, log_probs = pending[problem_id]
                log_probs[trace_idx] = result.log_prob

                # Check if all traces for this problem are complete
                if all(lp is not None for lp in log_probs):
                    # Cast to list[float] since we know all are non-None
                    final_log_probs = [lp for lp in log_probs if lp is not None]
                    record = self._compute_problem_record(problem, final_log_probs)
                    self._trace_writer.append([record])  # type: ignore[arg-type]
                    del pending[problem_id]
                    progress.advance(progress_task)

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate a placement using trace-based scoring.

        Args:
            parameters: Layer placement to evaluate, or None for default config.
                       When None, no placement switching is performed (useful for
                       non-supernet models or backends that don't support placement).

        Returns:
            Evaluation score (higher is better).
        """
        import time

        if parameters is not None:
            logger.info(
                f"Evaluating placement: {parameters.assignments[:4]}... "
                f"({len(parameters.assignments)} layers)"
            )
            await self._set_placement(parameters)
        self._init_trace_writer()

        # Evaluate each task
        start = time.perf_counter()

        for task_name in sorted(self.merged_traces.task_names):
            problems = self.merged_traces.by_task[task_name]
            logger.info(
                f"Evaluating task '{task_name}': "
                f"{len(problems)} problems, {self.merged_traces.num_traces(task_name)} traces"
            )

            # Get completed problems for resumption
            completed = self._trace_writer.get_completed_problems(task_name)

            await self._evaluate_task(task_name, problems, completed)

        elapsed = time.perf_counter() - start

        # Compute summary from disk (delegated to subclass)
        final_score = self._compute_summary()

        logger.info(f"Final score: {final_score:.4f} (in {elapsed:.1f}s)")

        return final_score

    async def evaluate_batch(
        self, parameters_list: list[LayerPlacement]
    ) -> list[float]:
        """Evaluate multiple placements efficiently.

        For trace-based evaluation, most of the cost is in the forward
        passes through the model. Since we need different mixer assignments for
        each placement, we can't batch across placements. However, we still
        benefit from batching traces within each placement evaluation.

        Args:
            parameters_list: List of placements to evaluate.

        Returns:
            List of scores in the same order as input.
        """
        scores = []
        for parameters in parameters_list:
            score = await self.evaluate(parameters)
            scores.append(score)
        return scores


class TraceBasedEvaluatorConfig(BackendEvaluatorConfig):
    """Base configuration for trace-based evaluators."""

    # Trace sources - list of paths to directories containing JSONL files
    trace_sources: list[str] = Field(
        description="List of paths to trace directories (merged by task_name)"
    )

    # Optional per-task weights for final score computation
    task_weights: dict[str, float] | None = Field(
        default=None,
        description="Per-task importance weights (task_name -> weight). "
        "If None, all tasks are weighted equally. "
        "Missing tasks default to 1.0 with a warning. "
        "Extra keys are ignored with a warning.",
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
