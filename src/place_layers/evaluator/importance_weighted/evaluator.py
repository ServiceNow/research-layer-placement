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

Multi-Dataset Support:
    When evaluating across multiple benchmarks (MATH, AIME, GPQA, etc.),
    we use stratified importance sampling. Each dataset is a separate stratum
    with its own IS estimate, then combined with user-specified weights:

        final_score = Σⱼ αⱼ × score_q(Dⱼ)

    This avoids issues with naive concatenation where dataset size would
    implicitly determine importance.

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
from typing import ClassVar, Literal, Sequence

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field, model_validator

from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.base_evaluator_config import BaseEvaluatorConfig
from place_layers.evaluator.importance_weighted.generation_backend import (
    AsyncBackend,
    BackendConfig,
    SyncBackend,
)
from place_layers.evaluator.importance_weighted.trace_writer import (
    TraceWriter,
    get_completed_problems,
    summarize_is,
)
from place_layers.evaluator.importance_weighted.traces import (
    ISProblemRecord,
    Trace,
    TraceDataset,
)
from place_layers.evaluator.progress import create_progress

logger = logging.getLogger(__name__)


class TraceSource(BaseModel):
    """Configuration for a single trace dataset source.

    Each source represents a benchmark/dataset with pre-computed traces
    and a weight indicating its importance in the final score.
    """

    model_config = {"frozen": True}

    path: str = Field(description="Path to trace directory containing JSONL files")
    weight: float = Field(
        description="Importance weight for this dataset (will be normalized)",
        gt=0.0,
    )
    name: str | None = Field(
        description="Display name for logging (defaults to path basename)",
        default=None,
    )

    @property
    def display_name(self) -> str:
        """Get display name, defaulting to path basename."""
        if self.name:
            return self.name
        return Path(self.path).stem


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
class LoadedDataset:
    """A loaded trace dataset with its normalized weight."""

    dataset: TraceDataset
    weight: float  # Normalized weight (sums to 1 across all datasets)
    name: str

    # Traces grouped by problem_id (computed lazily)
    _problems_by_id: dict[str, ProblemTraces] | None = field(
        default=None, repr=False
    )

    @property
    def problems(self) -> dict[str, ProblemTraces]:
        """Get traces grouped by problem_id."""
        if self._problems_by_id is None:
            self._problems_by_id = {}
            for trace in self.dataset.traces:
                if trace.problem_id not in self._problems_by_id:
                    self._problems_by_id[trace.problem_id] = ProblemTraces(
                        problem_id=trace.problem_id,
                        task_name=trace.task_name,
                    )
                self._problems_by_id[trace.problem_id].traces.append(trace)
        return self._problems_by_id


@dataclass
class DatasetScore:
    """Score and diagnostics for a single dataset."""

    name: str
    score: float
    weight: float
    num_problems: int
    num_traces: int
    avg_ess: float  # Average ESS across problems


class ImportanceWeightedEvaluator(BaseEvaluator[LayerPlacement]):
    """Evaluator using importance-weighted accuracy estimation.

    Loads a supernet student model and pre-computed teacher traces from
    one or more datasets. For each placement configuration, estimates
    accuracy by:
    1. Setting mixer assignments on the model
    2. For each problem: compute student log-probs, IS estimate, ESS
    3. Stream ISProblemRecord to disk for each problem
    4. Aggregate per-problem estimates into final score

    This is much faster than generation (~10-100x) because:
    - No autoregressive decoding (just forward passes)
    - Same traces reused for all configurations
    - Batched computation of log-probabilities
    """

    def __init__(self, config: "ImportanceWeightedEvaluatorConfig") -> None:
        super().__init__(config)
        self.config: ImportanceWeightedEvaluatorConfig

        # Load all trace datasets with normalized weights
        self.datasets = self._load_datasets()
        self._log_dataset_summary()

        # Create backend (but don't enter context yet)
        self._backend: AsyncBackend | SyncBackend = config.backend.create_backend()

        # Model structure (populated in __aenter__ from backend)
        self._num_layers: int | None = None
        self._mixer_names: tuple[str, ...] | None = None

        # Cache for current placement to avoid redundant switching
        self._current_placement: tuple[str, ...] | None = None

        # Pre-compute log weight clip bounds for numerical stability
        self._log_weight_clip_min = math.log(self.config.weight_clip_min)
        self._log_weight_clip_max = math.log(self.config.weight_clip_max)

        # Set up trace writer for streaming results (enables resumption)
        hydra_cfg = HydraConfig.get()
        self._output_dir = Path(hydra_cfg.runtime.output_dir)
        self._trace_writer = TraceWriter(self._output_dir)

        logger.info(f"Saving IS results to: {self._output_dir}/validation.jsonl")

    async def __aenter__(self) -> "ImportanceWeightedEvaluator":
        """Enter context: initialize backend and optionally get model structure."""
        if isinstance(self._backend, AsyncBackend):
            await self._backend.__aenter__()
        else:
            self._backend.__enter__()

        # Try to get model structure from backend (optional - not all backends support it)
        try:
            structure = self._backend.model_structure
            self._num_layers = structure.num_layers
            self._mixer_names = structure.mixer_names
            logger.info(
                f"ImportanceWeightedEvaluator initialized: {self._num_layers} layers, "
                f"{len(self._mixer_names)} mixers, backend={self.config.backend.backend_type}"
            )
        except NotImplementedError:
            # Model structure not available (e.g., vLLM backend)
            # This is fine if we're not doing placement switching
            logger.info(
                f"ImportanceWeightedEvaluator initialized: model structure not available "
                f"(backend={self.config.backend.backend_type}). "
                f"Placement switching will not be supported."
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: cleanup backend resources."""
        if isinstance(self._backend, AsyncBackend):
            await self._backend.__aexit__(exc_type, exc_val, exc_tb)
        else:
            self._backend.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_layers(self) -> int:
        """Number of layers in the model (derived from backend)."""
        if self._num_layers is None:
            raise RuntimeError(
                "num_layers not available - evaluator must be entered first, "
                "or backend doesn't support model structure discovery"
            )
        return self._num_layers

    @property
    def num_mixers(self) -> int:
        """Number of mixer types per layer (derived from backend)."""
        if self._mixer_names is None:
            raise RuntimeError(
                "num_mixers not available - evaluator must be entered first, "
                "or backend doesn't support model structure discovery"
            )
        return len(self._mixer_names)

    @property
    def mixer_names(self) -> tuple[str, ...]:
        """Available mixer types (derived from backend)."""
        if self._mixer_names is None:
            raise RuntimeError(
                "mixer_names not available - evaluator must be entered first, "
                "or backend doesn't support model structure discovery"
            )
        return self._mixer_names

    def _load_datasets(self) -> list[LoadedDataset]:
        """Load all trace datasets and normalize weights."""
        import time

        sources = self.config.trace_sources
        logger.info(f"Loading {len(sources)} trace source(s)...")

        # Load datasets
        loaded = []
        total_weight = sum(s.weight for s in sources)

        for source in sources:
            path = Path(source.path)
            logger.info(f"Loading traces from {path}")
            start = time.perf_counter()
            dataset = TraceDataset.load_from_directory(path)
            elapsed = time.perf_counter() - start
            logger.info(
                f"  Loaded {dataset.num_traces} traces from {len(dataset.task_names)} task(s) "
                f"in {elapsed:.1f}s"
            )
            normalized_weight = source.weight / total_weight

            loaded.append(
                LoadedDataset(
                    dataset=dataset,
                    weight=normalized_weight,
                    name=source.display_name,
                )
            )

        return loaded

    def _log_dataset_summary(self) -> None:
        """Log summary of loaded datasets."""
        total_traces = sum(d.dataset.num_traces for d in self.datasets)
        logger.info(f"Loaded {len(self.datasets)} datasets with {total_traces} total traces:")
        for d in self.datasets:
            logger.info(
                f"  {d.name}: {d.dataset.num_traces} traces, "
                f"weight={d.weight:.1%}, "
                f"teacher_acc={d.dataset.accuracy:.1%}"
            )

    async def _set_placement(self, parameters: LayerPlacement) -> None:
        """Set the mixer placement on the model via backend."""
        # Convert assignment indices to mixer names
        mixer_names = tuple(self.mixer_names[idx] for idx in parameters.assignments)

        if mixer_names == self._current_placement:
            return  # Already set

        # Skip placement for single-mixer models (no dynamic switching needed)
        if len(set(self.mixer_names)) > 1:
            if isinstance(self._backend, AsyncBackend):
                await self._backend.set_placement(mixer_names)
            else:
                self._backend.set_placement(mixer_names)
            logger.debug(f"Set placement: {mixer_names[:4]}...")

        self._current_placement = mixer_names

    def _compute_problem_record(
        self,
        problem: ProblemTraces,
        student_log_probs: Sequence[float],
        dataset_name: str,
    ) -> ISProblemRecord:
        """Compute IS estimate for a single problem and create record.

        Args:
            problem: The problem's traces.
            student_log_probs: Student log-probs for each trace (same order as problem.traces).
            dataset_name: Name of the dataset this problem belongs to.

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
                w * (1.0 if c else 0.0)
                for w, c in zip(weights, correct_flags)
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
            dataset_name=dataset_name,
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

    async def _compute_dataset_score(
        self,
        loaded: LoadedDataset,
        completed_problems: set[int],
    ) -> DatasetScore:
        """Compute importance-weighted score for a dataset, streaming per-problem records.

        Args:
            loaded: The loaded dataset.
            completed_problems: Set of problem indices already completed (for resumption).

        Returns:
            DatasetScore with aggregated metrics.
        """
        problems = loaded.problems
        problem_ids = list(problems.keys())

        # Filter to remaining problems
        remaining_problems: list[ProblemTraces] = []
        for pid in problem_ids:
            # Extract problem index from problem_id (format: "task_name:idx")
            try:
                idx = int(pid.split(":")[-1])
                if idx not in completed_problems:
                    remaining_problems.append(problems[pid])
            except ValueError:
                # If we can't parse the index, include it (conservative)
                remaining_problems.append(problems[pid])

        if not remaining_problems:
            # All problems already completed - load from existing records
            # For now, return zeros (we'd need to reload records for accurate stats)
            logger.info(f"  All {len(problem_ids)} problems already completed")
            return DatasetScore(
                name=loaded.name,
                score=0.0,
                weight=loaded.weight,
                num_problems=len(problem_ids),
                num_traces=loaded.dataset.num_traces,
                avg_ess=0.0,
            )

        logger.info(
            f"  Processing {len(remaining_problems)} problems "
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
            (trace.prompt_tokens, trace.completion_tokens)
            for trace in all_traces
        ]

        if isinstance(self._backend, AsyncBackend):
            all_student_log_probs = await self._backend.compute_batch_log_probs(traces_data)
        else:
            all_student_log_probs = self._backend.compute_batch_log_probs(traces_data)

        # Process each problem, create records, and stream to disk
        records: list[ISProblemRecord] = []
        total_is_estimate = 0.0
        total_ess = 0.0

        with create_progress() as progress:
            task = progress.add_task(
                f"[cyan]{loaded.name}",
                total=len(remaining_problems),
            )

            for problem, (start, end) in zip(remaining_problems, problem_trace_ranges):
                problem_log_probs = all_student_log_probs[start:end]
                record = self._compute_problem_record(
                    problem, problem_log_probs, dataset_name=loaded.name
                )
                records.append(record)

                total_is_estimate += record.is_correct_estimate
                total_ess += record.ess

                progress.advance(task)

        # Stream all records to disk
        self._trace_writer.append(records)  # type: ignore[arg-type]

        # Compute aggregates
        num_problems = len(remaining_problems)
        avg_score = total_is_estimate / num_problems if num_problems > 0 else 0.0
        avg_ess = total_ess / num_problems if num_problems > 0 else 0.0

        return DatasetScore(
            name=loaded.name,
            score=avg_score,
            weight=loaded.weight,
            num_problems=num_problems,
            num_traces=len(all_traces),
            avg_ess=avg_ess,
        )

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate a placement using stratified importance-weighted accuracy.

        Computes per-problem IS estimates, streams records to disk, and
        combines with normalized weights.

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
            # Set placement via backend
            await self._set_placement(parameters)
        else:
            logger.info("No placement specified, using default configuration")

        # Compute per-dataset scores
        start = time.perf_counter()
        dataset_scores = []

        for d in self.datasets:
            logger.info(
                f"Computing score for dataset '{d.name}' "
                f"({d.dataset.num_problems} problems, {d.dataset.num_traces} traces)"
            )

            # Get completed problems for resumption
            # Note: We check per task_name within the dataset
            completed: set[int] = set()
            for task_name in d.dataset.task_names:
                completed.update(get_completed_problems(self._output_dir, task_name))

            score = await self._compute_dataset_score(d, completed)
            dataset_scores.append(score)
            logger.info(
                f"  {d.name}: score={score.score:.4f}, "
                f"avg_ESS={score.avg_ess:.1f}, "
                f"problems={score.num_problems}"
            )

        elapsed = time.perf_counter() - start

        # Compute summary from disk (single source of truth)
        summary = summarize_is(self._output_dir)

        # Build weights dict for weighted combination
        weights = {d.name: d.weight for d in self.datasets}
        final_score = summary.weighted_accuracy(weights)

        logger.info(f"\n{summary}")
        logger.info(f"Weighted accuracy: {final_score:.4f} (in {elapsed:.1f}s)")

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


class ImportanceWeightedEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for importance-weighted accuracy evaluator."""

    _impl_class: ClassVar[type[BaseEvaluator]] = ImportanceWeightedEvaluator

    evaluator_type: Literal["importance_weighted"] = "importance_weighted"

    # Backend (polymorphic - vLLM, Transformers, or REST)
    backend: BackendConfig = Field(
        description="Backend configuration (vllm, transformers, or rest)"
    )

    # Trace sources - list of datasets with weights
    trace_sources: list[TraceSource] = Field(
        description="List of trace datasets with importance weights"
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
    def validate_trace_sources(self) -> "ImportanceWeightedEvaluatorConfig":
        """Ensure at least one trace source is provided."""
        if not self.trace_sources:
            raise ValueError("At least one trace source is required")
        return self
