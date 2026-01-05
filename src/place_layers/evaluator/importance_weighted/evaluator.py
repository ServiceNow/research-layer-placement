"""Importance-weighted accuracy estimator for supernet evaluation.

This evaluator uses importance sampling to estimate subnet accuracy without
generation. The key insight: sample reasoning traces from teacher once,
then reweight using student log-probabilities.

Theory:
    We want E_q[correct(x)] for subnet q, but sampling from q is expensive.
    Instead, sample from teacher p and reweight:
        E_q[correct(x)] = E_p[correct(x) × q(x)/p(x)]

    Using self-normalized importance sampling:
        score_q ≈ Σᵢ [correct(xᵢ) × wᵢ] / Σᵢ wᵢ
    where wᵢ = exp(log q(xᵢ) - log p(xᵢ))

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
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from place_layers.core.types import LayerPlacement
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.base_evaluator_config import BaseEvaluatorConfig
from place_layers.evaluator.importance_weighted.apriel_validation import (
    Apriel2Structure,
    validate_apriel2_model,
)
from place_layers.evaluator.importance_weighted.traces import Trace, TraceDataset
from place_layers.evaluator.importance_weighted.transformers_client import (
    TransformersClient,
    TransformersClientConfig,
)

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
class LoadedDataset:
    """A loaded trace dataset with its normalized weight."""

    dataset: TraceDataset
    weight: float  # Normalized weight (sums to 1 across all datasets)
    name: str


@dataclass
class DatasetScore:
    """Score and diagnostics for a single dataset."""

    name: str
    score: float
    weight: float
    ess: float  # Effective sample size
    num_traces: int
    weight_range: tuple[float, float]


class ImportanceWeightedEvaluator(BaseEvaluator[LayerPlacement]):
    """Evaluator using importance-weighted accuracy estimation.

    Loads a supernet student model and pre-computed teacher traces from
    one or more datasets. For each placement configuration, estimates
    accuracy by:
    1. Setting mixer assignments on the model
    2. Computing student log-probabilities for each trace
    3. Computing importance weights w = exp(log q - log p)
    4. For each dataset: compute self-normalized weighted accuracy
    5. Return weighted combination across datasets

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

        # Load model via TransformersClient
        client_config = TransformersClientConfig(
            model_path=self.config.model_path,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._client = TransformersClient(client_config)
        self._client.__enter__()  # Load model (cleanup in __del__ if needed)
        self.model = self._client.model

        # Validate model and extract structure
        self._structure = validate_apriel2_model(self.model)
        self._mixer_names = list(self._structure.mixer_names)
        logger.info(f"Model has {self._structure.num_layers} layers")
        logger.info(f"Available mixers: {self._mixer_names}")

        # Cache for current assignment to avoid redundant switching
        self._current_assignment: tuple[int, ...] | None = None

        # Pre-compute log weight clip bounds for numerical stability
        self._log_weight_clip_min = math.log(self.config.weight_clip_min)
        self._log_weight_clip_max = math.log(self.config.weight_clip_max)

    @property
    def num_layers(self) -> int:
        """Number of layers in the model (derived from loaded model)."""
        return self._structure.num_layers

    @property
    def num_mixers(self) -> int:
        """Number of mixer types per layer (derived from loaded model)."""
        return len(self._mixer_names)

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

    def _set_mixer_assignment(self, placement: LayerPlacement) -> None:
        """Set the mixer assignment on the model."""
        if self._current_assignment == placement.assignments:
            return  # Already set

        for layer_idx, mixer_idx in enumerate(placement.assignments):
            mixer_name = self._mixer_names[mixer_idx]
            self.model.model.decoder.blocks[layer_idx].mixer.main_mixer_name = mixer_name

        self._current_assignment = placement.assignments

    def _compute_dataset_score_batched(
        self, loaded: LoadedDataset, student_log_probs: list[float]
    ) -> DatasetScore:
        """Compute importance-weighted score using pre-computed log probs.

        Uses numerically stable log-space computation for weights.

        Args:
            loaded: The loaded dataset with traces.
            student_log_probs: Pre-computed student log-probabilities for each trace.

        Returns:
            DatasetScore with score and diagnostics.
        """
        traces = loaded.dataset.traces
        log_weights = []
        correct_flags = []

        for trace, student_log_prob in zip(traces, student_log_probs):
            # Log importance weight: log w = log q - log p
            log_weight = student_log_prob - trace.teacher_log_prob

            # Clip in log-space for numerical stability
            log_weight = max(
                self._log_weight_clip_min,
                min(self._log_weight_clip_max, log_weight),
            )
            log_weights.append(log_weight)
            correct_flags.append(trace.correct)

        # Compute score using log-sum-exp for numerical stability
        # score = sum(w_i * correct_i) / sum(w_i)
        #       = sum(exp(log_w_i) * correct_i) / sum(exp(log_w_i))
        #
        # Let max_log_w = max(log_weights)
        # score = sum(exp(log_w_i - max_log_w) * correct_i) / sum(exp(log_w_i - max_log_w))
        #
        # This avoids overflow when log_weights are large

        if not log_weights:
            return DatasetScore(
                name=loaded.name,
                score=0.0,
                weight=loaded.weight,
                ess=0.0,
                num_traces=0,
                weight_range=(0.0, 0.0),
            )

        max_log_weight = max(log_weights)

        # Compute normalized weights (for stability)
        weights = [math.exp(lw - max_log_weight) for lw in log_weights]

        # Compute weighted sum of correct flags
        total_weighted_correct = sum(
            w * (1.0 if c else 0.0) for w, c in zip(weights, correct_flags)
        )
        total_weight = sum(weights)

        # Self-normalized estimate
        if total_weight == 0:
            score = 0.0
        else:
            score = total_weighted_correct / total_weight

        # Compute ESS using actual weights (not normalized)
        actual_weights = [math.exp(lw) for lw in log_weights]
        ess = (sum(actual_weights) ** 2) / sum(w**2 for w in actual_weights)
        weight_range = (min(actual_weights), max(actual_weights))

        return DatasetScore(
            name=loaded.name,
            score=score,
            weight=loaded.weight,
            ess=ess,
            num_traces=len(log_weights),
            weight_range=weight_range,
        )

    def _compute_dataset_score(self, loaded: LoadedDataset) -> DatasetScore:
        """Compute importance-weighted score for a single dataset.

        Uses batched log-prob computation for efficiency.
        """
        # Prepare traces for batch computation
        traces_data = [
            (trace.prompt_tokens, trace.completion_tokens)
            for trace in loaded.dataset.traces
        ]

        # Compute all log-probs in batch using the client
        student_log_probs = self._client.compute_batch_log_probs(
            traces=traces_data,
            batch_size=self.config.trace_batch_size,
            length_tolerance=self.config.trace_length_tolerance,
        )

        return self._compute_dataset_score_batched(loaded, student_log_probs)

    async def evaluate(self, parameters: LayerPlacement) -> float:
        """Evaluate a placement using stratified importance-weighted accuracy.

        Computes per-dataset IS estimates and combines with normalized weights.

        Args:
            parameters: Layer placement to evaluate.

        Returns:
            Estimated accuracy (higher is better, in [0, 1]).
        """
        import time

        logger.info(f"Evaluating placement: {parameters.assignments[:4]}... ({len(parameters.assignments)} layers)")

        # Set mixer assignment
        self._set_mixer_assignment(parameters)

        # Compute per-dataset scores
        start = time.perf_counter()
        dataset_scores = []
        for d in self.datasets:
            logger.info(f"Computing score for dataset '{d.name}' ({d.dataset.num_traces} traces)")
            score = self._compute_dataset_score(d)
            dataset_scores.append(score)
            logger.info(
                f"  {d.name}: score={score.score:.4f}, "
                f"ESS={score.ess:.1f}/{score.num_traces}, "
                f"weights=[{score.weight_range[0]:.2e}, {score.weight_range[1]:.2e}]"
            )

        # Weighted combination
        final_score = sum(ds.score * ds.weight for ds in dataset_scores)
        elapsed = time.perf_counter() - start

        logger.info(f"Evaluation complete: score={final_score:.4f} in {elapsed:.1f}s")

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

    # Model configuration
    model_path: str = Field(
        description="Path to supernet model checkpoint (HuggingFace format)"
    )

    # Trace sources - list of datasets with weights
    trace_sources: list[TraceSource] = Field(
        description="List of trace datasets with importance weights"
    )

    # Model loading options
    torch_dtype: str = Field(
        description="Torch dtype for model (e.g., 'bfloat16', 'float16')",
    )
    device_map: str = Field(
        description="Device map for model loading (e.g., 'auto', 'cuda:0')",
    )
    trust_remote_code: bool = Field(
        description="Trust remote code when loading model (required for custom architectures)",
        default=True,
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

    # Batching parameters (for performance tuning)
    trace_batch_size: int = Field(
        description="Max traces per forward pass (higher = faster but more memory)",
        gt=0,
    )
    trace_length_tolerance: int = Field(
        description="Max token length difference within a batch (for padding efficiency)",
        gt=0,
    )

    @model_validator(mode="after")
    def validate_trace_sources(self) -> "ImportanceWeightedEvaluatorConfig":
        """Ensure at least one trace source is provided."""
        if not self.trace_sources:
            raise ValueError("At least one trace source is required")
        return self
