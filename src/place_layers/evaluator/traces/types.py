"""Data structures for pre-computed reasoning traces.

These structures store the pre-computed traces from the teacher model,
which are used for importance-weighted accuracy estimation during search.

The key insight: we sample reasoning traces from the teacher once, record
their correctness and teacher log-probabilities, then reuse these traces
to estimate accuracy for any subnet via importance weighting.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from place_layers.backend.base import ChatMessage


class BaseGeneration(BaseModel):
    """Base class for generation records (shared fields).

    Contains the tokenized prompt and completion, correctness label,
    and metadata. Subclasses add specific fields (e.g., teacher_log_prob).
    """

    model_config = {"frozen": True}

    # Tokenized sequences
    prompt_tokens: tuple[int, ...] = Field(
        description="Tokenized prompt"
    )
    completion_tokens: tuple[int, ...] = Field(
        description="Tokenized completion"
    )

    # Evaluation result
    correct: bool | None = Field(
        description="Whether the answer extracted from completion is correct. "
        "None if correctness could not be determined."
    )

    # Metadata
    problem_id: str = Field(
        description="Unique identifier for the source problem"
    )
    task_name: str = Field(
        description="lm-eval task name (e.g., 'gsm8k')"
    )
    sample_idx: int = Field(
        description="Sample index (when multiple samples per problem)"
    )
    completion_text: str = Field(
        description="Decoded completion text (for debugging)",
        default="",
    )

    @property
    def total_tokens(self) -> int:
        """Total number of tokens (prompt + completion)."""
        return len(self.prompt_tokens) + len(self.completion_tokens)

    @property
    def completion_length(self) -> int:
        """Number of completion tokens."""
        return len(self.completion_tokens)


class Trace(BaseGeneration):
    """A reasoning trace with pre-computed teacher log-probability.

    Used for importance-weighted accuracy estimation during search.
    """

    record_type: Literal["trace"] = "trace"

    teacher_log_prob: float = Field(
        description="Sum of log p(token | prefix) over completion tokens under teacher"
    )


class FilterResult(BaseModel):
    """Result of applying a filter to a completion."""

    model_config = {"frozen": True}

    correct: bool | None = Field(description="Whether the filtered answer is correct")
    filtered_response: str = Field(description="The extracted answer after filtering")


class Generation(BaseGeneration):
    """A generation from evaluation (without teacher log-probs).

    Used by the generative evaluator to save completions for debugging.
    """

    record_type: Literal["generation"] = "generation"

    # Legacy field for backward compatibility (deprecated)
    filtered_response: str = Field(
        default="",
        description="(Deprecated) Use filter_results instead",
    )

    # Per-filter results: {filter_name: FilterResult}
    filter_results: dict[str, FilterResult] = Field(
        default_factory=dict,
        description="Per-filter correctness and extracted answers",
    )


class SkippedProblem(BaseModel):
    """A problem that was skipped during trace generation."""

    model_config = {"frozen": True}

    record_type: Literal["skipped"] = "skipped"

    problem_id: str = Field(description="Unique identifier for the source problem")
    task_name: str = Field(description="lm-eval task name")
    reason: str = Field(description="Why the problem was skipped")
    truncated_generations: list[str] | None = Field(
        default=None,
        description="Truncated generation texts (for debugging)",
    )


class LikelihoodRecord(BaseModel):
    """Multiple-choice evaluation record with full token-level observability.

    Used by the LikelihoodEvaluator to stream evaluation results to disk.
    Records everything needed to debug and analyze likelihood scoring:
    - Full chat messages (few-shot + question + each choice as assistant message)
    - Per-choice token sequences and logprobs
    - Answer range detection results
    - Predictions and correctness
    """

    model_config = {"frozen": True}

    record_type: Literal["likelihood"] = "likelihood"

    # === Problem Identification ===
    problem_id: str = Field(description="Unique identifier (e.g., 'mmlu_astronomy:42')")
    task_name: str = Field(description="lm-eval task name")

    # === Input: Full Chat Messages ===
    context_messages: tuple[ChatMessage, ...] = Field(
        description="Chat messages forming the shared context (few-shot + question)"
    )
    raw_choices: tuple[str, ...] = Field(
        description="Raw choice strings (e.g., ('A', 'B', 'C', 'D'))"
    )
    choices: tuple[ChatMessage, ...] = Field(
        description="Full assistant message for each choice"
    )

    # === Token-Level Data Per Choice (parallel arrays indexed by choice) ===
    choice_token_ids: tuple[tuple[int, ...], ...] = Field(
        description="Full tokenized sequence for each choice (after chat template)"
    )
    choice_logprobs: tuple[tuple[float | None, ...], ...] = Field(
        description="Per-token logprobs for each choice (None for first token)"
    )
    choice_answer_ranges: tuple[tuple[int, int], ...] = Field(
        description="(start, end) range identifying answer tokens for each choice"
    )

    # === Scores (derived but stored for convenience) ===
    choice_log_probs: tuple[float, ...] = Field(
        description="Sum of answer logprobs for each choice (used for prediction)"
    )
    choice_lengths: tuple[int, ...] = Field(
        description="Character length of each choice (for length normalization)"
    )

    # === Ground Truth and Predictions ===
    gold_idx: int = Field(description="Index of correct answer")
    pred_idx: int = Field(description="Index of predicted answer (argmax log-prob)")
    pred_idx_norm: int = Field(description="Index predicted with length normalization")
    correct: bool = Field(description="Whether pred_idx == gold_idx")
    correct_norm: bool = Field(description="Whether pred_idx_norm == gold_idx")


class ISProblemRecord(BaseModel):
    """Importance-weighted evaluation record for a single problem.

    Used by the ImportanceWeightedEvaluator to stream per-problem results.
    For each problem, we have N teacher samples (traces). We compute the
    student log-prob for each, then use importance sampling to estimate
    P(correct) for this problem.

    Records everything needed to debug and analyze IS scoring:
    - Per-sample student/teacher log-probs and importance weights
    - Problem-level IS estimate and effective sample size
    - Weight diagnostics (range, sum)
    """

    model_config = {"frozen": True}

    record_type: Literal["is_problem"] = "is_problem"

    # === Problem Identification ===
    problem_id: str = Field(description="Unique identifier (e.g., 'gsm8k:42')")
    task_name: str = Field(description="lm-eval task name")

    # === Per-Sample Data (parallel arrays, one element per trace/sample) ===
    sample_indices: tuple[int, ...] = Field(
        description="Original sample_idx values from traces"
    )
    student_log_probs: tuple[float, ...] = Field(
        description="Student model log-prob for each sample"
    )
    teacher_log_probs: tuple[float, ...] = Field(
        description="Teacher model log-prob for each sample (from input traces)"
    )
    correct_flags: tuple[bool | None, ...] = Field(
        description="Correctness flag for each sample (from input traces)"
    )
    log_importance_weights: tuple[float, ...] = Field(
        description="Clipped log importance weights: log(student) - log(teacher)"
    )

    # === Problem-Level IS Estimate ===
    is_correct_estimate: float = Field(
        description="IS estimate of P(correct): sum(w*correct) / sum(w)"
    )

    # === Problem-Level Diagnostics ===
    ess: float = Field(
        description="Effective sample size: (sum(w))^2 / sum(w^2)"
    )
    weight_sum: float = Field(
        description="Sum of importance weights for this problem"
    )
    weight_range: tuple[float, float] = Field(
        description="(min, max) importance weights across samples"
    )
    num_samples: int = Field(
        description="Number of samples (traces) for this problem"
    )


TraceRecord = Annotated[
    Trace | Generation | SkippedProblem | LikelihoodRecord | ISProblemRecord,
    Field(discriminator="record_type"),
]

TraceRecordAdapter: TypeAdapter[TraceRecord] = TypeAdapter(TraceRecord)


class TraceDataset(BaseModel):
    """Collection of pre-computed traces from teacher model.

    This is the serializable artifact produced by the pre-computation step
    and loaded by the importance-weighted evaluator.
    """

    model_config = {"frozen": True}

    # The traces themselves
    traces: tuple[Trace, ...] = Field(
        description="Pre-computed reasoning traces"
    )

    # Provenance information
    teacher_model_id: str = Field(
        description="Model ID used as teacher (vLLM model name)"
    )
    generation_config: dict = Field(
        description="Generation parameters used when sampling traces",
        default_factory=dict,
    )

    # Task information
    task_names: tuple[str, ...] = Field(
        description="lm-eval task names included in this dataset"
    )
    samples_per_problem: int = Field(
        description="Number of traces sampled per problem"
    )

    @property
    def num_traces(self) -> int:
        """Total number of traces."""
        return len(self.traces)

    @property
    def num_correct(self) -> int:
        """Number of traces with correct=True."""
        return sum(1 for t in self.traces if t.correct is True)

    @property
    def num_incorrect(self) -> int:
        """Number of traces with correct=False."""
        return sum(1 for t in self.traces if t.correct is False)

    @property
    def num_unknown(self) -> int:
        """Number of traces with correct=None (unknown correctness)."""
        return sum(1 for t in self.traces if t.correct is None)

    @property
    def num_evaluated(self) -> int:
        """Number of traces with known correctness (True or False)."""
        return self.num_correct + self.num_incorrect

    @property
    def accuracy(self) -> float:
        """Teacher accuracy (fraction of correct among evaluated traces)."""
        return self.num_correct / self.num_evaluated if self.num_evaluated > 0 else 0.0

    @property
    def avg_completion_length(self) -> float:
        """Average completion length in tokens."""
        if not self.traces:
            return 0.0
        return sum(t.completion_length for t in self.traces) / len(self.traces)

    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "TraceDataset":
        """Load dataset from JSON file."""
        return cls.model_validate_json(path.read_text())

    @classmethod
    def load_from_directory(cls, output_dir: Path) -> "TraceDataset":
        """Load dataset from output directory with validation.jsonl.

        This loads traces from validation.jsonl created by the streaming
        trace writer. The metadata.json file is used for provenance.

        The output directory structure:
            output_dir/
                metadata.json       # Provenance info
                validation.jsonl    # All valid traces (HF-loadable)
                skipped.jsonl       # Skipped problems (not loaded here)

        Args:
            output_dir: Directory containing metadata.json and validation.jsonl.

        Returns:
            TraceDataset with all traces from validation.jsonl.
        """
        import json
        import logging

        logger = logging.getLogger(__name__)
        metadata_path = output_dir / "metadata.json"
        validation_path = output_dir / "validation.jsonl"

        # Load metadata for provenance (teacher_model, samples_per_problem)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            teacher_model = metadata.get("teacher_model", "unknown")
            samples_per_problem = metadata.get("samples_per_problem", 0)
        else:
            teacher_model = "unknown"
            samples_per_problem = 0

        # Load all traces from validation.jsonl
        all_traces: list[Trace] = []
        task_names: set[str] = set()
        malformed = 0

        if validation_path.exists():
            with validation_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        trace = Trace.model_validate_json(line)
                        all_traces.append(trace)
                        task_names.add(trace.task_name)
                    except Exception:
                        malformed += 1

        if malformed > 0:
            logger.warning(f"{malformed} malformed lines ignored in validation.jsonl")

        return cls(
            traces=tuple(all_traces),
            teacher_model_id=teacher_model,
            generation_config={},
            task_names=tuple(sorted(task_names)),
            samples_per_problem=samples_per_problem,
        )

    @property
    def num_problems(self) -> int:
        """Number of unique problems."""
        return len({t.problem_id for t in self.traces})

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        lines = [
            f"TraceDataset(",
            f"  teacher={self.teacher_model_id}",
            f"  tasks={list(self.task_names)}",
            f"  traces={self.num_traces} "
            f"({self.num_problems} problems × {self.samples_per_problem} samples)",
            f"  teacher_accuracy={self.accuracy:.1%} "
            f"({self.num_correct}/{self.num_evaluated} correct)",
            f"  avg_completion_length={self.avg_completion_length:.0f} tokens",
        ]
        if self.num_unknown > 0:
            lines.append(f"  WARNING: {self.num_unknown} traces with unknown correctness")
        lines.append(")")
        return "\n".join(lines)
