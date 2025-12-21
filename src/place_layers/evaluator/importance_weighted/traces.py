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


class Trace(BaseModel):
    """A single reasoning trace with pre-computed teacher statistics.

    Contains the tokenized prompt and completion, correctness label,
    and teacher log-probability needed for importance weighting.
    """

    model_config = {"frozen": True}

    record_type: Literal["trace"] = "trace"

    # Tokenized sequences
    prompt_tokens: tuple[int, ...] = Field(
        description="Tokenized prompt (shared prefix for teacher and student)"
    )
    completion_tokens: tuple[int, ...] = Field(
        description="Tokenized completion (the reasoning trace + answer)"
    )

    # Pre-computed teacher statistics
    correct: bool | None = Field(
        description="Whether the answer extracted from completion is correct. "
        "None if correctness could not be determined from task metrics."
    )
    teacher_log_prob: float = Field(
        description="Sum of log p(token | prefix) over completion tokens under teacher"
    )

    # Metadata for analysis/debugging
    problem_id: str = Field(
        description="Unique identifier for the source problem"
    )
    task_name: str = Field(
        description="lm-eval task name (e.g., 'gsm8k', 'minerva_math_algebra')"
    )
    sample_idx: int = Field(
        description="Sample index (when multiple traces per problem)"
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


class SkippedProblem(BaseModel):
    """A problem that was skipped during trace generation."""

    model_config = {"frozen": True}

    record_type: Literal["skipped"] = "skipped"

    problem_id: str = Field(description="Unique identifier for the source problem")
    task_name: str = Field(description="lm-eval task name")
    reason: str = Field(description="Why the problem was skipped")


TraceRecord = Annotated[
    Trace | SkippedProblem,
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
