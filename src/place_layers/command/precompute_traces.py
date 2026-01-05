"""Pre-compute teacher traces for importance-weighted evaluation.

This command generates reasoning traces from a teacher model served by vLLM,
records their correctness (using lm-eval task definitions) and log-probabilities,
and saves them incrementally for use during importance-weighted accuracy estimation.

Usage
-----
1. Start vLLM server with the teacher model:

    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2-0.5B \\
        --port 8000 \\
        --max-model-len 4096 \\
        --gpu-memory-utilization 0.8

   WARNING: In vLLM v0.11.0+, you MUST use --model flag explicitly.
   Positional arguments are silently ignored and may load a wrong model!

2. Run the precompute_traces command:

    VLLM_MODEL="Qwen/Qwen2-0.5B" python -m place_layers \\
        command=precompute_traces \\
        tasks='[gsm8k]' \\
        max_problems_per_task=100 \\
        samples_per_problem=8 \\
        max_new_tokens=1024

   Key parameters:
   - tasks: List of lm-eval tasks (e.g., '[gsm8k,triviaqa]')
   - max_problems_per_task: Limit problems per task (null for all)
   - samples_per_problem: Number of traces to sample per problem
   - max_new_tokens: Maximum tokens to generate (must be enough for full solutions)
   - temperature: Sampling temperature (default 0.7)
   - num_fewshot: Number of few-shot examples (default 5)

Architecture
------------
- vLLM: Out-of-process inference server (async HTTP API)
- lm-eval: Task definitions, prompt formatting, correctness checking
- Streaming: Process problems as they complete, write incrementally
- Resumption: State reconstructed from traces on disk (source of truth)
- Lazy loading: Datasets loaded only when processing begins for each task

Output structure
----------------
    output_dir/
        metadata.json       # Config hash, teacher model, samples_per_problem
        validation.jsonl    # All valid traces (HF-loadable via load_dataset)
        skipped.jsonl       # Skipped problems (for debugging)

The validation.jsonl file can be loaded directly with HuggingFace datasets:

    from datasets import load_dataset
    ds = load_dataset(output_dir, split='validation')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from hydra.core.hydra_config import HydraConfig
from pydantic import Field, model_validator

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.evaluator.importance_weighted.generation_backend import (
    AsyncBackend,
    BackendConfig,
    SyncBackend,
)
from place_layers.evaluator.importance_weighted.memory_logging import (
    log_memory,
    reset_peak_memory_stats,
)
from place_layers.evaluator.importance_weighted.trace_writer import (
    PrecomputeMetadata,
    TraceWriter,
    compute_config_hash,
    load_metadata,
    save_metadata,
)
from place_layers.evaluator.importance_weighted.traces import Trace
from place_layers.evaluator.importance_weighted.vllm_client import GenerationResult
from place_layers.evaluator.lm_eval_utils import (
    TaskSpecInput,
    check_correctness,
    get_task_name,
)
from place_layers.evaluator.task_runner import (
    prepare_task,
    run_generation_async,
    run_generation_sync,
)

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


# =============================================================================
# Record Factory
# =============================================================================


def _make_trace_records(
    task: "Task", doc: dict, idx: int, result: GenerationResult
) -> list[Trace]:
    """Create Trace records from a generation result.

    Skips truncated samples (they don't have valid log-probs).
    Returns empty list if all samples are truncated.
    """
    task_name = task.config.task
    traces = []
    prompt_tokens = tuple(result.prompt_token_ids)
    problem_id = f"{task_name}:{idx}"

    for sample_idx, sample in enumerate(result.samples):
        if sample.truncated:
            logger.debug(f"Skipping truncated sample {problem_id}:{sample_idx}")
            continue

        correct = check_correctness(task, doc, sample.text)
        trace = Trace(
            prompt_tokens=prompt_tokens,
            completion_tokens=tuple(sample.token_ids),
            correct=correct,
            teacher_log_prob=sample.total_logprob,
            problem_id=problem_id,
            task_name=task_name,
            sample_idx=sample_idx,
            completion_text=sample.text,
        )
        traces.append(trace)

    return traces


# =============================================================================
# Command Implementation
# =============================================================================


class PrecomputeTracesCommand(BaseCommand):
    """Pre-compute teacher traces using streaming architecture."""

    config: "PrecomputeTracesCommandConfig"

    async def run(self) -> None:
        """Execute trace pre-computation with streaming and resumption."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info(f"Output directory: {output_dir}")

        # Log initial memory state and reset peak tracking
        log_memory("INIT")
        reset_peak_memory_stats()

        # Compute config hash for resumption validation
        config_dict = self.config.model_dump(mode="json")
        config_hash = compute_config_hash(config_dict)

        # Check for existing metadata (resumption case)
        existing_metadata = load_metadata(output_dir)
        if existing_metadata is not None:
            if existing_metadata.config_hash != config_hash:
                raise ValueError(
                    f"Config hash mismatch: existing={existing_metadata.config_hash}, "
                    f"current={config_hash}. Cannot resume with different config."
                )
            logger.info("Resuming from existing traces")
        else:
            # Save metadata for new run
            metadata = PrecomputeMetadata(
                config_hash=config_hash,
                teacher_model=self.config.model_name,
                samples_per_problem=self.config.samples_per_problem,
            )
            save_metadata(output_dir, metadata)
            logger.info("Starting new precomputation run")

        # Initialize simple trace writer (just appends to JSONL)
        writer = TraceWriter(output_dir)

        # Create generation backend from config
        backend = self.config.backend.create_backend()
        logger.info(f"Backend: {self.config.backend.backend_type}")

        if self.config.placement is not None:
            logger.info(
                f"Placement: {self.config.placement[:4]}... "
                f"({len(self.config.placement)} layers)"
            )

        # Process each task lazily (load dataset only when processing begins)
        task_names = [get_task_name(ts) for ts in self.config.tasks]
        logger.info(f"Tasks to process: {task_names}")

        # Dispatch based on backend category
        if isinstance(backend, AsyncBackend):
            await self._run_tasks_async(backend, writer, output_dir)
        elif isinstance(backend, SyncBackend):
            self._run_tasks_sync(backend, writer, output_dir)
        else:
            raise TypeError(f"Unknown backend type: {type(backend)}")

    async def _run_tasks_async(
        self,
        backend: AsyncBackend,
        writer: TraceWriter,
        output_dir: Path,
    ) -> None:
        """Run all tasks with async backend."""
        total_processed = 0

        async with backend:
            # Set placement if specified (after model is loaded)
            if self.config.placement is not None:
                await backend.set_placement(self.config.placement)

            for task_spec in self.config.tasks:
                prepared = prepare_task(
                    task_spec=task_spec,
                    output_dir=output_dir,
                    num_fewshot=self.config.num_fewshot,
                    max_problems=self.config.max_problems_per_task,
                    split=self.config.split,
                    seed=self.config.seed,
                )

                if prepared is None:
                    continue

                task, task_name, docs, remaining = prepared

                num_processed = await run_generation_async(
                    task=task,
                    task_name=task_name,
                    docs=docs,
                    remaining_indices=remaining,
                    backend=backend,
                    writer=writer,
                    record_factory=_make_trace_records,
                    samples_per_problem=self.config.samples_per_problem,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )

                total_processed += num_processed

        logger.info(f"All tasks complete! {total_processed} problems processed")

    def _run_tasks_sync(
        self,
        backend: SyncBackend,
        writer: TraceWriter,
        output_dir: Path,
    ) -> None:
        """Run all tasks with sync backend."""
        total_processed = 0

        with backend:
            # Set placement if specified (after model is loaded)
            if self.config.placement is not None:
                backend.set_placement(self.config.placement)

            for task_spec in self.config.tasks:
                prepared = prepare_task(
                    task_spec=task_spec,
                    output_dir=output_dir,
                    num_fewshot=self.config.num_fewshot,
                    max_problems=self.config.max_problems_per_task,
                    split=self.config.split,
                    seed=self.config.seed,
                )

                if prepared is None:
                    continue

                task, task_name, docs, remaining = prepared

                num_processed = run_generation_sync(
                    task=task,
                    task_name=task_name,
                    docs=docs,
                    remaining_indices=remaining,
                    backend=backend,
                    writer=writer,
                    record_factory=_make_trace_records,
                    samples_per_problem=self.config.samples_per_problem,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )

                total_processed += num_processed

        logger.info(f"All tasks complete! {total_processed} problems processed")


# =============================================================================
# Configuration
# =============================================================================


class PrecomputeTracesCommandConfig(BaseCommandConfig):
    """Configuration for trace pre-computation command."""

    _command_class: ClassVar[type[BaseCommand]] = PrecomputeTracesCommand

    command: Literal["precompute_traces"] = "precompute_traces"

    # Generation backend (vLLM or Transformers)
    backend: BackendConfig = Field(
        description="Generation backend configuration (vllm or transformers)"
    )

    # Task configuration
    tasks: list[TaskSpecInput] = Field(
        description="List of tasks (strings or TaskSpec with overrides)"
    )
    split: str = Field(description="Dataset split ('test' or 'validation')")
    num_fewshot: int = Field(
        description="Default few-shot examples (can override per-task)"
    )
    max_problems_per_task: int | None = Field(
        description="Max problems per task (None=all)"
    )

    # Sampling configuration
    samples_per_problem: int = Field(
        description="Traces to sample per problem", gt=0
    )
    seed: int = Field(description="Random seed for few-shot sampling")

    # Generation configuration
    max_new_tokens: int = Field(
        description="Max tokens to generate per sample", gt=0
    )
    temperature: float = Field(description="Sampling temperature", gt=0.0)
    top_p: float = Field(
        description="Top-p (nucleus) sampling", gt=0.0, le=1.0
    )

    # Mixer placement (optional, for supernet models)
    placement: list[str] | None = Field(
        default=None,
        description="Mixer names per layer (e.g., ['attention', 'gdn', ...]). "
        "If None, uses model's default placement.",
    )

    @model_validator(mode="after")
    def validate_tasks(self) -> "PrecomputeTracesCommandConfig":
        """Ensure at least one task is specified."""
        if not self.tasks:
            raise ValueError("At least one task must be specified")
        return self

    @property
    def model_name(self) -> str:
        """Get model name/path from backend config."""
        from place_layers.evaluator.importance_weighted.generation_backend import (
            TransformersBackendConfig,
            VLLMBackendConfig,
        )

        if isinstance(self.backend, VLLMBackendConfig):
            return self.backend.model
        elif isinstance(self.backend, TransformersBackendConfig):
            return self.backend.model_path
        return "unknown"
