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

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Protocol, cast

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field, model_validator

from place_layers.command.base_command import BaseCommand
from place_layers.command.base_command_config import BaseCommandConfig
from place_layers.evaluator.importance_weighted.trace_writer import (
    PrecomputeMetadata,
    TraceWriter,
    compute_config_hash,
    get_completed_problems,
    load_metadata,
    save_metadata,
)
from place_layers.evaluator.importance_weighted.traces import SkippedProblem, Trace
from place_layers.evaluator.importance_weighted.generation_backend import (
    AsyncBackend,
    BackendConfig,
    ChatMessage,
    SyncBackend,
)
from place_layers.evaluator.importance_weighted.vllm_client import (
    GenerationResult,
    VLLMBadRequestError,
)
from place_layers.evaluator.importance_weighted.memory_logging import (
    log_memory,
    log_memory_delta,
    get_memory_stats,
    MemoryMonitor,
    force_garbage_collection,
    log_gpu_memory_summary,
    reset_peak_memory_stats,
)

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


class Docs(Protocol):
    """Protocol for lm-eval document collections.

    HuggingFace Dataset and list[dict] both satisfy this protocol.
    We only require __len__ and __getitem__ - the minimal interface
    needed for indexed access and our resumption logic.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict: ...


class TaskSpec(BaseModel):
    """Configuration for a single lm-eval task."""

    model_config = {"frozen": True, "extra": "forbid"}

    task: str = Field(description="lm-eval task name (e.g., 'gsm8k')")
    num_fewshot: int | None = Field(
        default=None, description="Override number of few-shot examples"
    )
    description: str | None = Field(
        default=None, description="Override system/task description"
    )
    fewshot_sampler: Literal["first_n", "random", "balanced"] | None = Field(
        default=None, description="Few-shot sampling strategy"
    )

    def to_lm_eval_config(self, default_num_fewshot: int) -> dict:
        """Convert to lm-eval task config dict."""
        config: dict = {"task": self.task}
        config["num_fewshot"] = (
            self.num_fewshot if self.num_fewshot is not None else default_num_fewshot
        )
        if self.description is not None:
            config["description"] = self.description
        if self.fewshot_sampler is not None:
            config["fewshot_config"] = {"sampler": self.fewshot_sampler}
        return config


TaskSpecInput = Annotated[
    str | TaskSpec,
    Field(description="Task name or TaskSpec with overrides"),
]


def _normalize_task_spec(task: str | TaskSpec, default_num_fewshot: int) -> dict:
    """Convert a single task spec to lm-eval config dict."""
    if isinstance(task, str):
        return {"task": task, "num_fewshot": default_num_fewshot}
    return task.to_lm_eval_config(default_num_fewshot)


def _get_task_name(task_spec: str | TaskSpec) -> str:
    """Extract task name from task spec."""
    if isinstance(task_spec, str):
        return task_spec
    return task_spec.task


def _get_docs(task: "Task", split: str) -> Docs:
    """Get documents from task for the specified split.

    Returns an indexable sequence (HuggingFace Dataset or list of dicts).
    lm-eval requires this for distributed evaluation, so all standard tasks
    support len() and __getitem__.
    """
    docs: Any
    if split == "test" and task.has_test_docs():
        docs = task.test_docs()
    elif split == "validation" and task.has_validation_docs():
        docs = task.validation_docs()
    elif task.has_test_docs():
        docs = task.test_docs()
    elif task.has_validation_docs():
        docs = task.validation_docs()
    else:
        raise RuntimeError(f"Task '{task.config.task}' has no test or validation docs")

    # Verify docs supports indexing (required by our resumption logic)
    # lm-eval's own build_all_requests() also requires this
    try:
        _ = len(docs)
        if len(docs) > 0:
            _ = docs[0]
    except (TypeError, KeyError) as e:
        raise TypeError(
            f"Task '{task.config.task}' docs must support len() and __getitem__ "
            f"(got {type(docs).__name__}). This is required by lm-eval and our "
            f"resumption logic."
        ) from e

    return cast(Docs, docs)


def _build_chat_messages(task: "Task", doc: dict, num_fewshot: int) -> list[ChatMessage]:
    """Build chat messages for a problem using proper multi-turn few-shot format.

    Uses lm-eval's get_chat_context with fewshot_as_multiturn=True to ensure
    few-shot examples appear as alternating user/assistant turns. This helps
    the model learn the expected answer format from the assistant turns.

    Args:
        task: The lm-eval task.
        doc: The problem document.
        num_fewshot: Number of few-shot examples.

    Returns:
        List of ChatMessage dicts with role/content keys.
    """
    # Get few-shot context as proper multi-turn messages
    # This returns list[dict] with alternating user/assistant messages
    messages: list[ChatMessage] = task.sampler.get_chat_context(
        doc, num_fewshot=num_fewshot, fewshot_as_multiturn=True
    )

    # Append the actual question as the final user message
    question = task.doc_to_text(doc)
    messages.append({"role": "user", "content": question})

    return messages


def _check_correctness(task: "Task", doc: dict, completion: str) -> bool | None:
    """Check if completion is correct using task's process_results.

    For generate_until tasks, lm-eval returns metrics like:
    - GSM8K: {"exact_match": 0 or 1}
    - MATH: {"exact_match": 0 or 1, "exact_match_original": 0 or 1}
    - AIME: {"exact_match": 0 or 1}

    We prefer "exact_match" when present (uses symbolic math verification
    for MATH tasks), falling back to the first binary metric.

    Returns:
        True if correct, False if incorrect, None if correctness cannot
        be determined from the task's metrics. The None case is expected
        to be rare for well-formed tasks, but allows faithful recording
        without introducing bias.
    """
    metrics: dict[str, Any] | None = task.process_results(doc, [completion])

    if metrics is None:
        logger.warning(
            f"process_results returned None. Trace will be saved with correct=None."
        )
        return None

    # Prefer exact_match - it's the standard metric and uses symbolic
    # verification for math tasks (more accurate than string matching)
    if "exact_match" in metrics:
        value = metrics["exact_match"]
        if isinstance(value, (int, float)) and value in (0, 1, 0.0, 1.0):
            return value == 1 or value == 1.0

    # Fallback: first binary metric we find
    for value in metrics.values():
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1, 0.0, 1.0):
            return value == 1 or value == 1.0

    logger.warning(
        f"Could not determine correctness from metrics {metrics}. "
        f"Trace will be saved with correct=None."
    )
    return None


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
            logger.info(f"Placement: {self.config.placement[:4]}... ({len(self.config.placement)} layers)")

        # Process each task lazily (load dataset only when processing begins)
        task_names = [_get_task_name(ts) for ts in self.config.tasks]
        logger.info(f"Tasks to process: {task_names}")

        # Dispatch based on backend category
        if isinstance(backend, AsyncBackend):
            await self._run_async_pool(backend, writer, output_dir)
        elif isinstance(backend, SyncBackend):
            self._run_sync_loop(backend, writer, output_dir)
        else:
            raise TypeError(f"Unknown backend type: {type(backend)}")

    async def _run_async_pool(
        self,
        backend: AsyncBackend,
        writer: TraceWriter,
        output_dir: Path,
    ) -> None:
        """Run with async backend using worker pool for concurrent requests."""
        total_problems = 0
        total_completed = 0

        async with backend:
            # Set placement if specified (after model is loaded)
            if self.config.placement is not None:
                await backend.set_placement(self.config.placement)
            for task_spec in self.config.tasks:
                prepared = self._prepare_task(task_spec, output_dir)
                if prepared is None:
                    continue

                task, task_name, docs, remaining_indices, total, num_already_done = prepared
                total_problems += total

                num_completed = await self._process_problems_async(
                    task=task,
                    task_name=task_name,
                    docs=docs,
                    remaining_indices=remaining_indices,
                    backend=backend,
                    writer=writer,
                )

                logger.info(
                    f"Task '{task_name}': complete, "
                    f"{num_already_done + num_completed}/{total} problems"
                )
                total_completed += num_completed

        logger.info(
            f"All tasks complete! {total_completed}/{total_problems} problems processed"
        )

    def _run_sync_loop(
        self,
        backend: SyncBackend,
        writer: TraceWriter,
        output_dir: Path,
    ) -> None:
        """Run with sync backend using sequential processing."""
        total_problems = 0
        total_completed = 0

        with backend:
            # Set placement if specified (after model is loaded)
            if self.config.placement is not None:
                backend.set_placement(self.config.placement)
            for task_spec in self.config.tasks:
                prepared = self._prepare_task(task_spec, output_dir)
                if prepared is None:
                    continue

                task, task_name, docs, remaining_indices, total, num_already_done = prepared
                total_problems += total

                num_completed = self._process_problems_sync(
                    task=task,
                    task_name=task_name,
                    docs=docs,
                    remaining_indices=remaining_indices,
                    backend=backend,
                    writer=writer,
                )

                logger.info(
                    f"Task '{task_name}': complete, "
                    f"{num_already_done + num_completed}/{total} problems"
                )
                total_completed += num_completed

        logger.info(
            f"All tasks complete! {total_completed}/{total_problems} problems processed"
        )

    def _prepare_task(
        self,
        task_spec: str | TaskSpec,
        output_dir: Path,
    ) -> tuple["Task", str, Docs, set[int], int, int] | None:
        """Load and prepare a task, returning None if already complete.

        Returns:
            Tuple of (task, task_name, docs, remaining_indices, total_problems, num_already_done)
            or None if the task is already complete.
        """
        from lm_eval.api.task import Task
        from lm_eval.tasks import TaskManager, get_task_dict

        task_name = _get_task_name(task_spec)
        logger.info(f"Task '{task_name}': loading...")

        # Load the task lazily
        task_config = _normalize_task_spec(task_spec, self.config.num_fewshot)
        task_manager = TaskManager()
        task_dict: dict[str, Task] = get_task_dict([task_config], task_manager=task_manager)
        task = task_dict[task_name]

        # Validate task type
        if task.config.output_type != "generate_until":
            raise ValueError(
                f"Task '{task_name}' has output_type='{task.config.output_type}', "
                f"expected 'generate_until'"
            )

        # Prepare task
        task.set_fewshot_seed(self.config.seed)
        task.download()

        # Get documents
        docs = _get_docs(task, self.config.split)

        # Apply max_problems_per_task limit
        total_problems = len(docs)
        if self.config.max_problems_per_task is not None:
            total_problems = min(total_problems, self.config.max_problems_per_task)

        # Reconstruct completed state from JSONL (source of truth)
        completed_indices = get_completed_problems(output_dir, task_name)
        all_indices = set(range(total_problems))
        remaining_indices = all_indices - completed_indices

        if not remaining_indices:
            logger.info(f"Task '{task_name}': already complete, skipping")
            return None

        num_already_done = len(completed_indices)
        logger.info(
            f"Task '{task_name}': {len(remaining_indices)} problems remaining "
            f"({total_problems} total, {num_already_done} done)"
        )

        return task, task_name, docs, remaining_indices, total_problems, num_already_done

    async def _process_problems_async(
        self,
        task: "Task",
        task_name: str,
        docs: Docs,
        remaining_indices: set[int],
        backend: AsyncBackend,
        writer: TraceWriter,
    ) -> int:
        """Process remaining problems with async worker pool pattern.

        Uses a queue-based worker pool to limit memory usage. Only
        max_concurrent_requests docs are held in memory at once,
        rather than all remaining docs.

        Returns:
            Number of problems completed in this run.
        """
        queue: asyncio.Queue[int] = asyncio.Queue()
        total_remaining = len(remaining_indices)
        done = False

        async def worker() -> None:
            """Worker that pulls problem indices from queue and processes them."""
            while True:
                try:
                    idx = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                # Fetch doc only when we're ready to process it
                doc = docs[idx]

                # Build chat messages with proper multi-turn few-shot format
                messages = _build_chat_messages(task, doc, task.config.num_fewshot)
                try:
                    result = await backend.agenerate(
                        messages=messages,
                        n=self.config.samples_per_problem,
                        max_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        request_id=f"{task_name}:{idx}",
                    )
                except VLLMBadRequestError as e:
                    logger.warning(f"Skipping {task_name}:{idx} - {e}")
                    skipped = SkippedProblem(
                        problem_id=f"{task_name}:{idx}",
                        task_name=task_name,
                        reason=str(e),
                    )
                    writer.append([skipped])
                    continue

                # Convert to traces
                traces = self._result_to_traces(
                    task=task,
                    task_name=task_name,
                    doc=doc,
                    problem_idx=idx,
                    result=result,
                )

                # Append to JSONL - either traces or a skip record if all truncated
                if traces:
                    writer.append(traces)
                else:
                    skipped = SkippedProblem(
                        problem_id=f"{task_name}:{idx}",
                        task_name=task_name,
                        reason=f"all {len(result.samples)} samples truncated",
                    )
                    writer.append([skipped])

        async def log_progress() -> None:
            """Periodically log queue size."""
            while not done:
                remaining = queue.qsize()
                processed = total_remaining - remaining
                logger.info(
                    f"Task '{task_name}': {processed}/{total_remaining} "
                    f"({remaining} in queue)"
                )
                await asyncio.sleep(10)

        # Populate queue with remaining problem indices
        for idx in remaining_indices:
            queue.put_nowait(idx)

        # Start worker pool and progress logger
        workers = [
            asyncio.create_task(worker())
            for _ in range(min(backend.max_concurrent_requests, total_remaining))
        ]
        progress_task = asyncio.create_task(log_progress())

        # Wait for all workers to complete, then stop progress logger
        await asyncio.gather(*workers)
        done = True
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

        return total_remaining

    def _process_problems_sync(
        self,
        task: "Task",
        task_name: str,
        docs: Docs,
        remaining_indices: set[int],
        backend: SyncBackend,
        writer: TraceWriter,
    ) -> int:
        """Process remaining problems synchronously with batching.

        Used for local transformers models. Batches multiple prompts together
        for efficient GPU utilization.

        Returns:
            Number of problems completed in this run.
        """
        import time

        total_remaining = len(remaining_indices)
        remaining_list = sorted(remaining_indices)
        batch_size = backend.batch_size
        last_log_time = time.time()
        last_memory_log_time = time.time()
        log_interval = 10  # seconds
        memory_log_interval = 30  # seconds

        logger.info(f"Processing {total_remaining} problems with batch_size={batch_size}")
        log_memory(f"TASK {task_name} START")

        # Process in batches
        for batch_start in range(0, total_remaining, batch_size):
            batch_end = min(batch_start + batch_size, total_remaining)
            batch_indices = remaining_list[batch_start:batch_end]

            # Log progress periodically
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                logger.info(
                    f"Task '{task_name}': {batch_start}/{total_remaining} "
                    f"({total_remaining - batch_start} remaining)"
                )
                last_log_time = current_time

            # Log memory periodically
            if current_time - last_memory_log_time >= memory_log_interval:
                log_memory(f"BATCH {batch_start}/{total_remaining}")
                last_memory_log_time = current_time

            # Collect docs and build chat messages for this batch
            batch_docs = [docs[idx] for idx in batch_indices]
            batch_messages = [
                _build_chat_messages(task, doc, task.config.num_fewshot)
                for doc in batch_docs
            ]

            # Generate completions for the entire batch
            mem_before = get_memory_stats()
            try:
                results = backend.batch_generate(
                    messages_batch=batch_messages,
                    n=self.config.samples_per_problem,
                    max_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
            except Exception as e:
                # Log memory state on exception
                logger.error(f"Batch {batch_start}-{batch_end} failed: {type(e).__name__}: {e}")
                log_memory(f"BATCH {batch_start} EXCEPTION", level=logging.ERROR)

                # Check if OOM and log detailed info
                if "CUDA out of memory" in str(e) or "OutOfMemory" in type(e).__name__:
                    logger.error("OUT OF MEMORY DETECTED!")
                    log_gpu_memory_summary()

                # Skip all problems in the batch
                logger.warning(f"Skipping {len(batch_indices)} problems due to error")
                for idx in batch_indices:
                    skipped = SkippedProblem(
                        problem_id=f"{task_name}:{idx}",
                        task_name=task_name,
                        reason=str(e),
                    )
                    writer.append([skipped])

                # Try to recover memory
                force_garbage_collection()
                continue

            # Process results for each problem in the batch
            for idx, doc, result in zip(batch_indices, batch_docs, results):
                traces = self._result_to_traces(
                    task=task,
                    task_name=task_name,
                    doc=doc,
                    problem_idx=idx,
                    result=result,
                )

                # Append to JSONL - either traces or a skip record if all truncated
                if traces:
                    writer.append(traces)
                else:
                    skipped = SkippedProblem(
                        problem_id=f"{task_name}:{idx}",
                        task_name=task_name,
                        reason=f"all {len(result.samples)} samples truncated",
                    )
                    writer.append([skipped])

        # Log final memory state for this task
        log_memory(f"TASK {task_name} END")

        return total_remaining

    def _result_to_traces(
        self,
        task: "Task",
        task_name: str,
        doc: dict,
        problem_idx: int,
        result: GenerationResult,
    ) -> list[Trace]:
        """Convert a generation result to traces, skipping truncated samples."""
        traces = []
        prompt_tokens = tuple(result.prompt_token_ids)
        problem_id = f"{task_name}:{problem_idx}"

        for sample_idx, sample in enumerate(result.samples):
            if sample.truncated:
                logger.debug(f"Skipping truncated sample {problem_id}:{sample_idx}")
                continue

            correct = _check_correctness(task, doc, sample.text)
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
    num_fewshot: int = Field(description="Default few-shot examples (can override per-task)")
    max_problems_per_task: int | None = Field(description="Max problems per task (None=all)")

    # Sampling configuration
    samples_per_problem: int = Field(description="Traces to sample per problem", gt=0)
    seed: int = Field(description="Random seed for few-shot sampling")

    # Generation configuration
    max_new_tokens: int = Field(description="Max tokens to generate per sample", gt=0)
    temperature: float = Field(description="Sampling temperature", gt=0.0)
    top_p: float = Field(description="Top-p (nucleus) sampling", gt=0.0, le=1.0)

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
            VLLMBackendConfig,
            TransformersBackendConfig,
        )
        if isinstance(self.backend, VLLMBackendConfig):
            return self.backend.model
        elif isinstance(self.backend, TransformersBackendConfig):
            return self.backend.model_path
        return "unknown"
