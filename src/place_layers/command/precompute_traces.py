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
from place_layers.evaluator.importance_weighted.vllm_client import (
    GenerationResult,
    VLLMBadRequestError,
    VLLMClient,
    VLLMClientConfig,
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
                teacher_model=self.config.vllm_model,
                samples_per_problem=self.config.samples_per_problem,
            )
            save_metadata(output_dir, metadata)
            logger.info("Starting new precomputation run")

        # Initialize simple trace writer (just appends to JSONL)
        writer = TraceWriter(output_dir)

        # Initialize vLLM client config
        vllm_config = VLLMClientConfig(
            base_url=self.config.vllm_base_url,
            model=self.config.vllm_model,
            api_key=self.config.vllm_api_key,
            timeout_seconds=self.config.vllm_timeout_seconds,
            max_concurrent_requests=self.config.max_concurrent_requests,
        )
        logger.info(f"vLLM: {self.config.vllm_base_url} model={self.config.vllm_model}")

        # Process each task lazily (load dataset only when processing begins)
        task_names = [_get_task_name(ts) for ts in self.config.tasks]
        logger.info(f"Tasks to process: {task_names}")

        # Track totals for final summary
        total_problems = 0
        total_completed = 0

        async with VLLMClient(vllm_config) as client:
            for task_spec in self.config.tasks:
                problems, completed = await self._process_task_spec(
                    task_spec=task_spec,
                    client=client,
                    writer=writer,
                    output_dir=output_dir,
                )
                total_problems += problems
                total_completed += completed

        logger.info(
            f"All tasks complete! {total_completed}/{total_problems} problems processed"
        )

    async def _process_task_spec(
        self,
        task_spec: str | TaskSpec,
        client: VLLMClient,
        writer: TraceWriter,
        output_dir: Path,
    ) -> tuple[int, int]:
        """Process a single task spec with lazy loading.

        Returns:
            Tuple of (total_problems, num_completed_this_run).
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
            return total_problems, 0

        num_already_done = len(completed_indices)
        logger.info(
            f"Task '{task_name}': {len(remaining_indices)} problems remaining "
            f"({total_problems} total, {num_already_done} done)"
        )

        # Process remaining problems
        num_completed = await self._process_problems(
            task=task,
            task_name=task_name,
            docs=docs,
            remaining_indices=remaining_indices,
            total_problems=total_problems,
            client=client,
            writer=writer,
        )

        logger.info(
            f"Task '{task_name}': complete, "
            f"{num_already_done + num_completed}/{total_problems} problems"
        )
        return total_problems, num_completed

    async def _process_problems(
        self,
        task: "Task",
        task_name: str,
        docs: Docs,
        remaining_indices: set[int],
        total_problems: int,
        client: VLLMClient,
        writer: TraceWriter,
    ) -> int:
        """Process remaining problems with worker pool pattern.

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

                # Generate completions
                prompt = task.fewshot_context(doc, num_fewshot=task.config.num_fewshot)
                try:
                    result = await client.chat_generate(
                        prompt=prompt,
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
            for _ in range(min(self.config.max_concurrent_requests, total_remaining))
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

    # vLLM server configuration
    vllm_base_url: str = Field(
        description="vLLM server base URL (e.g., 'http://localhost:8000')"
    )
    vllm_model: str = Field(description="Model name as registered in vLLM server")
    vllm_api_key: str = Field(description="API key for vLLM (use 'EMPTY' if not required)")
    vllm_timeout_seconds: int = Field(description="Request timeout in seconds")

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

    # Concurrency
    max_concurrent_requests: int = Field(description="Max concurrent vLLM requests", gt=0)

    @model_validator(mode="after")
    def validate_tasks(self) -> "PrecomputeTracesCommandConfig":
        """Ensure at least one task is specified."""
        if not self.tasks:
            raise ValueError("At least one task must be specified")
        return self
