"""Base class for evaluators that iterate over lm-eval tasks.

This module provides the TaskBasedEvaluator which adds task iteration
infrastructure on top of BackendEvaluator:

- Task loading via lm-eval
- Task type validation (generate_until vs multiple_choice)
- Document retrieval and max_problems limiting
- Resumability via TraceWriter.get_completed_problems()
- Async doc prefetching for pipeline efficiency

Concrete evaluators (GenerativeEvaluator, LikelihoodEvaluator, etc.)
inherit from this and implement their specific evaluation logic.
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import Field, model_validator

from place_layers.backend.base import AsyncBackend, SyncBackend
from place_layers.backend.types import BackendError
from place_layers.evaluator.backend_evaluator import (
    BackendEvaluator,
    BackendEvaluatorConfig,
)
from place_layers.evaluator.lm_eval_utils import (
    Docs,
    TaskSpecInput,
    get_docs,
    get_task_name,
    load_tasks,
)
from place_layers.evaluator.progress import create_progress
from place_layers.evaluator.traces.types import SkippedProblem

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


class TaskBasedEvaluator(BackendEvaluator):
    """Base class for evaluators that iterate over lm-eval tasks.

    Extends BackendEvaluator with task iteration infrastructure:
    - _prepare_task(): Get task from cache, validate type, check resumability
    - Uses self._trace_writer.get_completed_problems() for resumability
    - Caches tasks and docs for efficiency across multiple placements

    Subclasses must implement:
    - evaluate(): The actual evaluation logic
    - _get_fewshot_seed(): Return seed for few-shot sampling (or None)

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "TaskBasedEvaluatorConfig"

    def __init__(
        self,
        config: "TaskBasedEvaluatorConfig",
        *,
        output_dir: "Path",
        backend: "AsyncBackend | SyncBackend",
    ) -> None:
        """Initialize task-based evaluator.

        Args:
            config: Evaluator configuration with task settings.
            output_dir: Base output directory for results.
            backend: Pre-created backend instance (lifecycle managed by caller).
        """
        super().__init__(config, output_dir=output_dir, backend=backend)
        # Cache for batch-loaded tasks (populated in __aenter__)
        self._task_cache: dict[str, "Task"] | None = None
        # Cache for docs - persists across placements (populated by prefetcher)
        self._docs_cache: dict[str, Docs] = {}
        # Events to signal when docs are ready (for prefetch synchronization)
        # Uses asyncio.Event so we can await without blocking the event loop
        self._docs_ready: dict[str, asyncio.Event] = {}
        # Background prefetch task
        self._prefetch_task: asyncio.Task[None] | None = None

    def _get_fewshot_seed(self) -> int | None:
        """Return seed for few-shot sampling, or None for default behavior.

        Subclasses can override this to provide a specific seed.
        """
        return None

    async def __aenter__(self) -> Self:
        """Enter context: initialize backend, load tasks, start doc prefetcher.

        Tasks are loaded here (not in __init__) because:
        1. Backend must be initialized first
        2. Seed may depend on evaluator-specific logic (_get_fewshot_seed)
        3. Loading all tasks together is ~6x faster than individual loading
        4. Doc prefetching starts immediately to overlap with evaluation
        """
        await super().__aenter__()

        # Batch load all tasks for efficiency
        seed = self._get_fewshot_seed()
        metadata: dict[str, Any] | None = None
        if self.config.tokenizer or self.config.max_seq_lengths:
            metadata = {}
            if self.config.tokenizer:
                metadata["pretrained"] = self.config.tokenizer
            if self.config.max_seq_lengths is not None:
                lengths = self.config.max_seq_lengths
                metadata["max_seq_lengths"] = (
                    [lengths] if isinstance(lengths, int) else lengths
                )
        logger.info(f"Batch-loading {len(self.config.tasks)} tasks...")
        self._task_cache = load_tasks(
            self.config.tasks, self.config.num_fewshot, seed, metadata=metadata
        )
        logger.info(f"Loaded {len(self._task_cache)} tasks")

        # Initialize doc-ready events for each task
        for task_name in self._task_cache:
            self._docs_ready[task_name] = asyncio.Event()
            # If already cached (from previous placement), mark as ready
            if task_name in self._docs_cache:
                self._docs_ready[task_name].set()

        # Start background doc prefetcher (runs in parallel with evaluation)
        self._prefetch_task = asyncio.create_task(
            self._prefetch_docs(prefetch_ahead=self.config.doc_prefetch_count)
        )

        return self

    async def _prefetch_docs(self, prefetch_ahead: int) -> None:
        """Prefetch docs for tasks with limited concurrency.

        Runs in background, loading docs ahead of evaluation to hide I/O latency.
        Uses asyncio.to_thread to avoid blocking the event loop.

        Args:
            prefetch_ahead: Max number of tasks to prefetch concurrently.
        """
        sem = asyncio.Semaphore(prefetch_ahead)

        async def load_one(task_spec: TaskSpecInput) -> None:
            task_name = get_task_name(task_spec)
            async with sem:
                if task_name not in self._docs_cache:
                    task = self._task_cache[task_name]  # type: ignore[index]
                    # Run blocking I/O in thread to not block event loop
                    task.download()
                    docs = await asyncio.to_thread(get_docs, task, self.config.split)
                    self._docs_cache[task_name] = docs
                    logger.debug(f"Prefetched docs for '{task_name}'")
                self._docs_ready[task_name].set()

        await asyncio.gather(*[load_one(ts) for ts in self.config.tasks])

    async def _prepare_task(
        self,
        task_spec: TaskSpecInput,
        expected_output_type: Literal["generate_until", "multiple_choice"],
    ) -> tuple["Task", str, Docs, list[int]] | None:
        """Get task from cache and prepare for evaluation.

        This handles:
        - Retrieving the pre-loaded task from cache
        - Validating task type (generate_until or multiple_choice)
        - Waiting for docs to be prefetched (async, doesn't block event loop)
        - Applying max_problems limit
        - Checking completed problems for resumability
        - Computing remaining indices

        Args:
            task_spec: Task name or TaskSpec with overrides.
            expected_output_type: Expected task type ('generate_until' or 'multiple_choice').

        Returns:
            None if task is already complete.
            (task, task_name, docs, remaining_indices) otherwise.
        """
        task_name = get_task_name(task_spec)
        logger.info(f"Task '{task_name}': preparing...")

        # Tasks must be pre-loaded in __aenter__
        if self._task_cache is None:
            raise RuntimeError(
                "Tasks not loaded. TaskBasedEvaluator must be used as async context manager."
            )

        task = self._task_cache[task_name]

        # Validate task type
        if task.config.output_type != expected_output_type:
            raise ValueError(
                f"Task '{task_name}' has output_type='{task.config.output_type}', "
                f"expected '{expected_output_type}'"
            )

        # Wait for docs to be prefetched (yields to event loop, doesn't block)
        if task_name in self._docs_ready:
            await self._docs_ready[task_name].wait()

        # Get documents from cache (should be populated by prefetcher)
        if task_name not in self._docs_cache:
            # Fallback: load synchronously if not prefetched
            task.download()
            self._docs_cache[task_name] = get_docs(task, self.config.split)

        docs = self._docs_cache[task_name]

        # Apply max_problems limit
        total_problems = len(docs)
        if self.config.max_problems_per_task is not None:
            total_problems = min(total_problems, self.config.max_problems_per_task)
            # Convert to list for slicing (docs might be HF Dataset)
            docs = cast(Docs, [docs[i] for i in range(total_problems)])

        # Check completed problems for resumability
        completed = self._trace_writer.get_completed_problems(task_name)
        remaining = [i for i in range(total_problems) if i not in completed]

        if not remaining:
            logger.info(f"Task '{task_name}': already complete, skipping")
            return None

        logger.info(
            f"Task '{task_name}': {len(remaining)} remaining "
            f"({len(completed)} already done, {total_problems} total)"
        )

        return task, task_name, docs, remaining

    # -------------------------------------------------------------------------
    # Abstract hooks for streaming evaluation
    # -------------------------------------------------------------------------

    @abstractmethod
    def _expected_output_type(self) -> Literal["generate_until", "multiple_choice"]:
        """Return expected task output type for validation.

        Returns:
            "generate_until" for generative tasks, "multiple_choice" for likelihood.
        """
        pass

    @abstractmethod
    def _build_request(
        self, task: "Task", doc: dict, idx: int, task_name: str
    ) -> tuple[Any, Any]:
        """Build a request and context for a single problem.

        Args:
            task: The lm-eval task.
            doc: The problem document.
            idx: Problem index within the task.
            task_name: Name of the task.

        Returns:
            (request, context) where request is sent to backend,
            context is stored for result processing. Context can be None
            if not needed for processing.
        """
        pass

    @abstractmethod
    def _stream_to_backend_async(
        self, requests: AsyncIterable[Any]
    ) -> AsyncIterator[Any]:
        """Stream requests to async backend, yield results.

        Args:
            requests: Async iterable of requests to send.

        Returns:
            Async iterator of results (may include BackendError).
        """
        pass

    @abstractmethod
    def _stream_to_backend_sync(
        self, requests: Iterable[Any]
    ) -> Iterator[Any]:
        """Stream requests to sync backend, yield results.

        Args:
            requests: Iterable of requests to send.

        Returns:
            Iterator of results (may include BackendError).
        """
        pass

    @abstractmethod
    def _process_result(
        self,
        result: Any,
        context: Any,
        task: "Task",
        task_name: str,
        doc: dict,
        idx: int,
    ) -> None:
        """Process a single result and write to disk.

        Args:
            result: Result from backend.
            context: Context returned by _build_request.
            task: The lm-eval task.
            task_name: Name of the task.
            doc: The problem document.
            idx: Problem index within the task.
        """
        pass

    # -------------------------------------------------------------------------
    # Streaming methods
    # -------------------------------------------------------------------------

    async def _run_streaming_async(self) -> int:
        """Run streaming evaluation with async backend.

        Uses interleaved prefetch - later tasks load while earlier ones evaluate.
        The _prepare_task calls are async, yielding to event loop to allow
        prefetch of subsequent tasks to run concurrently.

        Returns:
            Number of problems processed.
        """
        assert isinstance(self._backend, AsyncBackend)

        key_to_context: dict[str, tuple["Task", str, dict, int, Any]] = {}

        async def request_generator():
            for task_spec in self.config.tasks:
                # Wait for this task's docs (yields to event loop, letting prefetch run)
                prepared = await self._prepare_task(
                    task_spec, self._expected_output_type()
                )
                if prepared is None:
                    continue

                task, task_name, docs, remaining = prepared
                logger.info(
                    f"Task '{task_name}': evaluating {len(remaining)} problems"
                )

                # Build requests for this task
                for idx in remaining:
                    doc = docs[idx]
                    request, context = self._build_request(task, doc, idx, task_name)
                    key_to_context[request.key] = (task, task_name, doc, idx, context)
                    yield request

        results_stream = self._stream_to_backend_async(request_generator())
        processed = 0

        with create_progress() as progress:
            progress_task = progress.add_task("[cyan]Evaluating", total=None)
            async for result in results_stream:
                if isinstance(result, BackendError):
                    self._handle_backend_error(result)
                else:
                    task, task_name, doc, idx, context = key_to_context[result.key]
                    self._process_result(result, context, task, task_name, doc, idx)
                processed += 1
                progress.advance(progress_task)

        logger.info(f"Evaluation complete: {processed} problems processed")
        return processed

    async def _run_streaming_sync(self) -> int:
        """Run streaming evaluation with sync backend.

        Prepares all tasks upfront (sync backends block event loop during
        generation), then streams requests.

        Returns:
            Number of problems processed.
        """
        assert isinstance(self._backend, SyncBackend)

        key_to_context: dict[str, tuple["Task", str, dict, int, Any]] = {}
        prepared_tasks: list[tuple["Task", str, list[int]]] = []
        total_problems = 0

        # Prepare all tasks upfront (async - waits for prefetch)
        for task_spec in self.config.tasks:
            prepared = await self._prepare_task(
                task_spec, self._expected_output_type()
            )
            if prepared is None:
                continue

            task, task_name, docs, remaining = prepared
            prepared_tasks.append((task, task_name, remaining))
            total_problems += len(remaining)
            logger.info(f"Task '{task_name}': {len(remaining)} problems to evaluate")

        if total_problems == 0:
            logger.info("All tasks already complete")
            return 0

        # Sync request generator (uses pre-prepared data)
        def request_generator():
            for task, task_name, remaining in prepared_tasks:
                docs = self._docs_cache[task_name]
                for idx in remaining:
                    doc = docs[idx]
                    request, context = self._build_request(task, doc, idx, task_name)
                    key_to_context[request.key] = (task, task_name, doc, idx, context)
                    yield request

        results_iter = self._stream_to_backend_sync(request_generator())
        processed = 0

        with create_progress() as progress:
            progress_task = progress.add_task("[cyan]Evaluating", total=total_problems)
            for result in results_iter:
                if isinstance(result, BackendError):
                    self._handle_backend_error(result)
                else:
                    task, task_name, doc, idx, context = key_to_context[result.key]
                    self._process_result(result, context, task, task_name, doc, idx)
                processed += 1
                progress.advance(progress_task)

        logger.info(f"Evaluation complete: {processed} problems processed")
        return processed

    def _handle_backend_error(self, error: BackendError) -> None:
        """Handle a backend error by writing a SkippedProblem record.

        Args:
            error: The backend error to handle.
        """
        assert self._trace_writer is not None, "trace_writer not initialized"
        # Extract task_name from key (format: "task_name:idx")
        task_name = error.key.rsplit(":", 1)[0]
        logger.warning(f"Skipping {error.key}: {error.error_type} - {error.message}")
        self._trace_writer.append([
            SkippedProblem(
                problem_id=error.key,
                task_name=task_name,
                reason=f"{error.error_type}: {error.message}",
            )
        ])


class TaskBasedEvaluatorConfig(BackendEvaluatorConfig):
    """Config for evaluators that process lm-eval benchmark tasks."""

    tasks: list[TaskSpecInput] = Field(
        description="List of lm-eval tasks to evaluate (e.g., ['gsm8k', 'math'])"
    )
    num_fewshot: int = Field(
        description="Number of few-shot examples",
        ge=0,
    )
    max_problems_per_task: int | None = Field(
        description="Max problems per task (None = all). Use for faster evaluation.",
    )
    split: str = Field(
        description="Dataset split to use ('test', 'validation', or 'train' if fitting a router)",
    )
    doc_prefetch_count: int = Field(
        description="Number of tasks to prefetch docs for ahead of evaluation. "
        "Higher values hide I/O latency but use more memory.",
        ge=1,
    )
    tokenizer: str | None = Field(
        default=None,
        description="Tokenizer name/path for synthetic task generation (e.g., RULER). "
        "Required when using tasks with custom_dataset that need a tokenizer.",
    )
    max_seq_lengths: list[int] | int | None = Field(
        default=None,
        description="Context lengths for synthetic tasks (e.g., RULER). "
        "Accepts a single int or list of ints. "
        "Generates samples at each specified length. Default depends on task.",
    )

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        """Validate that at least one task is specified."""
        if not self.tasks:
            raise ValueError("At least one task must be specified")
        return self
