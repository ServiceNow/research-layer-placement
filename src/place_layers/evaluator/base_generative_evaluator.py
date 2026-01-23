"""Base class for evaluators that generate completions.

This module provides BaseGenerativeEvaluator which handles the generation
loop infrastructure for evaluators that sample completions from models:

- Sync batched generation with memory logging
- Async worker pool generation for throughput
- SkippedProblem handling for generation errors
- Progress bars
- Resumability via _prepare_task()

Concrete evaluators (GenerativeEvaluator, TracePrecomputeEvaluator) inherit
from this and provide their record factory function.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Sequence

from pydantic import Field

from place_layers.backend.base import (
    AsyncBackend,
    SyncBackend,
)
from place_layers.backend.vllm import (
    GenerationResult,
    VLLMBadRequestError,
)
from place_layers.core.memory_logging import (
    force_garbage_collection,
    log_gpu_memory_summary,
    log_memory,
)
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.lm_eval_utils import Docs, build_chat_messages, build_text_prompt
from place_layers.evaluator.progress import create_progress
from place_layers.evaluator.task_based_evaluator import (
    TaskBasedEvaluator,
    TaskBasedEvaluatorConfig,
)
from place_layers.evaluator.traces.types import SkippedProblem, TraceRecord
from place_layers.evaluator.traces.writer import summarize_generations

if TYPE_CHECKING:
    from lm_eval.api.task import Task

    from place_layers.evaluator.traces.writer import GenerationSummary

logger = logging.getLogger(__name__)


def _batched(iterable: Sequence, n: int):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


class BaseGenerativeEvaluator(TaskBasedEvaluator):
    """Base class for evaluators that generate completions.

    Provides the generation loop infrastructure:
    - _run_tasks_sync(): Batched sync generation with memory logging
    - _run_tasks_async(): Worker pool async generation
    - evaluate(): Handles placement, dispatches to sync/async, computes summary

    Subclasses must implement:
    - _make_records(): Create records from generation results
    - _get_fewshot_seed(): Return seed for few-shot sampling (or None)

    Must be used as an async context manager:

        async with config.create_evaluator() as evaluator:
            score = await evaluator.evaluate(placement)
    """

    config: "BaseGenerativeEvaluatorConfig"

    def __init__(self, config: "BaseGenerativeEvaluatorConfig") -> None:
        super().__init__(config)

    @abstractmethod
    def _make_records(
        self, task: "Task", doc: dict, idx: int, result: GenerationResult
    ) -> Sequence[TraceRecord]:
        """Create records from a generation result.

        Args:
            task: The lm-eval task.
            doc: The problem document.
            idx: Problem index.
            result: Generation result with samples.

        Returns:
            Sequence of records (Trace or Generation). Empty sequence means all
            samples were truncated/invalid.
        """
        pass

    def _get_fewshot_seed(self) -> int | None:
        """Return seed for few-shot sampling, or None for default behavior."""
        return None

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Evaluate by generating completions and checking accuracy.

        Args:
            parameters: Layer placement to evaluate, or None for default config.

        Returns:
            Problem accuracy (fraction with any correct sample).
        """
        if parameters is not None:
            await self._set_placement(parameters)
        else:
            logger.debug("No placement specified, using default configuration")

        # Dispatch to async or sync generation
        if isinstance(self._backend, AsyncBackend):
            await self._run_tasks_async()
        elif isinstance(self._backend, SyncBackend):
            self._run_tasks_sync()
        else:
            raise TypeError(f"Unknown backend type: {type(self._backend)}")

        # Compute summary from disk (single source of truth)
        summary = self._summarize()
        logger.info(f"\n{summary}")

        return summary.problem_accuracy

    def _summarize(self) -> "GenerationSummary":
        """Compute summary from disk. Override for different summary types."""
        return summarize_generations(self._output_dir)

    def _run_tasks_sync(self) -> None:
        """Run all tasks synchronously with batching."""
        assert isinstance(self._backend, SyncBackend)

        for task_spec in self.config.tasks:
            prepared = self._prepare_task(task_spec, "generate_until")

            if prepared is None:
                continue

            task, task_name, docs, remaining = prepared
            self._run_generation_sync(task, task_name, docs, remaining)

    def _run_generation_sync(
        self,
        task: "Task",
        task_name: str,
        docs: Docs,
        remaining_indices: list[int],
        memory_log_interval: float = 30.0,
    ) -> int:
        """Run sync generation with batching, error handling, and memory logging.

        Args:
            task: The lm-eval task.
            task_name: Task name for logging and problem IDs.
            docs: Document collection.
            remaining_indices: Problem indices to process.
            memory_log_interval: Seconds between memory logs.

        Returns:
            Number of problems processed.
        """
        assert isinstance(self._backend, SyncBackend)

        total = len(remaining_indices)
        batch_size = self._backend.batch_size

        last_memory_log_time = time.perf_counter()
        processed = 0

        logger.info(f"Processing {total} problems with batch_size={batch_size}")
        log_memory(f"TASK {task_name} START")

        with create_progress() as progress:
            progress_task = progress.add_task(f"[cyan]{task_name}", total=total)

            for batch_indices in _batched(remaining_indices, batch_size):
                # Build prompts for this batch
                batch_docs = [docs[idx] for idx in batch_indices]

                # Log memory periodically
                current_time = time.perf_counter()
                if current_time - last_memory_log_time >= memory_log_interval:
                    log_memory(f"BATCH {processed}/{total}")
                    last_memory_log_time = current_time

                # Generate completions (dispatch based on use_chat_template)
                try:
                    if self.config.use_chat_template:
                        # Chat-based generation (default)
                        messages_batch = [
                            build_chat_messages(task, doc, task.config.num_fewshot or 0)
                            for doc in batch_docs
                        ]
                        results = self._backend.batch_generate(
                            messages_batch,
                            n=self.config.samples_per_problem,
                            max_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                    else:
                        # Text-based generation (no chat template)
                        prompts = [
                            build_text_prompt(task, doc, task.config.num_fewshot or 0)
                            for doc in batch_docs
                        ]
                        results = self._backend.batch_generate_text(
                            prompts,
                            n=self.config.samples_per_problem,
                            max_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                except Exception as e:
                    # Log memory state on exception
                    logger.error(f"Batch failed: {type(e).__name__}: {e}")
                    log_memory(f"BATCH {processed} EXCEPTION", level=logging.ERROR)

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
                        self._trace_writer.append_sync([skipped])

                    # Try to recover memory
                    force_garbage_collection()
                    processed += len(batch_indices)
                    progress.advance(progress_task, len(batch_indices))
                    continue

                # Create and save records for each problem
                for idx, doc, result in zip(batch_indices, batch_docs, results):
                    records = self._make_records(task, doc, idx, result)

                    if records:
                        self._trace_writer.append_sync(records)
                    else:
                        # No samples returned (backend issue)
                        skipped = SkippedProblem(
                            problem_id=f"{task_name}:{idx}",
                            task_name=task_name,
                            reason="no samples returned",
                        )
                        self._trace_writer.append_sync([skipped])

                processed += len(batch_indices)
                progress.advance(progress_task, len(batch_indices))

        # Log final memory state
        log_memory(f"TASK {task_name} END")
        logger.info(f"Task '{task_name}': generation complete ({processed} problems)")

        return processed

    async def _run_tasks_async(self) -> None:
        """Run all tasks asynchronously with worker pool."""
        assert isinstance(self._backend, AsyncBackend)

        for task_spec in self.config.tasks:
            prepared = self._prepare_task(task_spec, "generate_until")

            if prepared is None:
                continue

            task, task_name, docs, remaining = prepared
            await self._run_generation_async(task, task_name, docs, remaining)

    async def _run_generation_async(
        self,
        task: "Task",
        task_name: str,
        docs: Docs,
        remaining_indices: list[int],
    ) -> int:
        """Run async generation with worker pool and error handling.

        Args:
            task: The lm-eval task.
            task_name: Task name for logging and problem IDs.
            docs: Document collection.
            remaining_indices: Problem indices to process.

        Returns:
            Number of problems processed.
        """
        assert isinstance(self._backend, AsyncBackend)
        backend = self._backend  # Local var for type narrowing in nested function

        total = len(remaining_indices)

        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in remaining_indices:
            await queue.put(idx)

        processed_count = 0
        write_lock = asyncio.Lock()

        with create_progress() as progress:
            progress_task = progress.add_task(f"[cyan]{task_name}", total=total)

            async def worker():
                nonlocal processed_count
                while True:
                    try:
                        idx = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    doc = docs[idx]

                    try:
                        # Generate completions (dispatch based on use_chat_template)
                        if self.config.use_chat_template:
                            # Chat-based generation (default)
                            messages = build_chat_messages(task, doc, task.config.num_fewshot or 0)
                            result = await backend.agenerate(
                                messages,
                                n=self.config.samples_per_problem,
                                max_tokens=self.config.max_new_tokens,
                                temperature=self.config.temperature,
                                top_p=self.config.top_p,
                                request_id=f"{task_name}:{idx}",
                            )
                        else:
                            # Text-based generation (no chat template)
                            prompt = build_text_prompt(task, doc, task.config.num_fewshot or 0)
                            result = await backend.agenerate_text(
                                prompt,
                                n=self.config.samples_per_problem,
                                max_tokens=self.config.max_new_tokens,
                                temperature=self.config.temperature,
                                top_p=self.config.top_p,
                                request_id=f"{task_name}:{idx}",
                            )

                        # Create records
                        records = self._make_records(task, doc, idx, result)

                        async with write_lock:
                            if records:
                                await self._trace_writer.append(records)
                            else:
                                # No samples returned (backend issue)
                                skipped = SkippedProblem(
                                    problem_id=f"{task_name}:{idx}",
                                    task_name=task_name,
                                    reason="no samples returned",
                                )
                                await self._trace_writer.append([skipped])

                    except VLLMBadRequestError as e:
                        logger.warning(f"Skipping {task_name}:{idx} - {e}")
                        async with write_lock:
                            skipped = SkippedProblem(
                                problem_id=f"{task_name}:{idx}",
                                task_name=task_name,
                                reason=str(e),
                            )
                            await self._trace_writer.append([skipped])

                    except Exception as e:
                        logger.warning(f"Problem {task_name}:{idx} failed: {e}")
                        async with write_lock:
                            skipped = SkippedProblem(
                                problem_id=f"{task_name}:{idx}",
                                task_name=task_name,
                                reason=str(e),
                            )
                            await self._trace_writer.append([skipped])

                    processed_count += 1
                    progress.advance(progress_task)

            # Start workers
            num_workers = min(self._backend.max_concurrent_requests, total)
            workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

            # Wait for completion
            await asyncio.gather(*workers)

        logger.info(f"Task '{task_name}': generation complete ({processed_count} problems)")
        return processed_count


class BaseGenerativeEvaluatorConfig(TaskBasedEvaluatorConfig):
    """Generation parameters for GenerativeEvaluator and TracePrecomputeEvaluator."""

    samples_per_problem: int = Field(
        description="Number of completions per problem",
        gt=0,
    )
    max_new_tokens: int = Field(
        description="Maximum tokens to generate per completion",
        gt=0,
    )
    temperature: float = Field(
        description="Sampling temperature (0.0 for greedy)",
        ge=0.0,
    )
    top_p: float = Field(
        description="Nucleus sampling parameter",
        gt=0.0,
        le=1.0,
    )
    use_chat_template: bool = Field(
        default=True,
        description="Whether to use chat template formatting. When False, prompts are "
        "formatted as raw text without any chat structure. "
        "Use False for evaluating base models or when chat template is undesired.",
    )
    evaluate_truncated_samples: bool = Field(
        default=False,
        description="Whether to evaluate truncated samples for correctness. "
        "When True, samples that reach max_new_tokens are still checked against the ground truth. "
        "Useful for extraction tasks where the answer may be complete before hitting the limit. "
        "When False (default), truncated samples are automatically marked as incorrect.",
    )
