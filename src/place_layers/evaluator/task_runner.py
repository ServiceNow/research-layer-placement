"""Shared task running infrastructure for generation commands.

This module provides common functionality for running lm-eval tasks with
generation backends. Both precompute_traces and generative evaluation
use this infrastructure.

Key components:
- Task preparation (loading, docs, resumability checking)
- Sync batched generation loop with error handling
- Async worker pool generation loop with error handling
- SkippedProblem handling for errors and truncations
- Memory logging
- Progress logging

Callers provide a record factory function to create the appropriate
record type (Trace or Generation) from generation results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from place_layers.evaluator.importance_weighted.generation_backend import (
    AsyncBackend,
    SyncBackend,
)
from place_layers.evaluator.importance_weighted.memory_logging import (
    force_garbage_collection,
    get_memory_stats,
    log_gpu_memory_summary,
    log_memory,
)
from place_layers.evaluator.importance_weighted.trace_writer import (
    TraceWriter,
    get_completed_problems,
)
from place_layers.evaluator.importance_weighted.traces import SkippedProblem, TraceRecord
from place_layers.evaluator.importance_weighted.vllm_client import (
    GenerationResult,
    VLLMBadRequestError,
)
from place_layers.evaluator.lm_eval_utils import (
    Docs,
    TaskSpecInput,
    build_chat_messages,
    get_docs,
    get_task_name,
    load_task,
)

if TYPE_CHECKING:
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


# Type alias for record factory function
# Returns list of records (Trace or Generation). Empty list means all samples were truncated/invalid.
RecordFactory = Callable[["Task", dict, int, GenerationResult], list[TraceRecord]]


def _batched(iterable: Sequence, n: int):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


# =============================================================================
# Task Preparation
# =============================================================================


def prepare_task(
    task_spec: TaskSpecInput,
    output_dir: Path,
    num_fewshot: int,
    max_problems: int | None,
    split: str,
    seed: int | None = None,
) -> tuple["Task", str, Docs, list[int]] | None:
    """Load task and get remaining problem indices.

    This handles:
    - Loading the lm-eval task
    - Getting documents for the split
    - Applying max_problems limit
    - Checking completed problems for resumability
    - Computing remaining indices

    Args:
        task_spec: Task name or TaskSpec with overrides.
        output_dir: Directory containing validation.jsonl for resumability.
        num_fewshot: Number of few-shot examples.
        max_problems: Maximum problems to process (None = all).
        split: Dataset split ('test' or 'validation').
        seed: Random seed for few-shot sampling.

    Returns:
        None if task is already complete.
        (task, task_name, docs, remaining_indices) otherwise.
    """
    task_name = get_task_name(task_spec)
    logger.info(f"Task '{task_name}': loading...")

    # Load task
    task = load_task(task_spec, num_fewshot, seed)

    # Validate task type
    if task.config.output_type != "generate_until":
        raise ValueError(
            f"Task '{task_name}' has output_type='{task.config.output_type}', "
            f"expected 'generate_until'"
        )

    # Download dataset if needed
    task.download()

    # Get documents
    docs = get_docs(task, split)

    # Apply max_problems limit
    total_problems = len(docs)
    if max_problems is not None:
        total_problems = min(total_problems, max_problems)
        # Convert to list for slicing (docs might be HF Dataset)
        docs = [docs[i] for i in range(total_problems)]

    # Check completed problems for resumability
    completed = get_completed_problems(output_dir, task_name)
    remaining = [i for i in range(total_problems) if i not in completed]

    if not remaining:
        logger.info(f"Task '{task_name}': already complete, skipping")
        return None

    logger.info(
        f"Task '{task_name}': {len(remaining)} remaining "
        f"({len(completed)} already done, {total_problems} total)"
    )

    return task, task_name, docs, remaining


# =============================================================================
# Sync Generation Loop
# =============================================================================


def run_generation_sync(
    task: "Task",
    task_name: str,
    docs: Docs,
    remaining_indices: list[int],
    backend: SyncBackend,
    writer: TraceWriter,
    record_factory: RecordFactory,
    samples_per_problem: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    log_interval: float = 10.0,
    memory_log_interval: float = 30.0,
) -> int:
    """Run generation synchronously with batching.

    Includes:
    - Batched generation for efficiency
    - Error handling with SkippedProblem records
    - Truncation handling (empty records -> SkippedProblem)
    - Memory logging
    - Progress logging

    Args:
        task: The lm-eval task.
        task_name: Task name for logging and problem IDs.
        docs: Document collection.
        remaining_indices: Problem indices to process.
        backend: Sync generation backend.
        writer: TraceWriter for saving records.
        record_factory: Function to create records from results.
        samples_per_problem: Number of samples per problem.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        log_interval: Seconds between progress logs.
        memory_log_interval: Seconds between memory logs.

    Returns:
        Number of problems processed.
    """
    total = len(remaining_indices)
    batch_size = backend.batch_size

    start_time = time.perf_counter()
    last_log_time = start_time
    last_memory_log_time = start_time
    processed = 0

    logger.info(f"Processing {total} problems with batch_size={batch_size}")
    log_memory(f"TASK {task_name} START")

    for batch_indices in _batched(remaining_indices, batch_size):
        # Build messages for this batch
        batch_docs = [docs[idx] for idx in batch_indices]
        messages_batch = [
            build_chat_messages(task, doc, task.config.num_fewshot)
            for doc in batch_docs
        ]

        # Log progress periodically
        current_time = time.perf_counter()
        if current_time - last_log_time >= log_interval:
            elapsed = current_time - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            logger.info(
                f"Task '{task_name}': {processed}/{total} "
                f"({100*processed/total:.1f}%), ETA {eta:.0f}s"
            )
            last_log_time = current_time

        # Log memory periodically
        if current_time - last_memory_log_time >= memory_log_interval:
            log_memory(f"BATCH {processed}/{total}")
            last_memory_log_time = current_time

        # Generate completions
        try:
            results = backend.batch_generate(
                messages_batch,
                n=samples_per_problem,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
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
                writer.append([skipped])

            # Try to recover memory
            force_garbage_collection()
            processed += len(batch_indices)
            continue

        # Create and save records for each problem
        for idx, doc, result in zip(batch_indices, batch_docs, results):
            records = record_factory(task, doc, idx, result)

            if records:
                writer.append(records)
            else:
                # All samples were truncated/invalid
                skipped = SkippedProblem(
                    problem_id=f"{task_name}:{idx}",
                    task_name=task_name,
                    reason=f"all {len(result.samples)} samples truncated",
                )
                writer.append([skipped])

        processed += len(batch_indices)

    # Log final memory state
    log_memory(f"TASK {task_name} END")
    logger.info(f"Task '{task_name}': generation complete ({processed} problems)")

    return processed


# =============================================================================
# Async Generation Loop
# =============================================================================


async def run_generation_async(
    task: "Task",
    task_name: str,
    docs: Docs,
    remaining_indices: list[int],
    backend: AsyncBackend,
    writer: TraceWriter,
    record_factory: RecordFactory,
    samples_per_problem: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    log_interval: float = 10.0,
) -> int:
    """Run generation asynchronously with worker pool.

    Includes:
    - Concurrent worker pool for throughput
    - Error handling with SkippedProblem records
    - Truncation handling (empty records -> SkippedProblem)
    - Progress logging

    Args:
        task: The lm-eval task.
        task_name: Task name for logging and problem IDs.
        docs: Document collection.
        remaining_indices: Problem indices to process.
        backend: Async generation backend.
        writer: TraceWriter for saving records.
        record_factory: Function to create records from results.
        samples_per_problem: Number of samples per problem.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        log_interval: Seconds between progress logs.

    Returns:
        Number of problems processed.
    """
    total = len(remaining_indices)

    queue: asyncio.Queue[int] = asyncio.Queue()
    for idx in remaining_indices:
        await queue.put(idx)

    processed_count = 0
    start_time = time.perf_counter()
    write_lock = asyncio.Lock()
    done = False

    async def worker():
        nonlocal processed_count
        while True:
            try:
                idx = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            doc = docs[idx]
            messages = build_chat_messages(task, doc, task.config.num_fewshot)

            try:
                result = await backend.agenerate(
                    messages,
                    n=samples_per_problem,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    request_id=f"{task_name}:{idx}",
                )

                # Create records
                records = record_factory(task, doc, idx, result)

                async with write_lock:
                    if records:
                        writer.append(records)
                    else:
                        # All samples were truncated/invalid
                        skipped = SkippedProblem(
                            problem_id=f"{task_name}:{idx}",
                            task_name=task_name,
                            reason=f"all {len(result.samples)} samples truncated",
                        )
                        writer.append([skipped])

            except VLLMBadRequestError as e:
                logger.warning(f"Skipping {task_name}:{idx} - {e}")
                async with write_lock:
                    skipped = SkippedProblem(
                        problem_id=f"{task_name}:{idx}",
                        task_name=task_name,
                        reason=str(e),
                    )
                    writer.append([skipped])

            except Exception as e:
                logger.warning(f"Problem {task_name}:{idx} failed: {e}")
                async with write_lock:
                    skipped = SkippedProblem(
                        problem_id=f"{task_name}:{idx}",
                        task_name=task_name,
                        reason=str(e),
                    )
                    writer.append([skipped])

            processed_count += 1

    # Progress logging task
    async def log_progress():
        last_count = 0
        while not done:
            await asyncio.sleep(log_interval)
            if processed_count > last_count:
                elapsed = time.perf_counter() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta = (total - processed_count) / rate if rate > 0 else 0
                logger.info(
                    f"Task '{task_name}': {processed_count}/{total} "
                    f"({100*processed_count/total:.1f}%), ETA {eta:.0f}s"
                )
                last_count = processed_count

    # Start workers and progress logger
    num_workers = min(backend.max_concurrent_requests, total)
    workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
    progress_task = asyncio.create_task(log_progress())

    # Wait for completion
    await asyncio.gather(*workers)
    done = True
    progress_task.cancel()
    try:
        await progress_task
    except asyncio.CancelledError:
        pass

    logger.info(f"Task '{task_name}': generation complete ({processed_count} problems)")
    return processed_count
