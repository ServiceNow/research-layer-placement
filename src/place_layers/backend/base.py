"""Abstract base classes for generation backends.

Provides a categorical abstraction for generation backends:
- AsyncBackend: For remote servers (vLLM, sglang, OpenAI-compatible APIs)
- SyncBackend: For local models (transformers)

Scoring methods use an iterator-based API with keyed requests:
- Caller submits requests with unique keys
- Backend yields results in any order (first-completed for async)
- Caller joins results by key, not position

This design hides batching/concurrency from callers:
- SyncBackend: Chunks requests internally, yields as batches complete
- AsyncBackend: Manages concurrency with semaphores, yields as requests complete
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
    Iterator,
)
from typing import Any, ClassVar, Self, Sequence

from pydantic import BaseModel, Field

from place_layers.backend.types import (
    BackendError,
    ChatMessage,
    GenerateChatRequest,
    GenerateTextRequest,
    ModelStructure,
    ScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest,
    ScoreTextResult,
    ScoreTokensRequest,
    ScoreTokensResult,
)
from place_layers.clients.types import GenerationResult


# =============================================================================
# Concurrency/Batching Helpers
# =============================================================================


def batched_map[T, R](
    items: Iterable[T | None],
    process: Callable[[list[T]], Iterable[R]],
    batch_size: int,
    flush_timeout: float | None = None,
) -> Iterator[R]:
    """Process items in batches, yielding results as batches complete.

    Accumulates items up to batch_size, calls process on the batch,
    and yields all results. Handles the final partial batch.

    This is a special case of batched_expand_map where each item expands
    to exactly one element.

    Args:
        items: Items to process. If flush_timeout is set, None values are
            treated as heartbeats for timeout checking.
        process: Function that takes a batch and returns results (1:1 with input).
        batch_size: Maximum items per batch.
        flush_timeout: If set, flush partial batches after this many seconds
            of no new items. Requires items to yield None as heartbeats.

    Yields:
        Results in input order.
    """
    # Use object() as unique keys since items may not be hashable
    item_id = 0

    def expand(item: T) -> tuple[int, list[T]]:
        nonlocal item_id
        key = item_id
        item_id += 1
        return (key, [item])

    def process_batch(elements: list[T]) -> list[R]:
        return list(process(elements))

    def combine(key: int, results: list[R]) -> R:
        return results[0]

    yield from batched_expand_map(
        items, expand, process_batch, combine, batch_size, flush_timeout
    )


def batched_expand_map[In, K, E, R, Out](
    items: Iterable[In | None],
    expand: Callable[[In], tuple[K, list[E]] | None],
    process_batch: Callable[[list[E]], list[R]],
    combine: Callable[[K, list[R]], Out],
    batch_size: int,
    flush_timeout: float | None = None,
) -> Iterator[Out]:
    """Expand items into elements, batch elements, process, and recombine.

    This handles the case where one input expands to multiple elements that
    may span batch boundaries. Elements are batched at fixed size for
    efficient processing, then results are recombined by input.

    Pattern: fan-out → batch → process → fan-in

    Example: Multiple-choice scoring where each question has N choices.
    Questions are expanded to N sequences, sequences are batched for GPU
    efficiency, then results are recombined per question.

    Streaming mode: When flush_timeout is set, the items iterator should
    yield None as heartbeats. This allows timeout-based flushing even when
    no new items are arriving, enabling true streaming from async sources.

    Args:
        items: Input items to process. If flush_timeout is set, None values
            are treated as heartbeats for timeout checking.
        expand: Fan-out function. Returns (key, elements) or None to skip.
            The key identifies this input for recombining results.
        process_batch: Processes a batch of elements, returns results 1:1.
        combine: Fan-in function. Takes key and all results for that input,
            returns the final output.
        batch_size: Fixed batch size for element processing.
        flush_timeout: If set, flush partial batches after this many seconds.
            Requires items to yield None as heartbeats for timeout checks.

    Yields:
        Outputs in input order, as soon as all elements for an input complete.
    """
    # Track pending elements: (key, index within expansion, element)
    pending: list[tuple[K, int, E]] = []

    # Track results per key: key -> [(index, result), ...]
    results_by_key: dict[K, list[tuple[int, R]]] = {}

    # Track expected element count per key
    expected_counts: dict[K, int] = {}

    # Track input order for yielding
    key_order: list[K] = []

    # Track last flush time for timeout-based flushing
    last_flush_time = time.monotonic() if flush_timeout else 0.0

    def process_pending_batch(batch: list[tuple[K, int, E]]) -> Iterator[Out]:
        """Process a batch of pending elements and yield completed outputs."""
        nonlocal last_flush_time

        # Process elements
        elements = [elem for _, _, elem in batch]
        batch_results = process_batch(elements)

        # Store results by key
        for (key, idx, _), result in zip(batch, batch_results):
            if key not in results_by_key:
                results_by_key[key] = []
            results_by_key[key].append((idx, result))

        # Update flush time
        if flush_timeout:
            last_flush_time = time.monotonic()

        # Yield any completed outputs (in order)
        yield from yield_completed()

    def flush_full_batches() -> Iterator[Out]:
        """Process complete batches (batch_size) and yield completed outputs."""
        nonlocal pending

        while len(pending) >= batch_size:
            batch = pending[:batch_size]
            pending = pending[batch_size:]
            yield from process_pending_batch(batch)

    def flush_partial_batch() -> Iterator[Out]:
        """Process any remaining pending elements as a partial batch."""
        nonlocal pending

        if pending:
            batch = pending
            pending = []
            yield from process_pending_batch(batch)

    def should_timeout_flush() -> bool:
        """Check if we should flush due to timeout."""
        if not flush_timeout or not pending:
            return False
        return time.monotonic() - last_flush_time >= flush_timeout

    def yield_completed() -> Iterator[Out]:
        """Yield outputs that have all their elements processed."""
        while key_order:
            key = key_order[0]
            results = results_by_key.get(key, [])

            if len(results) == expected_counts[key]:
                # All elements complete - sort by index and combine
                results.sort(key=lambda x: x[0])
                ordered_results = [r for _, r in results]
                yield combine(key, ordered_results)

                # Cleanup
                del results_by_key[key]
                del expected_counts[key]
                key_order.pop(0)
            else:
                # Not complete yet - can't yield out of order
                break

    # Process inputs
    for item in items:
        # Handle heartbeats (None) for timeout checking
        if item is None:
            if should_timeout_flush():
                yield from flush_partial_batch()
            continue

        expanded = expand(item)

        if expanded is None:
            # Skip this item
            continue

        key, elements = expanded

        # Track this key (even if empty, to preserve order)
        expected_counts[key] = len(elements)
        key_order.append(key)

        if not elements:
            # Empty expansion - mark as ready (will yield when at front of queue)
            results_by_key[key] = []
            yield from yield_completed()
            continue

        # Add elements to pending
        for idx, elem in enumerate(elements):
            pending.append((key, idx, elem))

        # Process any complete batches
        yield from flush_full_batches()

        # Also check timeout flush after adding new items
        if should_timeout_flush():
            yield from flush_partial_batch()

    # Drain remaining elements (final partial batch)
    yield from flush_partial_batch()

    # Yield any remaining completed outputs
    yield from yield_completed()


def batched_map_by_key[T, K, R](
    items: Iterable[T | None],
    get_key: Callable[[T], K],
    process: Callable[[K, list[T]], Iterable[R]],
    batch_size: int,
    flush_timeout: float | None = None,
) -> Iterator[R]:
    """Process items in batches, flushing when key changes.

    Like batched_map, but groups consecutive items by key. A batch is flushed
    when: (1) batch reaches batch_size, (2) key changes, or (3) timeout expires.

    Use case: Generation requests where all items in a batch must share the
    same parameters. When a request arrives with different params, flush the
    current batch before starting a new one.

    Args:
        items: Items to process. None values are heartbeats for timeout checking.
        get_key: Extract grouping key from item (e.g., generation params).
        process: Function that takes (key, batch) and returns results 1:1.
        batch_size: Maximum items per batch.
        flush_timeout: Flush partial batches after this many seconds.

    Yields:
        Results in input order.
    """
    pending: list[T] = []
    current_key: K | None = None
    last_flush_time = time.monotonic() if flush_timeout else 0.0

    def flush() -> Iterator[R]:
        nonlocal pending, last_flush_time
        if pending and current_key is not None:
            yield from process(current_key, pending)
            pending = []
            if flush_timeout:
                last_flush_time = time.monotonic()

    def should_timeout_flush() -> bool:
        if not flush_timeout or not pending:
            return False
        return time.monotonic() - last_flush_time >= flush_timeout

    for item in items:
        if item is None:  # Heartbeat
            if should_timeout_flush():
                yield from flush()
            continue

        item_key = get_key(item)

        if current_key is not None and item_key != current_key:
            yield from flush()

        current_key = item_key
        pending.append(item)

        if len(pending) >= batch_size:
            yield from flush()

        if should_timeout_flush():
            yield from flush()

    yield from flush()


async def _concurrent_expand_map_sync[In, K, E, R, Out](
    items: Iterable[In],
    expand: Callable[[In], tuple[K, list[E]] | None],
    process_one: Callable[[E], Coroutine[Any, Any, R]],
    combine: Callable[[K, list[R]], Out],
    max_concurrent: int,
) -> AsyncIterator[Out]:
    """Implementation for sync iterables. See concurrent_expand_map for docs."""
    # Track results per key: key -> [(index, result), ...]
    results_by_key: dict[K, list[tuple[int, R]]] = {}

    # Track expected element count per key
    expected_counts: dict[K, int] = {}

    # Track input order for yielding
    key_order: list[K] = []

    # Pending tasks: set of tasks, each tagged with (key, idx)
    pending: set[asyncio.Task[tuple[K, int, R]]] = set()

    # Element buffer: elements waiting to be scheduled
    buffer: list[tuple[K, int, E]] = []

    # Item iterator
    items_iter = iter(items)
    items_exhausted = False

    async def process_with_key(key: K, idx: int, element: E) -> tuple[K, int, R]:
        result = await process_one(element)
        return (key, idx, result)

    def expand_more_items() -> None:
        """Expand items to fill the buffer."""
        nonlocal items_exhausted

        while len(buffer) + len(pending) < max_concurrent and not items_exhausted:
            try:
                item = next(items_iter)
            except StopIteration:
                items_exhausted = True
                break

            expanded = expand(item)
            if expanded is None:
                continue

            key, elements = expanded
            key_order.append(key)
            expected_counts[key] = len(elements)

            if not elements:
                # Empty expansion - immediately ready
                results_by_key[key] = []
            else:
                for idx, elem in enumerate(elements):
                    buffer.append((key, idx, elem))

    def schedule_from_buffer() -> None:
        """Move elements from buffer to pending tasks."""
        while buffer and len(pending) < max_concurrent:
            key, idx, elem = buffer.pop(0)
            task = asyncio.create_task(process_with_key(key, idx, elem))
            pending.add(task)

    def yield_completed() -> Iterator[Out]:
        """Yield outputs for keys that have all results, in input order."""
        while key_order:
            key = key_order[0]

            if key not in results_by_key:
                break
            if len(results_by_key[key]) != expected_counts[key]:
                break

            # All results for this key - combine and yield
            results = results_by_key.pop(key)
            del expected_counts[key]
            key_order.pop(0)

            results.sort(key=lambda x: x[0])
            ordered_results = [r for _, r in results]
            yield combine(key, ordered_results)

    # Initial setup
    expand_more_items()
    schedule_from_buffer()

    # Yield any empty expansions that are already complete
    for out in yield_completed():
        yield out

    # Main loop
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        # Collect results
        for task in done:
            key, idx, result = task.result()
            if key not in results_by_key:
                results_by_key[key] = []
            results_by_key[key].append((idx, result))

        # Yield completed outputs
        for out in yield_completed():
            yield out

        # Refill
        expand_more_items()
        schedule_from_buffer()

    # Final yield (for any remaining completed outputs)
    for out in yield_completed():
        yield out


async def _concurrent_expand_map_async[In, K, E, R, Out](
    items: AsyncIterable[In | None],
    expand: Callable[[In], tuple[K, list[E]] | None],
    process_one: Callable[[E], Coroutine[Any, Any, R]],
    combine: Callable[[K, list[R]], Out],
    max_concurrent: int,
) -> AsyncIterator[Out]:
    """Implementation for async iterables. See concurrent_expand_map for docs."""
    # Track results per key: key -> [(index, result), ...]
    results_by_key: dict[K, list[tuple[int, R]]] = {}

    # Track expected element count per key
    expected_counts: dict[K, int] = {}

    # Track input order for yielding
    key_order: list[K] = []

    # Pending tasks: set of tasks, each tagged with (key, idx)
    pending: set[asyncio.Task[tuple[K, int, R]]] = set()

    # Element buffer: elements waiting to be scheduled
    buffer: list[tuple[K, int, E]] = []

    # Async iterator state
    items_aiter = items.__aiter__()
    items_exhausted = False
    next_item_task: asyncio.Task[In | None] | None = None

    async def process_with_key(key: K, idx: int, element: E) -> tuple[K, int, R]:
        result = await process_one(element)
        return (key, idx, result)

    def expand_item(item: In) -> None:
        """Expand a single item and add elements to buffer."""
        expanded = expand(item)
        if expanded is None:
            return

        key, elements = expanded
        key_order.append(key)
        expected_counts[key] = len(elements)

        if not elements:
            results_by_key[key] = []
        else:
            for idx, elem in enumerate(elements):
                buffer.append((key, idx, elem))

    def schedule_from_buffer() -> None:
        """Move elements from buffer to pending tasks."""
        while buffer and len(pending) < max_concurrent:
            key, idx, elem = buffer.pop(0)
            task = asyncio.create_task(process_with_key(key, idx, elem))
            pending.add(task)

    def yield_completed() -> Iterator[Out]:
        """Yield outputs for keys that have all results, in input order."""
        while key_order:
            key = key_order[0]

            if key not in results_by_key:
                break
            if len(results_by_key[key]) != expected_counts[key]:
                break

            results = results_by_key.pop(key)
            del expected_counts[key]
            key_order.pop(0)

            results.sort(key=lambda x: x[0])
            ordered_results = [r for _, r in results]
            yield combine(key, ordered_results)

    async def start_next_item_fetch() -> None:
        """Start fetching the next item from async iterator."""
        nonlocal next_item_task
        if items_exhausted or next_item_task is not None:
            return
        if len(buffer) + len(pending) >= max_concurrent:
            return

        async def fetch_next() -> In | None:
            return await items_aiter.__anext__()

        next_item_task = asyncio.create_task(fetch_next())

    # Initial fetch
    await start_next_item_fetch()
    schedule_from_buffer()

    # Yield any empty expansions that are already complete
    for out in yield_completed():
        yield out

    # Main loop - wait on both processing tasks and next-item fetch
    while pending or next_item_task is not None:
        # Build wait set
        wait_tasks: set[asyncio.Task[Any]] = set(pending)
        if next_item_task is not None:
            wait_tasks.add(next_item_task)

        if not wait_tasks:
            break

        done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

        # Handle completed next-item fetch
        if next_item_task is not None and next_item_task in done:
            done.discard(next_item_task)
            try:
                item = next_item_task.result()
                next_item_task = None

                # Skip heartbeats (None values)
                if item is not None:
                    expand_item(item)
            except StopAsyncIteration:
                items_exhausted = True
                next_item_task = None

        # Handle completed processing tasks
        for task in done:
            pending.discard(task)  # type: ignore[arg-type]
            key, idx, result = task.result()
            if key not in results_by_key:
                results_by_key[key] = []
            results_by_key[key].append((idx, result))

        # Yield completed outputs
        for out in yield_completed():
            yield out

        # Schedule more work
        schedule_from_buffer()
        await start_next_item_fetch()

    # Final yield
    for out in yield_completed():
        yield out


async def concurrent_expand_map[In, K, E, R, Out](
    items: Iterable[In] | AsyncIterable[In | None],
    expand: Callable[[In], tuple[K, list[E]] | None],
    process_one: Callable[[E], Coroutine[Any, Any, R]],
    combine: Callable[[K, list[R]], Out],
    max_concurrent: int,
) -> AsyncIterator[Out]:
    """Expand items into elements, process concurrently, and recombine.

    Async equivalent of batched_expand_map for fan-out/fan-in patterns.
    Instead of batching elements for GPU efficiency, processes elements
    concurrently with a semaphore limit.

    Pattern: fan-out → concurrent process → fan-in

    Example: Multiple-choice scoring where each question has N choices.
    Questions are expanded to N scoring requests, requests are processed
    concurrently (up to max_concurrent), then results are recombined
    per question.

    Accepts both sync and async iterables. For async iterables, heartbeats
    (None values) are skipped automatically.

    Args:
        items: Input items to process (sync or async iterable).
        expand: Fan-out function. Returns (key, elements) or None to skip.
            The key identifies this input for recombining results.
        process_one: Async function to process a single element.
        combine: Fan-in function. Takes key and all results for that input,
            returns the final output.
        max_concurrent: Maximum concurrent process_one calls.

    Yields:
        Outputs in input order, as soon as all elements for an input complete.
    """
    if isinstance(items, AsyncIterable):
        async for result in _concurrent_expand_map_async(
            items, expand, process_one, combine, max_concurrent  # type: ignore[arg-type]
        ):
            yield result
    else:
        async for result in _concurrent_expand_map_sync(
            items, expand, process_one, combine, max_concurrent  # type: ignore[arg-type]
        ):
            yield result


async def concurrent_map[T, R](
    items: Iterable[T] | AsyncIterable[T | None],
    process: Callable[[T], Coroutine[Any, Any, R]],
    max_concurrent: int,
) -> AsyncIterator[R]:
    """Process items concurrently, yielding results in input order.

    Maintains up to max_concurrent tasks in flight. Results are yielded
    in input order (waits for earlier items to complete before yielding
    later ones).

    Accepts both sync and async iterables. For async iterables, heartbeats
    (None values) are skipped automatically.

    This is a convenience wrapper around concurrent_expand_map for 1:1
    item-to-result mappings.

    Args:
        items: Items to process (sync or async iterable).
        process: Async function to apply to each item.
        max_concurrent: Maximum concurrent tasks.

    Yields:
        Results in input order.
    """
    item_id = 0

    def expand(item: T) -> tuple[int, list[T]]:
        nonlocal item_id
        key = item_id
        item_id += 1
        return (key, [item])

    def combine(key: int, results: list[R]) -> R:
        return results[0]

    async for result in concurrent_expand_map(
        items, expand, process, combine, max_concurrent
    ):
        yield result


# =============================================================================
# Abstract Base Classes
# =============================================================================


class AsyncBackend(ABC):
    """Base class for async generation backends (vLLM, sglang, etc.).

    These backends communicate with remote servers via HTTP and support
    concurrent requests. Scoring methods yield results as they complete
    (not necessarily in input order) - caller joins by key.
    """

    @property
    @abstractmethod
    def max_concurrent_requests(self) -> int:
        """Maximum number of concurrent requests this backend supports."""
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """Initialize backend (e.g., open HTTP session)."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup backend resources."""
        pass

    @abstractmethod
    def generate_chat(
        self,
        requests: Iterable[GenerateChatRequest]
        | AsyncIterable[GenerateChatRequest | None],
    ) -> AsyncIterator[GenerationResult | BackendError]:
        """Generate completions from chat messages, yielding as they complete.

        Results may arrive in any order. Use request_id to join with input.
        Each request contains its own generation params (n, max_tokens, etc.).

        On transient errors (bad request, timeout, server error), yields
        BackendError instead of killing the stream.

        Args:
            requests: Chat generation requests with keys and params. Can be sync
                iterable (batch mode) or async iterable (streaming mode with
                None heartbeats).

        Yields:
            GenerationResult with request_id matching request.key, or
            BackendError if the request failed.
        """
        pass

    @abstractmethod
    def generate_text(
        self,
        requests: Iterable[GenerateTextRequest]
        | AsyncIterable[GenerateTextRequest | None],
    ) -> AsyncIterator[GenerationResult | BackendError]:
        """Generate completions from text prompts, yielding as they complete.

        Results may arrive in any order. Use request_id to join with input.
        Each request contains its own generation params (n, max_tokens, etc.).

        On transient errors (bad request, timeout, server error), yields
        BackendError instead of killing the stream.

        Args:
            requests: Text generation requests with keys and params. Can be sync
                iterable (batch mode) or async iterable (streaming mode with
                None heartbeats).

        Yields:
            GenerationResult with request_id matching request.key, or
            BackendError if the request failed.
        """
        pass

    @abstractmethod
    def score_tokens(
        self,
        requests: Iterable[ScoreTokensRequest]
        | AsyncIterable[ScoreTokensRequest | None],
    ) -> AsyncIterator[ScoreTokensResult | BackendError]:
        """Score token sequences, yielding results as they complete.

        Results may arrive in any order. Use request.key to join with input.

        On transient errors, yields BackendError instead of killing the stream.

        Args:
            requests: Token scoring requests with keys. Can be sync or async.

        Yields:
            ScoreTokensResult with key and log_prob, or BackendError.
        """
        pass

    @abstractmethod
    def score_text(
        self,
        requests: Iterable[ScoreTextRequest] | AsyncIterable[ScoreTextRequest | None],
    ) -> AsyncIterator[ScoreTextResult | BackendError]:
        """Score text continuations (multiple-choice), yielding as complete.

        Results may arrive in any order. Use request.key to join with input.
        Each result contains LogProbResults for all choices in that request.

        On transient errors, yields BackendError instead of killing the stream.

        Args:
            requests: Text scoring requests with keys. Can be sync or async.

        Yields:
            ScoreTextResult with key and per-choice results, or BackendError.
        """
        pass

    @abstractmethod
    def score_chat(
        self,
        requests: Iterable[ScoreChatRequest] | AsyncIterable[ScoreChatRequest | None],
    ) -> AsyncIterator[ScoreChatResult | BackendError]:
        """Score chat completions (multiple-choice), yielding as complete.

        Results may arrive in any order. Use request.key to join with input.
        Each result contains LogProbResults for all choices in that request.

        On transient errors, yields BackendError instead of killing the stream.

        Args:
            requests: Chat scoring requests with keys. Can be sync or async.

        Yields:
            ScoreChatResult with key and per-choice results, or BackendError.
        """
        pass

    @abstractmethod
    async def set_placement(self, placement: Sequence[str]) -> None:
        """Set the mixer placement for each layer.

        Args:
            placement: Mixer to use at each layer (e.g., ["attention", "gdn", ...]).

        Raises:
            NotImplementedError: If backend doesn't support dynamic placement.
        """
        pass

    @property
    @abstractmethod
    def placement(self) -> list[str] | None:
        """Current mixer placement, or None if model doesn't support placements.

        For supernet models, returns the mixer name at each layer.
        For non-supernet models, returns None.
        """
        pass

    @property
    @abstractmethod
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names).

        Raises:
            NotImplementedError: If backend doesn't support model structure discovery.
        """
        pass


class SyncBackend(ABC):
    """Base class for sync generation backends (transformers, etc.).

    These backends run locally and process requests synchronously.
    Scoring methods yield results in input order as batches complete.
    Batching is handled internally - caller just iterates.
    """

    @abstractmethod
    def __enter__(self) -> Self:
        """Initialize backend (e.g., load model)."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup backend resources (e.g., free GPU memory)."""
        pass

    @abstractmethod
    def generate_chat(
        self,
        requests: Iterable[GenerateChatRequest | None],
    ) -> Iterator["GenerationResult"]:
        """Generate completions from chat messages, yielding as batches complete.

        Results are yielded in input order. Batching is handled internally.
        Requests with different generation params (n, max_tokens, temperature,
        top_p) trigger a batch flush - only requests with identical params
        are batched together.

        Streaming mode: None values are heartbeats for timeout-based flushing.

        Args:
            requests: Chat generation requests with keys and params. None values
                are heartbeats for timeout checking in streaming mode.

        Yields:
            GenerationResult with request_id matching request.key.
        """
        pass

    @abstractmethod
    def generate_text(
        self,
        requests: Iterable[GenerateTextRequest | None],
    ) -> Iterator["GenerationResult"]:
        """Generate completions from text prompts, yielding as batches complete.

        Results are yielded in input order. Batching is handled internally.
        Requests with different generation params (n, max_tokens, temperature,
        top_p) trigger a batch flush - only requests with identical params
        are batched together.

        Streaming mode: None values are heartbeats for timeout-based flushing.

        Args:
            requests: Text generation requests with keys and params. None values
                are heartbeats for timeout checking in streaming mode.

        Yields:
            GenerationResult with request_id matching request.key.
        """
        pass

    @abstractmethod
    def score_tokens(
        self,
        requests: Iterable[ScoreTokensRequest | None],
    ) -> Iterator[ScoreTokensResult]:
        """Score token sequences, yielding results as batches complete.

        Results are yielded in input order. Batching is handled internally.

        Streaming mode: When config.flush_timeout is set, requests may include
        None as heartbeats. Partial batches are flushed after flush_timeout.

        Args:
            requests: Token scoring requests with keys. None values are
                heartbeats for timeout checking in streaming mode.

        Yields:
            ScoreTokensResult with key and log_prob.
        """
        pass

    @abstractmethod
    def score_text(
        self,
        requests: Iterable[ScoreTextRequest | None],
    ) -> Iterator[ScoreTextResult]:
        """Score text continuations (multiple-choice), yielding as complete.

        Results are yielded in input order. Each result contains
        LogProbResults for all choices in that request.

        Streaming mode: When config.flush_timeout is set, requests may include
        None as heartbeats. Partial batches are flushed after flush_timeout.

        Args:
            requests: Text scoring requests with keys. None values are
                heartbeats for timeout checking in streaming mode.

        Yields:
            ScoreTextResult with key and per-choice results.
        """
        pass

    @abstractmethod
    def score_chat(
        self,
        requests: Iterable[ScoreChatRequest | None],
    ) -> Iterator[ScoreChatResult]:
        """Score chat completions (multiple-choice), yielding as complete.

        Results are yielded in input order. Each result contains
        LogProbResults for all choices in that request.

        Streaming mode: When config.flush_timeout is set, requests may include
        None as heartbeats. Partial batches are flushed after flush_timeout.

        Args:
            requests: Chat scoring requests with keys. None values are
                heartbeats for timeout checking in streaming mode.

        Yields:
            ScoreChatResult with key and per-choice results.
        """
        pass

    @abstractmethod
    def set_placement(self, placement: Sequence[str]) -> None:
        """Set the mixer placement for each layer."""
        pass

    @property
    @abstractmethod
    def placement(self) -> list[str] | None:
        """Current mixer placement, or None if model doesn't support placements."""
        pass

    @property
    @abstractmethod
    def model_structure(self) -> ModelStructure:
        """Model structure (num_layers, mixer_names)."""
        pass


# =============================================================================
# Config Base
# =============================================================================


class BaseBackendConfig(BaseModel):
    """Base configuration for generation backends."""

    model_config = {"frozen": True, "extra": "forbid"}

    backend_type: str = Field(description="Type identifier for backend")

    _backend_class: ClassVar[type[AsyncBackend] | type[SyncBackend]]

    def create_backend(self) -> AsyncBackend | SyncBackend:
        """Smart constructor: instantiate backend from config."""
        return self._backend_class(config=self)


def get_model_path(config: BaseBackendConfig) -> str:
    """Extract model path from backend config.

    Uses pattern matching on concrete config types.

    Args:
        config: Backend configuration.

    Returns:
        Model path string, or "rest-backend" for REST backends,
        or "unknown" for unrecognized config types.
    """
    # Lazy imports to avoid circular dependencies
    from place_layers.backend.rest import RESTBackendConfig, RESTDNSBackendConfig
    from place_layers.backend.transformers import TransformersBackendConfig
    from place_layers.backend.vllm import VLLMBackendConfig
    from place_layers.clients.local_vllm import LocalVLLMClientConfig
    from place_layers.clients.remote_vllm import RemoteVLLMClientConfig

    match config:
        case TransformersBackendConfig(model_path=model_path):
            return str(model_path)
        case VLLMBackendConfig(client=LocalVLLMClientConfig(model_path=model_path)):
            return str(model_path)
        case VLLMBackendConfig(client=RemoteVLLMClientConfig(model=model)):
            return model
        case RESTBackendConfig() | RESTDNSBackendConfig():
            return "rest-backend"
        case _:
            return "unknown"
