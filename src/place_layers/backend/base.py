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
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Self, Sequence

from pydantic import BaseModel, Field

from place_layers.backend.types import (
    ChatMessage,
    ChatScoringRequest,
    ChatScoringResult,
    GenerationRequest,
    ModelStructure,
    TextScoringRequest,
    TextScoringResult,
    TokenScoringRequest,
    TokenScoringResult,
)

if TYPE_CHECKING:
    from place_layers.clients.types import GenerationResult


# =============================================================================
# Concurrency/Batching Helpers
# =============================================================================


def batched_map[T, R](
    items: Iterable[T],
    process: Callable[[list[T]], Iterable[R]],
    batch_size: int,
) -> Iterator[R]:
    """Process items in batches, yielding results as batches complete.

    Accumulates items up to batch_size, calls process on the batch,
    and yields all results. Handles the final partial batch.

    This is a special case of batched_expand_map where each item expands
    to exactly one element.

    Args:
        items: Items to process.
        process: Function that takes a batch and returns results (1:1 with input).
        batch_size: Maximum items per batch.

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

    yield from batched_expand_map(items, expand, process_batch, combine, batch_size)


def batched_expand_map[In, K, E, R, Out](
    items: Iterable[In],
    expand: Callable[[In], tuple[K, list[E]] | None],
    process_batch: Callable[[list[E]], list[R]],
    combine: Callable[[K, list[R]], Out],
    batch_size: int,
) -> Iterator[Out]:
    """Expand items into elements, batch elements, process, and recombine.

    This handles the case where one input expands to multiple elements that
    may span batch boundaries. Elements are batched at fixed size for
    efficient processing, then results are recombined by input.

    Pattern: fan-out → batch → process → fan-in

    Example: Multiple-choice scoring where each question has N choices.
    Questions are expanded to N sequences, sequences are batched for GPU
    efficiency, then results are recombined per question.

    Args:
        items: Input items to process.
        expand: Fan-out function. Returns (key, elements) or None to skip.
            The key identifies this input for recombining results.
        process_batch: Processes a batch of elements, returns results 1:1.
        combine: Fan-in function. Takes key and all results for that input,
            returns the final output.
        batch_size: Fixed batch size for element processing.

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

    def flush_batches() -> Iterator[Out]:
        """Process complete batches and yield any completed outputs."""
        nonlocal pending

        while len(pending) >= batch_size:
            # Extract batch
            batch = pending[:batch_size]
            pending = pending[batch_size:]

            # Process elements
            elements = [elem for _, _, elem in batch]
            batch_results = process_batch(elements)

            # Store results by key
            for (key, idx, _), result in zip(batch, batch_results):
                if key not in results_by_key:
                    results_by_key[key] = []
                results_by_key[key].append((idx, result))

            # Yield any completed outputs (in order)
            yield from yield_completed()

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
        yield from flush_batches()

    # Drain remaining elements (final partial batch)
    if pending:
        elements = [elem for _, _, elem in pending]
        batch_results = process_batch(elements)

        for (key, idx, _), result in zip(pending, batch_results):
            if key not in results_by_key:
                results_by_key[key] = []
            results_by_key[key].append((idx, result))

        pending = []

    # Yield any remaining completed outputs
    yield from yield_completed()


async def concurrent_map[T, R](
    items: Iterable[T],
    process: Callable[[T], Coroutine[Any, Any, R]],
    max_concurrent: int,
) -> AsyncIterator[R]:
    """Process items concurrently, yielding results as they complete.

    Maintains up to max_concurrent tasks in flight. As each completes,
    yields its result and schedules the next item. Results are yielded
    in completion order, not input order.

    Args:
        items: Items to process.
        process: Async function to apply to each item.
        max_concurrent: Maximum concurrent tasks.

    Yields:
        Results in completion order.
    """
    pending: set[asyncio.Task[R]] = set()
    items_iter = iter(items)
    done_iterating = False

    # Fill up to max_concurrent
    for item in items_iter:
        pending.add(asyncio.create_task(process(item)))
        if len(pending) >= max_concurrent:
            break
    else:
        done_iterating = True

    # Process until all done
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            yield task.result()

        # Refill from remaining items
        if not done_iterating:
            for item in items_iter:
                pending.add(asyncio.create_task(process(item)))
                if len(pending) >= max_concurrent:
                    break
            else:
                done_iterating = True


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
    def generate(
        self,
        requests: Iterable[GenerationRequest],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator["GenerationResult"]:
        """Generate completions, yielding results as they complete.

        Results may arrive in any order. Use request_id to join with input.

        Args:
            requests: Generation requests with keys.
            n: Number of completions per request.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Yields:
            GenerationResult with request_id matching request.key.
        """
        pass

    @abstractmethod
    def score_tokens(
        self,
        requests: Iterable[TokenScoringRequest],
    ) -> AsyncIterator[TokenScoringResult]:
        """Score token sequences, yielding results as they complete.

        Results may arrive in any order. Use request.key to join with input.

        Args:
            requests: Token scoring requests with keys.

        Yields:
            TokenScoringResult with key and log_prob.
        """
        pass

    @abstractmethod
    def score_text(
        self,
        requests: Iterable[TextScoringRequest],
    ) -> AsyncIterator[TextScoringResult]:
        """Score text continuations (multiple-choice), yielding as complete.

        Results may arrive in any order. Use request.key to join with input.
        Each result contains LogProbResults for all choices in that request.

        Args:
            requests: Text scoring requests with keys.

        Yields:
            TextScoringResult with key and per-choice results.
        """
        pass

    @abstractmethod
    def score_chat(
        self,
        requests: Iterable[ChatScoringRequest],
    ) -> AsyncIterator[ChatScoringResult]:
        """Score chat completions (multiple-choice), yielding as complete.

        Results may arrive in any order. Use request.key to join with input.
        Each result contains LogProbResults for all choices in that request.

        Args:
            requests: Chat scoring requests with keys.

        Yields:
            ChatScoringResult with key and per-choice results.
        """
        pass

    @abstractmethod
    async def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set the mixer placement for each layer.

        Args:
            mixer_names: Mixer name for each layer (e.g., ["attention", "mamba", ...]).

        Raises:
            NotImplementedError: If backend doesn't support dynamic placement.
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
    def generate(
        self,
        requests: Iterable[GenerationRequest],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterator["GenerationResult"]:
        """Generate completions, yielding results as batches complete.

        Results are yielded in input order. Batching is handled internally.

        Args:
            requests: Generation requests with keys.
            n: Number of completions per request.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Yields:
            GenerationResult with request_id matching request.key.
        """
        pass

    @abstractmethod
    def score_tokens(
        self,
        requests: Iterable[TokenScoringRequest],
    ) -> Iterator[TokenScoringResult]:
        """Score token sequences, yielding results as batches complete.

        Results are yielded in input order. Batching is handled internally.

        Args:
            requests: Token scoring requests with keys.

        Yields:
            TokenScoringResult with key and log_prob.
        """
        pass

    @abstractmethod
    def score_text(
        self,
        requests: Iterable[TextScoringRequest],
    ) -> Iterator[TextScoringResult]:
        """Score text continuations (multiple-choice), yielding as complete.

        Results are yielded in input order. Each result contains
        LogProbResults for all choices in that request.

        Args:
            requests: Text scoring requests with keys.

        Yields:
            TextScoringResult with key and per-choice results.
        """
        pass

    @abstractmethod
    def score_chat(
        self,
        requests: Iterable[ChatScoringRequest],
    ) -> Iterator[ChatScoringResult]:
        """Score chat completions (multiple-choice), yielding as complete.

        Results are yielded in input order. Each result contains
        LogProbResults for all choices in that request.

        Args:
            requests: Chat scoring requests with keys.

        Yields:
            ChatScoringResult with key and per-choice results.
        """
        pass

    @abstractmethod
    def set_placement(self, mixer_names: Sequence[str]) -> None:
        """Set the mixer placement for each layer."""
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
