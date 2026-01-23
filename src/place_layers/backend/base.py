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
    ModelStructure,
    TextScoringRequest,
    TextScoringResult,
    TokenScoringRequest,
    TokenScoringResult,
)

if TYPE_CHECKING:
    from place_layers.clients.types import GenerationResult


# =============================================================================
# Async Concurrency Helper
# =============================================================================


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
    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str = "",
    ) -> "GenerationResult":
        """Generate n completions asynchronously.

        Args:
            messages: Chat messages (list of role/content dicts).
            n: Number of completions to generate.
            max_tokens: Maximum tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            request_id: Optional ID for tracking.

        Returns:
            GenerationResult with samples and prompt token IDs.
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
    def batch_generate(
        self,
        messages_batch: Sequence[list[ChatMessage]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list["GenerationResult"]:
        """Generate n completions for each message sequence in batch."""
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
