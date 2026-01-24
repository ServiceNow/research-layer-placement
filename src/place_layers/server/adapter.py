"""Backend adapters for unified async single-request interface.

Provides a common interface for both sync and async backends:
- AsyncBackendAdapter: Thin wrapper, forwards to backend directly
- SyncBackendAdapter: Queue-based batching for GPU efficiency

The server uses these adapters transparently without knowing the backend type.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    from place_layers.backend.base import AsyncBackend, SyncBackend
    from place_layers.backend.types import (
        GenerateRequest,
        ScoreChatRequest,
        ScoreChatResult,
        ScoreTextRequest,
        ScoreTextResult,
        ScoreTokensRequest,
        ScoreTokensResult,
    )
    from place_layers.clients.types import GenerationResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationParams:
    """Generation parameters (batch-level, not per-request)."""

    n: int
    max_tokens: int
    temperature: float
    top_p: float


class BackendAdapter(ABC):
    """Unified async single-request interface for any backend type."""

    @abstractmethod
    async def __aenter__(self) -> Self:
        """Initialize adapter and backend."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup adapter and backend."""
        pass

    @abstractmethod
    async def generate_one(
        self,
        request: "GenerateRequest",
        params: GenerationParams,
    ) -> "GenerationResult":
        """Generate completions for a single request."""
        pass

    @abstractmethod
    async def score_tokens_one(
        self,
        request: "ScoreTokensRequest",
    ) -> "ScoreTokensResult":
        """Score a single token sequence."""
        pass

    @abstractmethod
    async def score_text_one(
        self,
        request: "ScoreTextRequest",
    ) -> "ScoreTextResult":
        """Score text continuations for a single request."""
        pass

    @abstractmethod
    async def score_chat_one(
        self,
        request: "ScoreChatRequest",
    ) -> "ScoreChatResult":
        """Score chat completions for a single request."""
        pass


class AsyncBackendAdapter(BackendAdapter):
    """Adapter for AsyncBackend - thin wrapper, forwards directly."""

    def __init__(self, backend: "AsyncBackend") -> None:
        self._backend = backend

    async def __aenter__(self) -> Self:
        await self._backend.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._backend.__aexit__(exc_type, exc_val, exc_tb)

    async def generate_one(
        self,
        request: "GenerateRequest",
        params: GenerationParams,
    ) -> "GenerationResult":
        async for result in self._backend.generate(
            [request],
            n=params.n,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
        ):
            return result
        raise RuntimeError("Backend returned no results")

    async def score_tokens_one(
        self,
        request: "ScoreTokensRequest",
    ) -> "ScoreTokensResult":
        async for result in self._backend.score_tokens([request]):
            return result
        raise RuntimeError("Backend returned no results")

    async def score_text_one(
        self,
        request: "ScoreTextRequest",
    ) -> "ScoreTextResult":
        async for result in self._backend.score_text([request]):
            return result
        raise RuntimeError("Backend returned no results")

    async def score_chat_one(
        self,
        request: "ScoreChatRequest",
    ) -> "ScoreChatResult":
        async for result in self._backend.score_chat([request]):
            return result
        raise RuntimeError("Backend returned no results")


# =============================================================================
# SyncBackendAdapter - Queue-based batching
# =============================================================================


@dataclass
class _GenerateItem:
    """Queued generation request."""

    request: "GenerateRequest"
    params: GenerationParams
    future: asyncio.Future["GenerationResult"]


@dataclass
class _ScoreTokensItem:
    """Queued token scoring request."""

    request: "ScoreTokensRequest"
    future: asyncio.Future["ScoreTokensResult"]


@dataclass
class _ScoreTextItem:
    """Queued text scoring request."""

    request: "ScoreTextRequest"
    future: asyncio.Future["ScoreTextResult"]


@dataclass
class _ScoreChatItem:
    """Queued chat scoring request."""

    request: "ScoreChatRequest"
    future: asyncio.Future["ScoreChatResult"]


class SyncBackendAdapter(BackendAdapter):
    """Adapter for SyncBackend - queue-based batching for GPU efficiency.

    Incoming requests are queued. Background workers collect batches
    and process them in a thread pool. Results are delivered via futures.

    For generate: requests are grouped by params before batching.
    For scoring: requests are batched directly (no params).
    """

    def __init__(
        self,
        backend: "SyncBackend",
        batch_size: int = 32,
        batch_timeout: float = 0.01,  # 10ms
    ) -> None:
        self._backend = backend
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout

        # Queues for each operation type
        self._generate_queue: asyncio.Queue[_GenerateItem | None] = asyncio.Queue()
        self._score_tokens_queue: asyncio.Queue[_ScoreTokensItem | None] = (
            asyncio.Queue()
        )
        self._score_text_queue: asyncio.Queue[_ScoreTextItem | None] = asyncio.Queue()
        self._score_chat_queue: asyncio.Queue[_ScoreChatItem | None] = asyncio.Queue()

        # Worker tasks
        self._workers: list[asyncio.Task] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> Self:
        # Enter sync backend in thread pool
        await asyncio.to_thread(self._backend.__enter__)

        self._loop = asyncio.get_running_loop()

        # Start worker tasks
        self._workers = [
            asyncio.create_task(self._generate_worker()),
            asyncio.create_task(self._score_tokens_worker()),
            asyncio.create_task(self._score_text_worker()),
            asyncio.create_task(self._score_chat_worker()),
        ]

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Signal workers to stop
        await self._generate_queue.put(None)
        await self._score_tokens_queue.put(None)
        await self._score_text_queue.put(None)
        await self._score_chat_queue.put(None)

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)

        # Exit sync backend in thread pool
        await asyncio.to_thread(self._backend.__exit__, None, None, None)

    # =========================================================================
    # Public methods - queue requests and wait for results
    # =========================================================================

    async def generate_one(
        self,
        request: "GenerateRequest",
        params: GenerationParams,
    ) -> "GenerationResult":
        assert self._loop is not None
        future: asyncio.Future["GenerationResult"] = self._loop.create_future()
        await self._generate_queue.put(_GenerateItem(request, params, future))
        return await future

    async def score_tokens_one(
        self,
        request: "ScoreTokensRequest",
    ) -> "ScoreTokensResult":
        assert self._loop is not None
        future: asyncio.Future["ScoreTokensResult"] = self._loop.create_future()
        await self._score_tokens_queue.put(_ScoreTokensItem(request, future))
        return await future

    async def score_text_one(
        self,
        request: "ScoreTextRequest",
    ) -> "ScoreTextResult":
        assert self._loop is not None
        future: asyncio.Future["ScoreTextResult"] = self._loop.create_future()
        await self._score_text_queue.put(_ScoreTextItem(request, future))
        return await future

    async def score_chat_one(
        self,
        request: "ScoreChatRequest",
    ) -> "ScoreChatResult":
        assert self._loop is not None
        future: asyncio.Future["ScoreChatResult"] = self._loop.create_future()
        await self._score_chat_queue.put(_ScoreChatItem(request, future))
        return await future

    # =========================================================================
    # Workers - collect batches and process
    # =========================================================================

    async def _collect_batch[T](
        self,
        queue: asyncio.Queue[T | None],
    ) -> list[T] | None:
        """Collect up to batch_size items from queue with timeout.

        Returns None if shutdown sentinel received.
        """
        batch: list[T] = []

        # Wait for first item (blocking)
        item = await queue.get()
        if item is None:
            return None
        batch.append(item)

        # Collect more items up to batch_size or timeout
        deadline = asyncio.get_event_loop().time() + self._batch_timeout
        while len(batch) < self._batch_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=remaining)
                if item is None:
                    # Shutdown - process what we have, then signal done
                    # Put sentinel back for next iteration
                    await queue.put(None)
                    break
                batch.append(item)
            except asyncio.TimeoutError:
                break

        return batch

    async def _generate_worker(self) -> None:
        """Process generation requests with param grouping."""
        while True:
            batch = await self._collect_batch(self._generate_queue)
            if batch is None:
                break

            # Group by params
            by_params: dict[GenerationParams, list[_GenerateItem]] = {}
            for item in batch:
                by_params.setdefault(item.params, []).append(item)

            # Process each param group
            for params, group in by_params.items():
                requests = [item.request for item in group]
                futures = [item.future for item in group]

                try:
                    results = await asyncio.to_thread(
                        lambda r=requests, p=params: list(
                            self._backend.generate(
                                r,
                                n=p.n,
                                max_tokens=p.max_tokens,
                                temperature=p.temperature,
                                top_p=p.top_p,
                            )
                        )
                    )

                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)

                except Exception as e:
                    logger.exception("Generation batch failed")
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)

    async def _score_tokens_worker(self) -> None:
        """Process token scoring requests."""
        while True:
            batch = await self._collect_batch(self._score_tokens_queue)
            if batch is None:
                break

            requests = [item.request for item in batch]
            futures = [item.future for item in batch]

            try:
                results = await asyncio.to_thread(
                    lambda r=requests: list(self._backend.score_tokens(r))
                )

                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)

            except Exception as e:
                logger.exception("Token scoring batch failed")
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

    async def _score_text_worker(self) -> None:
        """Process text scoring requests."""
        while True:
            batch = await self._collect_batch(self._score_text_queue)
            if batch is None:
                break

            requests = [item.request for item in batch]
            futures = [item.future for item in batch]

            try:
                results = await asyncio.to_thread(
                    lambda r=requests: list(self._backend.score_text(r))
                )

                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)

            except Exception as e:
                logger.exception("Text scoring batch failed")
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

    async def _score_chat_worker(self) -> None:
        """Process chat scoring requests."""
        while True:
            batch = await self._collect_batch(self._score_chat_queue)
            if batch is None:
                break

            requests = [item.request for item in batch]
            futures = [item.future for item in batch]

            try:
                results = await asyncio.to_thread(
                    lambda r=requests: list(self._backend.score_chat(r))
                )

                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)

            except Exception as e:
                logger.exception("Chat scoring batch failed")
                for future in futures:
                    if not future.done():
                        future.set_exception(e)


def create_adapter(
    backend: "AsyncBackend | SyncBackend",
    batch_size: int = 32,
    batch_timeout: float = 0.01,
) -> BackendAdapter:
    """Create appropriate adapter for backend type."""
    from place_layers.backend.base import AsyncBackend, SyncBackend

    if isinstance(backend, SyncBackend):
        return SyncBackendAdapter(backend, batch_size, batch_timeout)
    elif isinstance(backend, AsyncBackend):
        return AsyncBackendAdapter(backend)
    else:
        raise TypeError(f"Unknown backend type: {type(backend)}")
