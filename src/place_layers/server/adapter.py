"""Backend adapters for unified async single-request interface.

Provides a common interface for both sync and async backends:
- AsyncBackendAdapter: Thin wrapper, forwards to backend directly
- SyncBackendAdapter: Streaming bridge - converts async requests to sync iterator

The server uses these adapters transparently without knowing the backend type.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

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
# SyncBackendAdapter - Streaming bridge from async to sync
# =============================================================================


class _StreamingBridge[Req, Res]:
    """Bridge async requests to a sync iterator with heartbeat-based flushing.

    Converts an infinite stream of async requests into a sync iterator that
    the backend can consume. The backend handles batching internally with
    timeout-based flushing triggered by None heartbeats.

    Pattern:
    1. Async side submits requests via submit()
    2. Worker thread runs backend method on the iterator from __iter__()
    3. Results stream back via result_queue
    4. Async side reads results and resolves futures
    """

    # Sentinel to signal shutdown
    _SHUTDOWN = object()

    def __init__(self, heartbeat_interval: float = 0.01) -> None:
        self._heartbeat_interval = heartbeat_interval

        # Request queue: (request, future) or _SHUTDOWN sentinel
        self._request_queue: queue.Queue[
            tuple[Req, asyncio.Future[Res]] | object
        ] = queue.Queue()

        # Result queue: (future, result | exception)
        self._result_queue: queue.Queue[
            tuple[asyncio.Future[Res], Res | BaseException]
        ] = queue.Queue()

        # Track pending futures in order
        self._pending_futures: list[asyncio.Future[Res]] = []
        self._lock = threading.Lock()

        # Event loop for resolving futures from worker thread
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for resolving futures."""
        self._loop = loop

    def submit(self, request: Req, future: asyncio.Future[Res]) -> None:
        """Submit a request from the async side."""
        with self._lock:
            self._pending_futures.append(future)
        self._request_queue.put((request, future))

    def shutdown(self) -> None:
        """Signal the worker to stop."""
        self._request_queue.put(self._SHUTDOWN)

    def __iter__(self) -> Iterator[Req | None]:
        """Iterate over requests, yielding None as heartbeats on timeout."""
        while True:
            try:
                item = self._request_queue.get(timeout=self._heartbeat_interval)

                if item is self._SHUTDOWN:
                    return

                request, _ = item  # type: ignore[misc]
                yield request

            except queue.Empty:
                # Timeout - yield heartbeat for flush checking
                yield None

    def deliver_result(self, result: Res) -> None:
        """Deliver a result from the worker thread (in input order)."""
        with self._lock:
            if not self._pending_futures:
                logger.error("Result delivered but no pending futures")
                return
            future = self._pending_futures.pop(0)

        # Resolve future from worker thread via event loop
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                lambda f=future, r=result: (
                    f.set_result(r) if not f.done() else None
                )
            )

    def deliver_exception(self, exc: BaseException) -> None:
        """Deliver an exception to all pending futures."""
        with self._lock:
            futures = list(self._pending_futures)
            self._pending_futures.clear()

        if self._loop is not None:
            for future in futures:
                self._loop.call_soon_threadsafe(
                    lambda f=future, e=exc: (
                        f.set_exception(e) if not f.done() else None
                    )
                )


@dataclass
class _GenerateItem:
    """Queued generation request."""

    request: "GenerateRequest"
    params: GenerationParams
    future: asyncio.Future["GenerationResult"]


class SyncBackendAdapter(BackendAdapter):
    """Adapter for SyncBackend - streaming bridge from async to sync.

    Uses _StreamingBridge to convert async requests into a sync iterator.
    The backend handles all batching internally with timeout-based flushing.
    No redundant batching in the adapter.
    """

    def __init__(
        self,
        backend: "SyncBackend",
        batch_size: int = 32,  # Unused - backend handles batching
        batch_timeout: float = 0.01,
    ) -> None:
        self._backend = backend
        self._batch_timeout = batch_timeout

        # Streaming bridges for each scoring operation
        self._score_tokens_bridge: _StreamingBridge[
            "ScoreTokensRequest", "ScoreTokensResult"
        ] | None = None
        self._score_text_bridge: _StreamingBridge[
            "ScoreTextRequest", "ScoreTextResult"
        ] | None = None
        self._score_chat_bridge: _StreamingBridge[
            "ScoreChatRequest", "ScoreChatResult"
        ] | None = None

        # Generate still uses queue-based approach (needs param grouping)
        self._generate_queue: asyncio.Queue[_GenerateItem | None] = asyncio.Queue()

        # Worker threads
        self._workers: list[threading.Thread] = []
        self._generate_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> Self:
        # Enter sync backend in thread pool
        await asyncio.to_thread(self._backend.__enter__)

        self._loop = asyncio.get_running_loop()

        # Create streaming bridges
        self._score_tokens_bridge = _StreamingBridge(self._batch_timeout)
        self._score_tokens_bridge.set_loop(self._loop)

        self._score_text_bridge = _StreamingBridge(self._batch_timeout)
        self._score_text_bridge.set_loop(self._loop)

        self._score_chat_bridge = _StreamingBridge(self._batch_timeout)
        self._score_chat_bridge.set_loop(self._loop)

        # Start worker threads for scoring (streaming)
        self._workers = [
            threading.Thread(
                target=self._score_tokens_worker,
                daemon=True,
                name="score_tokens_worker",
            ),
            threading.Thread(
                target=self._score_text_worker,
                daemon=True,
                name="score_text_worker",
            ),
            threading.Thread(
                target=self._score_chat_worker,
                daemon=True,
                name="score_chat_worker",
            ),
        ]
        for worker in self._workers:
            worker.start()

        # Start generate worker as async task (still uses queue-based approach)
        self._generate_task = asyncio.create_task(self._generate_worker())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Signal workers to stop
        if self._score_tokens_bridge:
            self._score_tokens_bridge.shutdown()
        if self._score_text_bridge:
            self._score_text_bridge.shutdown()
        if self._score_chat_bridge:
            self._score_chat_bridge.shutdown()

        await self._generate_queue.put(None)

        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)

        if self._generate_task:
            await self._generate_task

        # Exit sync backend in thread pool
        await asyncio.to_thread(self._backend.__exit__, None, None, None)

    # =========================================================================
    # Public methods
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
        assert self._loop is not None and self._score_tokens_bridge is not None
        future: asyncio.Future["ScoreTokensResult"] = self._loop.create_future()
        self._score_tokens_bridge.submit(request, future)
        return await future

    async def score_text_one(
        self,
        request: "ScoreTextRequest",
    ) -> "ScoreTextResult":
        assert self._loop is not None and self._score_text_bridge is not None
        future: asyncio.Future["ScoreTextResult"] = self._loop.create_future()
        self._score_text_bridge.submit(request, future)
        return await future

    async def score_chat_one(
        self,
        request: "ScoreChatRequest",
    ) -> "ScoreChatResult":
        assert self._loop is not None and self._score_chat_bridge is not None
        future: asyncio.Future["ScoreChatResult"] = self._loop.create_future()
        self._score_chat_bridge.submit(request, future)
        return await future

    # =========================================================================
    # Workers
    # =========================================================================

    def _score_tokens_worker(self) -> None:
        """Worker thread: consume requests from bridge, deliver results."""
        assert self._score_tokens_bridge is not None
        try:
            for result in self._backend.score_tokens(self._score_tokens_bridge):
                self._score_tokens_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("Token scoring worker failed")
            self._score_tokens_bridge.deliver_exception(e)

    def _score_text_worker(self) -> None:
        """Worker thread: consume requests from bridge, deliver results."""
        assert self._score_text_bridge is not None
        try:
            for result in self._backend.score_text(self._score_text_bridge):
                self._score_text_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("Text scoring worker failed")
            self._score_text_bridge.deliver_exception(e)

    def _score_chat_worker(self) -> None:
        """Worker thread: consume requests from bridge, deliver results."""
        assert self._score_chat_bridge is not None
        try:
            for result in self._backend.score_chat(self._score_chat_bridge):
                self._score_chat_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("Chat scoring worker failed")
            self._score_chat_bridge.deliver_exception(e)

    async def _generate_worker(self) -> None:
        """Process generation requests with param grouping.

        Generation still uses queue-based batching because requests need
        to be grouped by params before being sent to the backend.
        """
        while True:
            batch = await self._collect_generate_batch()
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

    async def _collect_generate_batch(self) -> list[_GenerateItem] | None:
        """Collect generation requests with timeout."""
        batch: list[_GenerateItem] = []

        # Wait for first item
        item = await self._generate_queue.get()
        if item is None:
            return None
        batch.append(item)

        # Collect more with timeout
        deadline = time.monotonic() + self._batch_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    self._generate_queue.get(), timeout=remaining
                )
                if item is None:
                    await self._generate_queue.put(None)
                    break
                batch.append(item)
            except asyncio.TimeoutError:
                break

        return batch


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
