"""Backend adapters for unified async single-request interface.

Provides a common interface for both sync and async backends:
- AsyncBackendAdapter: Thin wrapper, forwards to backend directly
- SyncBackendAdapter: Streaming bridge - async requests flow to sync iterator

The server uses these adapters transparently without knowing the backend type.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Self, assert_never

from place_layers.backend.types import BackendError

if TYPE_CHECKING:
    from place_layers.backend.base import AsyncBackend, SyncBackend
    from place_layers.backend.types import (
        EchoTokenLogprobsRequest,
        EchoTokenLogprobsResult,
        GenerateChatRequest,
        GenerateTextRequest,
        ScoreChatRequest,
        ScoreChatResult,
        ScoreTextRequest,
        ScoreTextResult,
        ScoreTokensRequest,
        ScoreTokensResult,
    )
    from place_layers.clients.types import GenerationResult


class BackendRequestError(Exception):
    """Raised when a backend request fails with a BackendError."""

    def __init__(self, error: BackendError) -> None:
        self.error = error
        super().__init__(f"{error.error_type}: {error.message}")

logger = logging.getLogger(__name__)


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
    async def generate_chat_one(
        self,
        request: "GenerateChatRequest",
    ) -> "GenerationResult":
        """Generate completions for a single chat request."""
        pass

    @abstractmethod
    async def generate_text_one(
        self,
        request: "GenerateTextRequest",
    ) -> "GenerationResult":
        """Generate completions for a single text request."""
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

    @abstractmethod
    async def echo_token_logprobs_one(
        self,
        request: "EchoTokenLogprobsRequest",
    ) -> "EchoTokenLogprobsResult":
        """Compute per-token logprobs for a single request."""
        pass


# =============================================================================
# Async Streaming Bridge - connects async callers to async backend streams
# =============================================================================


class _AsyncStreamingBridge[Req, Res]:
    """Bridge async single-request calls to an async streaming backend.

    All operations happen in the same event loop (no threads needed):
    - Request queue: asyncio.Queue for incoming requests
    - Future queue: asyncio.Queue to track futures in FIFO order

    Flow:
    1. Async caller submits request via submit(), gets future back
    2. Background task iterates via __aiter__(), gets requests
    3. Background task feeds requests to backend, delivers results
    4. Results are set on futures directly (same event loop)

    Heartbeats (None values) are yielded on timeout to allow the backend's
    concurrent_map_async to process items even when no requests arrive.
    """

    _SHUTDOWN = object()

    def __init__(self, heartbeat_interval: float) -> None:
        self._heartbeat_interval = heartbeat_interval
        self._request_queue: asyncio.Queue[Req | object] = asyncio.Queue()
        self._future_queue: asyncio.Queue[asyncio.Future[Res]] = asyncio.Queue()
        self._shutdown = False

    async def submit(self, request: Req) -> Res:
        """Submit a request and wait for result."""
        future: asyncio.Future[Res] = asyncio.get_running_loop().create_future()
        # Future first, then request - ensures future exists before result arrives
        await self._future_queue.put(future)
        await self._request_queue.put(request)
        return await future

    def shutdown(self) -> None:
        """Signal the iterator to stop (non-blocking)."""
        self._shutdown = True
        # Use put_nowait since we might be called from sync context during cleanup
        try:
            self._request_queue.put_nowait(self._SHUTDOWN)
        except asyncio.QueueFull:
            pass  # Queue will be drained anyway

    async def __aiter__(self) -> AsyncIterator[Req | None]:
        """Yield requests to the async consumer, with heartbeats on timeout."""
        while not self._shutdown:
            try:
                item = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=self._heartbeat_interval,
                )
                if item is self._SHUTDOWN:
                    return
                yield item  # type: ignore[misc]
            except asyncio.TimeoutError:
                yield None  # Heartbeat

    async def deliver_result(self, result: Res) -> None:
        """Deliver a result to the next pending future."""
        try:
            future = self._future_queue.get_nowait()
        except asyncio.QueueEmpty:
            logger.error("Result delivered but no pending future")
            return

        if not future.done():
            future.set_result(result)

    async def deliver_exception(self, exc: BaseException) -> None:
        """Deliver exception to all pending futures."""
        futures: list[asyncio.Future[Res]] = []
        while True:
            try:
                futures.append(self._future_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        for future in futures:
            if not future.done():
                future.set_exception(exc)


class AsyncBackendAdapter(BackendAdapter):
    """Adapter for AsyncBackend using async streaming bridges.

    All operations use streaming bridges where requests flow continuously
    to the backend. The backend handles concurrency internally via
    concurrent_map_async with max_concurrent limit.
    """

    def __init__(
        self,
        backend: "AsyncBackend",
        heartbeat_interval: float = 0.01,
    ) -> None:
        self._backend = backend
        self._heartbeat_interval = heartbeat_interval

        # Streaming bridges for all operations
        self._generate_chat_bridge: _AsyncStreamingBridge[
            "GenerateChatRequest", "GenerationResult | BackendError"
        ] | None = None
        self._generate_text_bridge: _AsyncStreamingBridge[
            "GenerateTextRequest", "GenerationResult | BackendError"
        ] | None = None
        self._score_tokens_bridge: _AsyncStreamingBridge[
            "ScoreTokensRequest", "ScoreTokensResult | BackendError"
        ] | None = None
        self._score_text_bridge: _AsyncStreamingBridge[
            "ScoreTextRequest", "ScoreTextResult | BackendError"
        ] | None = None
        self._score_chat_bridge: _AsyncStreamingBridge[
            "ScoreChatRequest", "ScoreChatResult | BackendError"
        ] | None = None
        self._echo_logprobs_bridge: _AsyncStreamingBridge[
            "EchoTokenLogprobsRequest", "EchoTokenLogprobsResult | BackendError"
        ] | None = None

        self._tasks: list[asyncio.Task[None]] = []

    async def __aenter__(self) -> Self:
        await self._backend.__aenter__()

        # Create streaming bridges
        self._generate_chat_bridge = _AsyncStreamingBridge(self._heartbeat_interval)
        self._generate_text_bridge = _AsyncStreamingBridge(self._heartbeat_interval)
        self._score_tokens_bridge = _AsyncStreamingBridge(self._heartbeat_interval)
        self._score_text_bridge = _AsyncStreamingBridge(self._heartbeat_interval)
        self._score_chat_bridge = _AsyncStreamingBridge(self._heartbeat_interval)
        self._echo_logprobs_bridge = _AsyncStreamingBridge(self._heartbeat_interval)

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._run_generate_chat(), name="generate_chat"),
            asyncio.create_task(self._run_generate_text(), name="generate_text"),
            asyncio.create_task(self._run_score_tokens(), name="score_tokens"),
            asyncio.create_task(self._run_score_text(), name="score_text"),
            asyncio.create_task(self._run_score_chat(), name="score_chat"),
            asyncio.create_task(self._run_echo_token_logprobs(), name="echo_token_logprobs"),
        ]

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Signal all bridges to stop
        if self._generate_chat_bridge:
            self._generate_chat_bridge.shutdown()
        if self._generate_text_bridge:
            self._generate_text_bridge.shutdown()
        if self._score_tokens_bridge:
            self._score_tokens_bridge.shutdown()
        if self._score_text_bridge:
            self._score_text_bridge.shutdown()
        if self._score_chat_bridge:
            self._score_chat_bridge.shutdown()
        if self._echo_logprobs_bridge:
            self._echo_logprobs_bridge.shutdown()

        # Cancel and wait for tasks
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

        await self._backend.__aexit__(exc_type, exc_val, exc_tb)

    # =========================================================================
    # Public API
    # =========================================================================

    async def generate_chat_one(
        self,
        request: "GenerateChatRequest",
    ) -> "GenerationResult":
        assert self._generate_chat_bridge is not None
        result = await self._generate_chat_bridge.submit(request)
        if isinstance(result, BackendError):
            raise BackendRequestError(result)
        return result

    async def generate_text_one(
        self,
        request: "GenerateTextRequest",
    ) -> "GenerationResult":
        assert self._generate_text_bridge is not None
        result = await self._generate_text_bridge.submit(request)
        if isinstance(result, BackendError):
            raise BackendRequestError(result)
        return result

    async def score_tokens_one(
        self,
        request: "ScoreTokensRequest",
    ) -> "ScoreTokensResult":
        assert self._score_tokens_bridge is not None
        result = await self._score_tokens_bridge.submit(request)
        if isinstance(result, BackendError):
            raise BackendRequestError(result)
        return result

    async def score_text_one(
        self,
        request: "ScoreTextRequest",
    ) -> "ScoreTextResult":
        assert self._score_text_bridge is not None
        result = await self._score_text_bridge.submit(request)
        if isinstance(result, BackendError):
            raise BackendRequestError(result)
        return result

    async def score_chat_one(
        self,
        request: "ScoreChatRequest",
    ) -> "ScoreChatResult":
        assert self._score_chat_bridge is not None
        result = await self._score_chat_bridge.submit(request)
        if isinstance(result, BackendError):
            raise BackendRequestError(result)
        return result

    async def echo_token_logprobs_one(
        self,
        request: "EchoTokenLogprobsRequest",
    ) -> "EchoTokenLogprobsResult":
        assert self._echo_logprobs_bridge is not None
        result = await self._echo_logprobs_bridge.submit(request)
        if isinstance(result, BackendError):
            raise BackendRequestError(result)
        return result

    # =========================================================================
    # Background Workers
    # =========================================================================

    async def _run_generate_chat(self) -> None:
        """Background task: stream chat requests to backend, deliver results."""
        assert self._generate_chat_bridge is not None
        try:
            async for result in self._backend.generate_chat(self._generate_chat_bridge):
                await self._generate_chat_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("generate_chat worker failed")
            await self._generate_chat_bridge.deliver_exception(e)

    async def _run_generate_text(self) -> None:
        """Background task: stream text requests to backend, deliver results."""
        assert self._generate_text_bridge is not None
        try:
            async for result in self._backend.generate_text(self._generate_text_bridge):
                await self._generate_text_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("generate_text worker failed")
            await self._generate_text_bridge.deliver_exception(e)

    async def _run_score_tokens(self) -> None:
        """Background task: stream requests to backend, deliver results."""
        assert self._score_tokens_bridge is not None
        try:
            async for result in self._backend.score_tokens(self._score_tokens_bridge):
                await self._score_tokens_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("score_tokens worker failed")
            await self._score_tokens_bridge.deliver_exception(e)

    async def _run_score_text(self) -> None:
        """Background task: stream requests to backend, deliver results."""
        assert self._score_text_bridge is not None
        try:
            async for result in self._backend.score_text(self._score_text_bridge):
                await self._score_text_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("score_text worker failed")
            await self._score_text_bridge.deliver_exception(e)

    async def _run_score_chat(self) -> None:
        """Background task: stream requests to backend, deliver results."""
        assert self._score_chat_bridge is not None
        try:
            async for result in self._backend.score_chat(self._score_chat_bridge):
                await self._score_chat_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("score_chat worker failed")
            await self._score_chat_bridge.deliver_exception(e)

    async def _run_echo_token_logprobs(self) -> None:
        """Background task: stream echo logprob requests to backend, deliver results."""
        assert self._echo_logprobs_bridge is not None
        try:
            async for result in self._backend.echo_token_logprobs(
                self._echo_logprobs_bridge
            ):
                await self._echo_logprobs_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("echo_token_logprobs worker failed")
            await self._echo_logprobs_bridge.deliver_exception(e)


# =============================================================================
# Streaming Bridge - connects async callers to sync iterator consumers
# =============================================================================


class _StreamingBridge[Req, Res]:
    """Bridge async requests to a sync iterator.

    Thread-safe bridge using two queues (no locks in async context):
    - Request queue: carries requests from async to sync worker
    - Future queue: tracks futures in FIFO order for result delivery

    Flow:
    1. Async caller submits (request, future) via submit()
    2. Worker thread iterates via __iter__(), gets requests
    3. Worker processes requests, calls deliver_result() with results
    4. Results are dispatched to futures via call_soon_threadsafe()

    Heartbeats (None values) are yielded on timeout to allow the backend's
    batched_map to flush partial batches even when no requests arrive.
    """

    _SHUTDOWN = object()

    def __init__(self, heartbeat_interval: float) -> None:
        self._heartbeat_interval = heartbeat_interval
        self._request_queue: queue.Queue[Req | object] = queue.Queue()
        self._future_queue: queue.Queue[asyncio.Future[Res]] = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def submit(self, request: Req, future: asyncio.Future[Res]) -> None:
        """Submit a request from async context (non-blocking)."""
        # Future first, then request - ensures future exists before result arrives
        self._future_queue.put(future)
        self._request_queue.put(request)

    def shutdown(self) -> None:
        """Signal the worker to stop."""
        self._request_queue.put(self._SHUTDOWN)

    def __iter__(self) -> Iterator[Req | None]:
        """Yield requests to the sync consumer, with heartbeats on timeout."""
        while True:
            try:
                item = self._request_queue.get(timeout=self._heartbeat_interval)
                if item is self._SHUTDOWN:
                    return
                yield item  # type: ignore[misc]
            except queue.Empty:
                yield None  # Heartbeat for timeout-based batch flushing

    def deliver_result(self, result: Res) -> None:
        """Deliver a result to the next pending future (called from worker)."""
        try:
            future = self._future_queue.get_nowait()
        except queue.Empty:
            logger.error("Result delivered but no pending future")
            return

        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                lambda f=future, r=result: f.set_result(r) if not f.done() else None
            )

    def deliver_exception(self, exc: BaseException) -> None:
        """Deliver exception to all pending futures (called on worker failure)."""
        futures: list[asyncio.Future[Res]] = []
        while True:
            try:
                futures.append(self._future_queue.get_nowait())
            except queue.Empty:
                break

        if self._loop is not None:
            for future in futures:
                self._loop.call_soon_threadsafe(
                    lambda f=future, e=exc: f.set_exception(e) if not f.done() else None
                )


# =============================================================================
# SyncBackendAdapter - streaming bridge from async to sync
# =============================================================================


class SyncBackendAdapter(BackendAdapter):
    """Adapter for SyncBackend using streaming bridges.

    All operations use streaming bridges where requests flow continuously
    to the backend. The backend handles batching internally via batched_map
    (for scoring) or batched_map_by_key (for generation).
    """

    def __init__(
        self,
        backend: "SyncBackend",
        heartbeat_interval: float = 0.01,
    ) -> None:
        self._backend = backend
        self._heartbeat_interval = heartbeat_interval

        # Streaming bridges for all operations
        self._generate_chat_bridge: _StreamingBridge[
            "GenerateChatRequest", "GenerationResult"
        ] | None = None
        self._generate_text_bridge: _StreamingBridge[
            "GenerateTextRequest", "GenerationResult"
        ] | None = None
        self._score_tokens_bridge: _StreamingBridge[
            "ScoreTokensRequest", "ScoreTokensResult"
        ] | None = None
        self._score_text_bridge: _StreamingBridge[
            "ScoreTextRequest", "ScoreTextResult"
        ] | None = None
        self._score_chat_bridge: _StreamingBridge[
            "ScoreChatRequest", "ScoreChatResult"
        ] | None = None
        self._echo_logprobs_bridge: _StreamingBridge[
            "EchoTokenLogprobsRequest", "EchoTokenLogprobsResult"
        ] | None = None

        self._workers: list[threading.Thread] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> Self:
        await asyncio.to_thread(self._backend.__enter__)
        self._loop = asyncio.get_running_loop()

        # Create streaming bridges
        self._generate_chat_bridge = _StreamingBridge(self._heartbeat_interval)
        self._generate_chat_bridge.set_loop(self._loop)

        self._generate_text_bridge = _StreamingBridge(self._heartbeat_interval)
        self._generate_text_bridge.set_loop(self._loop)

        self._score_tokens_bridge = _StreamingBridge(self._heartbeat_interval)
        self._score_tokens_bridge.set_loop(self._loop)

        self._score_text_bridge = _StreamingBridge(self._heartbeat_interval)
        self._score_text_bridge.set_loop(self._loop)

        self._score_chat_bridge = _StreamingBridge(self._heartbeat_interval)
        self._score_chat_bridge.set_loop(self._loop)

        self._echo_logprobs_bridge = _StreamingBridge(self._heartbeat_interval)
        self._echo_logprobs_bridge.set_loop(self._loop)

        # Start worker threads
        self._workers = [
            threading.Thread(
                target=self._run_generate_chat, daemon=True, name="generate_chat"
            ),
            threading.Thread(
                target=self._run_generate_text, daemon=True, name="generate_text"
            ),
            threading.Thread(
                target=self._run_score_tokens, daemon=True, name="score_tokens"
            ),
            threading.Thread(
                target=self._run_score_text, daemon=True, name="score_text"
            ),
            threading.Thread(
                target=self._run_score_chat, daemon=True, name="score_chat"
            ),
            threading.Thread(
                target=self._run_echo_token_logprobs, daemon=True, name="echo_token_logprobs"
            ),
        ]
        for worker in self._workers:
            worker.start()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Signal all workers to stop
        if self._generate_chat_bridge:
            self._generate_chat_bridge.shutdown()
        if self._generate_text_bridge:
            self._generate_text_bridge.shutdown()
        if self._score_tokens_bridge:
            self._score_tokens_bridge.shutdown()
        if self._score_text_bridge:
            self._score_text_bridge.shutdown()
        if self._score_chat_bridge:
            self._score_chat_bridge.shutdown()
        if self._echo_logprobs_bridge:
            self._echo_logprobs_bridge.shutdown()

        # Wait for workers
        for worker in self._workers:
            worker.join(timeout=5.0)

        await asyncio.to_thread(self._backend.__exit__, None, None, None)

    # =========================================================================
    # Public API
    # =========================================================================

    async def generate_chat_one(
        self,
        request: "GenerateChatRequest",
    ) -> "GenerationResult":
        assert self._loop is not None and self._generate_chat_bridge is not None
        future: asyncio.Future["GenerationResult"] = self._loop.create_future()
        self._generate_chat_bridge.submit(request, future)
        return await future

    async def generate_text_one(
        self,
        request: "GenerateTextRequest",
    ) -> "GenerationResult":
        assert self._loop is not None and self._generate_text_bridge is not None
        future: asyncio.Future["GenerationResult"] = self._loop.create_future()
        self._generate_text_bridge.submit(request, future)
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

    async def echo_token_logprobs_one(
        self,
        request: "EchoTokenLogprobsRequest",
    ) -> "EchoTokenLogprobsResult":
        assert self._loop is not None and self._echo_logprobs_bridge is not None
        future: asyncio.Future["EchoTokenLogprobsResult"] = self._loop.create_future()
        self._echo_logprobs_bridge.submit(request, future)
        return await future

    # =========================================================================
    # Workers
    # =========================================================================

    def _run_generate_chat(self) -> None:
        """Worker thread: stream chat requests to backend, deliver results."""
        assert self._generate_chat_bridge is not None
        try:
            for result in self._backend.generate_chat(self._generate_chat_bridge):
                self._generate_chat_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("generate_chat worker failed")
            self._generate_chat_bridge.deliver_exception(e)

    def _run_generate_text(self) -> None:
        """Worker thread: stream text requests to backend, deliver results."""
        assert self._generate_text_bridge is not None
        try:
            for result in self._backend.generate_text(self._generate_text_bridge):
                self._generate_text_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("generate_text worker failed")
            self._generate_text_bridge.deliver_exception(e)

    def _run_score_tokens(self) -> None:
        """Worker thread: stream requests to backend, deliver results."""
        assert self._score_tokens_bridge is not None
        try:
            for result in self._backend.score_tokens(self._score_tokens_bridge):
                self._score_tokens_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("score_tokens worker failed")
            self._score_tokens_bridge.deliver_exception(e)

    def _run_score_text(self) -> None:
        """Worker thread: stream requests to backend, deliver results."""
        assert self._score_text_bridge is not None
        try:
            for result in self._backend.score_text(self._score_text_bridge):
                self._score_text_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("score_text worker failed")
            self._score_text_bridge.deliver_exception(e)

    def _run_score_chat(self) -> None:
        """Worker thread: stream requests to backend, deliver results."""
        assert self._score_chat_bridge is not None
        try:
            for result in self._backend.score_chat(self._score_chat_bridge):
                self._score_chat_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("score_chat worker failed")
            self._score_chat_bridge.deliver_exception(e)

    def _run_echo_token_logprobs(self) -> None:
        """Worker thread: stream echo logprob requests to backend, deliver results."""
        assert self._echo_logprobs_bridge is not None
        try:
            for result in self._backend.echo_token_logprobs(self._echo_logprobs_bridge):
                self._echo_logprobs_bridge.deliver_result(result)
        except Exception as e:
            logger.exception("echo_token_logprobs worker failed")
            self._echo_logprobs_bridge.deliver_exception(e)


def create_adapter(
    backend: "AsyncBackend | SyncBackend",
    heartbeat_interval: float = 0.01,
) -> BackendAdapter:
    """Create appropriate adapter for backend type."""
    from place_layers.backend.base import AsyncBackend, SyncBackend

    match backend:
        case SyncBackend():
            return SyncBackendAdapter(backend, heartbeat_interval)
        case AsyncBackend():
            return AsyncBackendAdapter(backend, heartbeat_interval)
        case other:
            assert_never(other)
