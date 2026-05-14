"""PlacementSerialBackend - local per-request placement execution.

This backend wraps local worker backends and guarantees that each worker runs at
most one placement-homogeneous batch at a time. Requests may carry their own
placement; otherwise the backend's current placement, set via set_placement(),
is used as a default.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import logging
import os
import queue
import signal
import sys
import tempfile
import traceback
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, TypeVar

from pydantic import Field, model_validator

from place_layers.backend.base import AsyncBackend, BaseBackendConfig, SyncBackend
from place_layers.backend.mock import MockBackendConfig
from place_layers.backend.transformers import TransformersBackendConfig
from place_layers.backend.types import (
    BackendError,
    GenerateChatRequest,
    GenerateTextRequest,
    ModelStructure,
    NoStructure,
    ScoreChatRequest,
    ScoreChatResult,
    ScoreTextRequest,
    ScoreTextResult,
    ScoreTokensRequest,
    ScoreTokensResult,
    SupernetStructure,
)
from place_layers.backend.vllm import VLLMBackendConfig
from place_layers.clients.local_vllm import LocalVLLMClientConfig
from place_layers.clients.types import GenerationResult

logger = logging.getLogger(__name__)


WorkerBackendConfig = Annotated[
    TransformersBackendConfig | VLLMBackendConfig | MockBackendConfig,
    Field(discriminator="backend_type"),
]

ReqT = TypeVar("ReqT")
ResT = TypeVar("ResT")


async def _enter_backend(backend: AsyncBackend | SyncBackend) -> None:
    if isinstance(backend, SyncBackend):
        await asyncio.to_thread(backend.__enter__)
    else:
        await backend.__aenter__()


async def _exit_backend(
    backend: AsyncBackend | SyncBackend,
    exc_type,
    exc_val,
    exc_tb,
) -> None:
    if isinstance(backend, SyncBackend):
        await asyncio.to_thread(backend.__exit__, exc_type, exc_val, exc_tb)
    else:
        await backend.__aexit__(exc_type, exc_val, exc_tb)


async def _set_backend_placement(
    backend: AsyncBackend | SyncBackend,
    placement: list[str] | None,
) -> None:
    if placement is None:
        return
    if isinstance(backend, SyncBackend):
        await asyncio.to_thread(backend.set_placement, placement)
    else:
        await backend.set_placement(placement)


async def _run_backend_request(
    backend: AsyncBackend | SyncBackend,
    operation: str,
    request: Any,
) -> Any:
    if isinstance(backend, SyncBackend):
        return await asyncio.to_thread(
            _run_sync_backend_request,
            backend,
            operation,
            request,
        )
    return await _run_async_backend_request(backend, operation, request)


async def _run_backend_batch(
    backend: AsyncBackend | SyncBackend,
    operation: str,
    requests: list[Any],
) -> list[Any]:
    if isinstance(backend, SyncBackend):
        return await asyncio.to_thread(
            _run_sync_backend_batch,
            backend,
            operation,
            requests,
        )
    return await _run_async_backend_batch(backend, operation, requests)


def _run_sync_backend_request(
    backend: SyncBackend,
    operation: str,
    request: Any,
) -> Any:
    stream = getattr(backend, operation)([request])
    for result in stream:
        return result
    raise RuntimeError(f"{operation} yielded no result for request {request.key}")


def _run_sync_backend_batch(
    backend: SyncBackend,
    operation: str,
    requests: list[Any],
) -> list[Any]:
    results = list(getattr(backend, operation)(requests))
    if len(results) != len(requests):
        raise RuntimeError(
            f"{operation} yielded {len(results)} results for {len(requests)} requests"
        )
    return results


async def _run_async_backend_request(
    backend: AsyncBackend,
    operation: str,
    request: Any,
) -> Any:
    stream = getattr(backend, operation)([request])
    async for result in stream:
        return result
    raise RuntimeError(f"{operation} yielded no result for request {request.key}")


async def _run_async_backend_batch(
    backend: AsyncBackend,
    operation: str,
    requests: list[Any],
) -> list[Any]:
    results = [result async for result in getattr(backend, operation)(requests)]
    if len(results) != len(requests):
        raise RuntimeError(
            f"{operation} yielded {len(results)} results for {len(requests)} requests"
        )
    return results


async def _run_worker_batch_job(
    backend: AsyncBackend | SyncBackend,
    operation: str,
    requests: list[Any],
    placement: list[str] | None,
) -> list[Any | BackendError]:
    try:
        await _set_backend_placement(backend, placement)
        return await _run_backend_batch(backend, operation, requests)
    except ValueError as e:
        return [
            BackendError(
                key=getattr(request, "key", ""),
                error_type="bad_request",
                message=str(e),
            )
            for request in requests
        ]
    except NotImplementedError as e:
        return [
            BackendError(
                key=getattr(request, "key", ""),
                error_type="bad_request",
                message=str(e),
            )
            for request in requests
        ]
    except Exception as e:
        logger.exception("Placement-serial %s batch failed", operation)
        return [
            BackendError(
                key=getattr(request, "key", ""),
                error_type="server_error",
                message=str(e),
            )
            for request in requests
        ]


def _configure_subprocess_lifecycle() -> None:
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass

    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            pr_set_pdeathsig = 1
            libc.prctl(pr_set_pdeathsig, signal.SIGTERM)
        except Exception:
            pass
    _install_worker_termination_handlers()


def _install_worker_termination_handlers() -> None:
    def terminate_process_group(signum: int, _frame: Any) -> None:
        try:
            signal.signal(signum, signal.SIG_IGN)
        except Exception:
            pass
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpid(), signum)
            except ProcessLookupError:
                pass
            except OSError:
                pass
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, terminate_process_group)
        except Exception:
            pass


def _redirect_output_to_log(log_path: Path) -> Any:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
    return log_file


def _subprocess_worker_entry(
    *,
    index: int,
    worker_backend_config: WorkerBackendConfig,
    request_queue: Any,
    response_queue: Any,
    log_path: Path,
) -> None:
    _configure_subprocess_lifecycle()
    log_file = _redirect_output_to_log(log_path)
    try:
        print(
            f"[placement_serial worker {index}] starting local worker process "
            f"pid={os.getpid()}",
            flush=True,
        )
        asyncio.run(
            _subprocess_worker_main(
                index=index,
                worker_backend_config=worker_backend_config,
                request_queue=request_queue,
                response_queue=response_queue,
            )
        )
    finally:
        print(
            f"[placement_serial worker {index}] exiting local worker process",
            flush=True,
        )
        log_file.close()


async def _subprocess_worker_main(
    *,
    index: int,
    worker_backend_config: WorkerBackendConfig,
    request_queue: Any,
    response_queue: Any,
) -> None:
    backend: AsyncBackend | SyncBackend | None = None
    entered = False
    try:
        backend = worker_backend_config.create_backend()
        await _enter_backend(backend)
        entered = True
        response_queue.put(
            {
                "type": "ready",
                "worker_index": index,
                "model_structure": backend.model_structure,
            }
        )

        while True:
            message = await asyncio.to_thread(request_queue.get)
            message_type = message.get("type")
            if message_type == "shutdown":
                break
            if message_type not in {"run", "run_batch"}:
                raise ValueError(f"Unknown worker message type: {message_type!r}")

            requests = (
                message["requests"]
                if message_type == "run_batch"
                else [message["request"]]
            )
            results = await _run_worker_batch_job(
                backend=backend,
                operation=message["operation"],
                requests=requests,
                placement=message["placement"],
            )
            response_queue.put(
                {
                    "type": "result",
                    "worker_index": index,
                    "results": results,
                }
            )
    except BaseException as e:
        response_queue.put(
            {
                "type": "fatal_error" if entered else "startup_error",
                "worker_index": index,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        if entered and backend is not None:
            await _exit_backend(backend, None, None, None)
        response_queue.put({"type": "shutdown_complete", "worker_index": index})


@dataclass
class _Worker:
    index: int
    backend: AsyncBackend | SyncBackend | None = None
    subprocess: "_SubprocessWorker | None" = None
    active: bool = True


@dataclass(frozen=True)
class _CudaDeviceSlot:
    visible_id: str
    local_index: int


@dataclass
class _PlacedWork:
    request: Any
    placement: list[str] | None


@dataclass
class _PlacedBatch:
    requests: list[Any]
    placement: list[str] | None


def _placement_key(placement: list[str] | None) -> tuple[str, ...] | None:
    if placement is None:
        return None
    return tuple(placement)


class _SubprocessWorker:
    def __init__(
        self,
        *,
        index: int,
        worker_backend_config: WorkerBackendConfig,
        log_path: Path,
    ) -> None:
        self.index = index
        self.log_path = log_path
        self._failed = False
        ctx = mp.get_context("spawn")
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_subprocess_worker_entry,
            kwargs={
                "index": index,
                "worker_backend_config": worker_backend_config,
                "request_queue": self._request_queue,
                "response_queue": self._response_queue,
                "log_path": log_path,
            },
        )

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def is_healthy(self) -> bool:
        return not self._failed and self._process.is_alive()

    async def start(self) -> ModelStructure:
        self._process.start()
        message = await self._get_message({"ready", "startup_error"})
        if message["type"] == "startup_error":
            raise RuntimeError(
                f"Placement-serial worker {self.index} failed to start. "
                f"See {self.log_path}. Error: {message['error']}\n"
                f"{message['traceback']}"
            )
        return message["model_structure"]

    async def run_batch(
        self,
        *,
        operation: str,
        batch: _PlacedBatch,
    ) -> list[Any | BackendError]:
        self._request_queue.put(
            {
                "type": "run_batch",
                "operation": operation,
                "requests": batch.requests,
                "placement": batch.placement,
            }
        )
        message = await self._get_message({"result", "fatal_error"})
        if message["type"] == "fatal_error":
            self._failed = True
            raise RuntimeError(
                f"Placement-serial worker {self.index} failed. "
                f"See {self.log_path}. Error: {message['error']}\n"
                f"{message['traceback']}"
            )
        return message["results"]

    async def run(
        self,
        *,
        operation: str,
        request: Any,
        placement: list[str] | None,
    ) -> Any:
        results = await self.run_batch(
            operation=operation,
            batch=_PlacedBatch(requests=[request], placement=placement),
        )
        return results[0]

    async def shutdown(self, timeout_seconds: float) -> None:
        if self._process.pid is None:
            return
        if self._process.is_alive():
            self._request_queue.put({"type": "shutdown"})
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._process.join, timeout_seconds),
                    timeout_seconds + 1.0,
                )
            except asyncio.TimeoutError:
                pass

        if self._process.is_alive():
            self._terminate_process_group(signal.SIGTERM)
            await asyncio.to_thread(self._process.join, 5.0)

        if self._process.is_alive():
            self._terminate_process_group(signal.SIGKILL)
            await asyncio.to_thread(self._process.join, 5.0)

        await asyncio.to_thread(self._process.join, 0.0)
        self._request_queue.close()
        self._response_queue.close()

    async def _get_message(self, expected_types: set[str]) -> dict[str, Any]:
        while True:
            try:
                message = await asyncio.to_thread(
                    self._response_queue.get,
                    True,
                    1.0,
                )
            except queue.Empty:
                if not self._process.is_alive():
                    self._failed = True
                    raise RuntimeError(
                        f"Placement-serial worker {self.index} exited unexpectedly "
                        f"with code {self._process.exitcode}. See {self.log_path}."
                    )
                continue

            message_type = message.get("type")
            if message_type in expected_types:
                return message
            if message_type == "shutdown_complete":
                continue
            raise RuntimeError(
                f"Placement-serial worker {self.index} sent unexpected message "
                f"{message_type!r}. See {self.log_path}."
            )

    def _terminate_process_group(self, sig: signal.Signals) -> None:
        pid = self._process.pid
        if pid is None:
            return
        if hasattr(os, "killpg"):
            try:
                os.killpg(pid, sig)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        if sig == signal.SIGTERM:
            self._process.terminate()
        else:
            self._process.kill()


class PlacementSerialBackend(AsyncBackend):
    """Async backend for strict in-process per-request placement.

    Each local worker owns one backend instance and processes a single
    placement-homogeneous batch at a time. A batch may contain multiple requests
    when placement_batch_size > 1, but every request in the batch has the same
    placement and no other placement can run on that worker until the batch
    finishes. When all workers are busy, new requests wait until a worker
    returns to the idle queue.
    """

    def __init__(self, config: "PlacementSerialBackendConfig") -> None:
        self._config = config
        self._workers: list[_Worker] = []
        self._idle_workers: asyncio.Queue[_Worker] | None = None
        self._live_worker_count = 0
        self._current_placement: list[str] | None = None
        self._model_structure: ModelStructure = NoStructure()
        self._worker_log_dir: Path | None = None

    @property
    def config(self) -> "PlacementSerialBackendConfig":
        return self._config

    @property
    def max_concurrent_requests(self) -> int:
        if self._idle_workers is not None:
            return self._live_worker_count
        return self._resolved_num_workers()

    async def __aenter__(self) -> Self:
        idle_workers: asyncio.Queue[_Worker] = asyncio.Queue()
        cuda_device_slots = self._cuda_device_slots()
        num_workers = self._resolved_num_workers(cuda_device_slots)
        startup_tasks: list[tuple[_Worker, asyncio.Task[ModelStructure]]] = []
        if cuda_device_slots and num_workers > len(cuda_device_slots):
            raise ValueError(
                "PlacementSerialBackend requested "
                f"{num_workers} worker(s), but only {len(cuda_device_slots)} CUDA "
                "device(s) are visible."
            )

        try:
            self._worker_log_dir = self._resolve_worker_log_dir()
            self._worker_log_dir.mkdir(parents=True, exist_ok=True)
            for index in range(num_workers):
                cuda_device_slot = (
                    cuda_device_slots[index] if cuda_device_slots else None
                )
                worker_backend_config = self._worker_backend_config(cuda_device_slot)
                if self._use_subprocess_worker(worker_backend_config):
                    subprocess = _SubprocessWorker(
                        index=index,
                        worker_backend_config=worker_backend_config,
                        log_path=self._worker_log_dir / f"worker_{index}.log",
                    )
                    worker = _Worker(index=index, subprocess=subprocess)
                    startup_tasks.append(
                        (worker, asyncio.create_task(subprocess.start()))
                    )
                else:
                    backend = worker_backend_config.create_backend()
                    await _enter_backend(backend)
                    worker = _Worker(index=index, backend=backend)
                self._workers.append(worker)
                idle_workers.put_nowait(worker)

            for worker, task in startup_tasks:
                structure = await task
                if worker.index == 0:
                    self._model_structure = structure
        except Exception:
            await self.__aexit__(None, None, None)
            raise

        self._live_worker_count = len(self._workers)
        self._idle_workers = idle_workers
        if (
            self._workers
            and isinstance(self._model_structure, NoStructure)
            and self._workers[0].backend is not None
        ):
            self._model_structure = self._workers[0].backend.model_structure

        logger.info(
            "PlacementSerialBackend initialized with %d local worker(s), worker=%s, cuda_devices=%s, log_dir=%s",
            len(self._workers),
            self._config.worker_backend.backend_type,
            [slot.visible_id for slot in cuda_device_slots],
            self._worker_log_dir,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        while self._workers:
            worker = self._workers.pop()
            if worker.subprocess is not None:
                await worker.subprocess.shutdown(self._config.shutdown_timeout_seconds)
            elif worker.backend is not None:
                await _exit_backend(worker.backend, exc_type, exc_val, exc_tb)
        self._idle_workers = None
        self._live_worker_count = 0
        self._model_structure = NoStructure()

    @property
    def model_structure(self) -> ModelStructure:
        return self._model_structure

    def _resolve_worker_log_dir(self) -> Path:
        if self._config.worker_log_dir is not None:
            return self._config.worker_log_dir
        return Path(tempfile.mkdtemp(prefix="placement_serial_workers_"))

    def _use_subprocess_worker(self, worker_backend: WorkerBackendConfig) -> bool:
        return (
            self._config.subprocess_workers
            and isinstance(worker_backend, VLLMBackendConfig)
            and isinstance(worker_backend.client, LocalVLLMClientConfig)
        )

    def _resolved_num_workers(
        self,
        cuda_device_slots: list[_CudaDeviceSlot] | None = None,
    ) -> int:
        if self._config.num_workers is not None:
            return self._config.num_workers

        if cuda_device_slots is None:
            cuda_device_slots = self._cuda_device_slots()
        if cuda_device_slots:
            return len(cuda_device_slots)
        return 1

    def _cuda_device_slots(self) -> list[_CudaDeviceSlot]:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            device_ids = [
                device_id.strip()
                for device_id in cuda_visible_devices.split(",")
                if device_id.strip()
            ]
            if device_ids == ["-1"]:
                return []
            return [
                _CudaDeviceSlot(visible_id=device_id, local_index=index)
                for index, device_id in enumerate(device_ids)
            ]

        try:
            import torch
        except ImportError:
            return []

        if not torch.cuda.is_available():
            return []
        return [
            _CudaDeviceSlot(visible_id=str(index), local_index=index)
            for index in range(torch.cuda.device_count())
        ]

    def _worker_backend_config(
        self,
        cuda_device_slot: _CudaDeviceSlot | None,
    ) -> WorkerBackendConfig:
        worker_backend = self._config.worker_backend

        if isinstance(worker_backend, VLLMBackendConfig) and isinstance(
            worker_backend.client, LocalVLLMClientConfig
        ):
            client_updates: dict[str, Any] = {
                "max_concurrent_requests": self._config.placement_batch_size,
            }
            if cuda_device_slot is not None:
                client_updates["cuda_visible_devices"] = cuda_device_slot.visible_id
            return worker_backend.model_copy(
                update={
                    "client": worker_backend.client.model_copy(
                        update=client_updates,
                    )
                }
            )

        if cuda_device_slot is None:
            return worker_backend

        if isinstance(worker_backend, TransformersBackendConfig):
            return worker_backend.model_copy(
                update={"device_map": f"cuda:{cuda_device_slot.local_index}"}
            )

        return worker_backend

    @property
    def placement(self) -> list[str] | None:
        return self._current_placement

    async def set_placement(self, placement: Sequence[str]) -> None:
        """Set the default placement used by requests without placement fields."""
        validated_placement = list(placement)
        self._validate_placement(validated_placement)
        self._current_placement = validated_placement

    def _validate_placement(self, placement: list[str]) -> None:
        if not isinstance(self._model_structure, SupernetStructure):
            return
        if len(placement) != self._model_structure.num_layers:
            raise ValueError(
                f"Placement length {len(placement)} doesn't match "
                f"num_layers {self._model_structure.num_layers}"
            )
        unknown_mixers = set(placement) - set(self._model_structure.mixer_names)
        if unknown_mixers:
            raise ValueError(f"Unknown mixer names: {sorted(unknown_mixers)}")

    def generate_chat(
        self,
        requests: Iterable[GenerateChatRequest]
        | AsyncIterable[GenerateChatRequest | None],
    ) -> AsyncIterator[GenerationResult | BackendError]:
        return self._run_stream(requests, "generate_chat")

    def generate_text(
        self,
        requests: Iterable[GenerateTextRequest]
        | AsyncIterable[GenerateTextRequest | None],
    ) -> AsyncIterator[GenerationResult | BackendError]:
        return self._run_stream(requests, "generate_text")

    async def score_tokens(
        self,
        requests: Iterable[ScoreTokensRequest]
        | AsyncIterable[ScoreTokensRequest | None],
    ) -> AsyncIterator[ScoreTokensResult | BackendError]:
        async for result in self._run_stream(requests, "score_tokens"):
            yield result

    async def score_text(
        self,
        requests: Iterable[ScoreTextRequest] | AsyncIterable[ScoreTextRequest | None],
    ) -> AsyncIterator[ScoreTextResult | BackendError]:
        async for result in self._run_stream(requests, "score_text"):
            yield result

    async def score_chat(
        self,
        requests: Iterable[ScoreChatRequest] | AsyncIterable[ScoreChatRequest | None],
    ) -> AsyncIterator[ScoreChatResult | BackendError]:
        async for result in self._run_stream(requests, "score_chat"):
            yield result

    async def _run_stream(
        self,
        requests: Iterable[ReqT] | AsyncIterable[ReqT | None],
        operation: str,
    ) -> AsyncIterator[ResT | BackendError]:
        tasks: set[asyncio.Task[list[ResT | BackendError]]] = set()
        pending_by_placement: dict[tuple[str, ...] | None, list[_PlacedWork]] = {}
        pending_count = 0

        def enqueue(work: _PlacedWork) -> None:
            nonlocal pending_count
            pending_by_placement.setdefault(
                _placement_key(work.placement),
                [],
            ).append(work)
            pending_count += 1

        def pop_batch(*, full_only: bool) -> _PlacedBatch | None:
            nonlocal pending_count
            if not pending_by_placement:
                return None

            eligible = [
                (key, works)
                for key, works in pending_by_placement.items()
                if works and (not full_only or len(works) >= self._config.placement_batch_size)
            ]
            if not eligible:
                return None

            key, works = max(eligible, key=lambda item: len(item[1]))
            batch_size = min(self._config.placement_batch_size, len(works))
            batch_works = works[:batch_size]
            del works[:batch_size]
            if not works:
                del pending_by_placement[key]
            pending_count -= len(batch_works)
            return _PlacedBatch(
                requests=[work.request for work in batch_works],
                placement=batch_works[0].placement,
            )

        def dispatch(batch: _PlacedBatch) -> None:
            tasks.add(asyncio.create_task(self._process_batch(operation, batch)))

        def fail_pending_no_workers() -> list[BackendError]:
            nonlocal pending_count
            errors: list[BackendError] = []
            for works in pending_by_placement.values():
                for work in works:
                    errors.append(
                        BackendError(
                            key=getattr(work.request, "key", ""),
                            error_type="server_error",
                            message="No live placement-serial workers remain.",
                        )
                    )
            pending_by_placement.clear()
            pending_count = 0
            return errors

        async def drain_one() -> AsyncIterator[ResT | BackendError]:
            nonlocal tasks
            done, tasks = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for completed in done:
                for result in completed.result():
                    yield result

        async def dispatch_available(*, full_only: bool) -> None:
            if self.max_concurrent_requests <= 0:
                return
            while len(tasks) < self.max_concurrent_requests:
                batch = pop_batch(full_only=full_only)
                if batch is None:
                    break
                dispatch(batch)

        async for work in self._capture_work(requests):
            enqueue(work)
            if self.max_concurrent_requests <= 0:
                for result in fail_pending_no_workers():
                    yield result
                continue

            await dispatch_available(full_only=True)

            if len(tasks) >= self.max_concurrent_requests:
                async for result in drain_one():
                    yield result
                if self.max_concurrent_requests <= 0:
                    for result in fail_pending_no_workers():
                        yield result
                    continue
                await dispatch_available(full_only=True)

            if pending_count >= self._max_pending_requests():
                await dispatch_available(full_only=False)
                if len(tasks) >= self.max_concurrent_requests:
                    async for result in drain_one():
                        yield result
                    if self.max_concurrent_requests <= 0:
                        for result in fail_pending_no_workers():
                            yield result

        await dispatch_available(full_only=False)
        while tasks:
            async for result in drain_one():
                yield result
            if self.max_concurrent_requests <= 0:
                for result in fail_pending_no_workers():
                    yield result
                break
            await dispatch_available(full_only=False)

        if pending_count > 0 and self.max_concurrent_requests <= 0:
            for result in fail_pending_no_workers():
                yield result

    async def _capture_work(
        self,
        requests: Iterable[ReqT] | AsyncIterable[ReqT | None],
    ) -> AsyncIterator[_PlacedWork]:
        if isinstance(requests, AsyncIterable):
            async for request in requests:
                if request is None:
                    continue
                yield _PlacedWork(
                    request=request,
                    placement=self._placement_for_request(request),
                )
        else:
            for request in requests:
                yield _PlacedWork(
                    request=request,
                    placement=self._placement_for_request(request),
                )

    def _placement_for_request(self, request: object) -> list[str] | None:
        placement = getattr(request, "placement", None)
        if placement is not None:
            request_placement = list(placement)
            self._validate_placement(request_placement)
            return request_placement
        if self._current_placement is not None:
            return list(self._current_placement)
        return None

    def _max_pending_requests(self) -> int:
        return max(
            self._config.placement_batch_size * max(self.max_concurrent_requests, 1) * 8,
            self._config.placement_batch_size,
        )

    def _retire_worker(self, worker: _Worker) -> None:
        if not worker.active:
            return
        worker.active = False
        self._live_worker_count = max(0, self._live_worker_count - 1)
        try:
            self._workers.remove(worker)
        except ValueError:
            pass

    async def _process_batch(
        self,
        operation: str,
        batch: _PlacedBatch,
    ) -> list[Any | BackendError]:
        if self._idle_workers is None:
            raise RuntimeError(
                "PlacementSerialBackend must be used as an async context manager"
            )

        worker = await self._idle_workers.get()
        requeue_worker = True
        try:
            if worker.subprocess is not None:
                return await worker.subprocess.run_batch(
                    operation=operation,
                    batch=batch,
                )
            if worker.backend is None:
                raise RuntimeError(f"Worker {worker.index} has no backend")
            return await _run_worker_batch_job(
                backend=worker.backend,
                operation=operation,
                requests=batch.requests,
                placement=batch.placement,
            )
        except ValueError as e:
            return [
                BackendError(
                    key=getattr(request, "key", ""),
                    error_type="bad_request",
                    message=str(e),
                )
                for request in batch.requests
            ]
        except NotImplementedError as e:
            return [
                BackendError(
                    key=getattr(request, "key", ""),
                    error_type="bad_request",
                    message=str(e),
                )
                for request in batch.requests
            ]
        except Exception as e:
            if worker.subprocess is not None:
                requeue_worker = False
            logger.exception("Placement-serial %s request failed", operation)
            return [
                BackendError(
                    key=getattr(request, "key", ""),
                    error_type="server_error",
                    message=str(e),
                )
                for request in batch.requests
            ]
        finally:
            if worker.subprocess is not None and not worker.subprocess.is_healthy:
                requeue_worker = False
            if requeue_worker:
                self._idle_workers.put_nowait(worker)
            elif worker.subprocess is not None:
                self._retire_worker(worker)
                await worker.subprocess.shutdown(self._config.shutdown_timeout_seconds)


class PlacementSerialBackendConfig(BaseBackendConfig):
    """Configuration for in-process placement-serial execution."""

    _impl_class: ClassVar[type[AsyncBackend]] = PlacementSerialBackend

    backend_type: Literal["placement_serial"] = "placement_serial"
    worker_backend: WorkerBackendConfig = Field(
        description=(
            "Local backend config to instantiate for each worker. Supports "
            "TransformersBackend and local VLLMBackend."
        )
    )
    num_workers: int | None = Field(
        default=None,
        description="Number of local model workers to instantiate.",
    )
    placement_batch_size: int = Field(
        default=1,
        ge=1,
        description=(
            "Maximum number of same-placement requests in one worker batch. "
            "The worker still runs only one placement batch at a time."
        ),
    )
    subprocess_workers: bool = Field(
        default=True,
        description=(
            "Run local vLLM workers in subprocesses so model loads and logs are "
            "isolated from the parent scheduler."
        ),
    )
    worker_log_dir: Path | None = Field(
        default=None,
        description=(
            "Directory for subprocess worker stdout/stderr logs. Defaults to a "
            "temporary directory when subprocess workers are used."
        ),
    )
    shutdown_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Graceful shutdown timeout before terminating worker processes.",
    )

    @model_validator(mode="after")
    def validate_worker_backend(self) -> "PlacementSerialBackendConfig":
        if self.num_workers is not None and self.num_workers < 1:
            raise ValueError("num_workers must be >= 1 when set")
        if isinstance(self.worker_backend, VLLMBackendConfig) and not isinstance(
            self.worker_backend.client, LocalVLLMClientConfig
        ):
            raise ValueError(
                "PlacementSerialBackend only supports local vLLM workers; "
                "remote vLLM already owns its own concurrent server scheduler."
            )
        return self
