"""Tests for the in-process placement-serial backend."""

from __future__ import annotations

import asyncio

import pytest

from place_layers.backend.mock import MockBackend, MockBackendConfig, ReplayConfig
from place_layers.backend.placement_serial import (
    PlacementSerialBackend,
    PlacementSerialBackendConfig,
    _SubprocessWorker,
)
from place_layers.backend.types import (
    BackendError,
    GenerateTextRequest,
    PlacedGenerateTextRequest,
)
from place_layers.backend.vllm import VLLMBackendConfig
from place_layers.clients.local_vllm import LocalVLLMClientConfig
from place_layers.clients.types import GeneratedSample, GenerationResult
from place_layers.clients.remote_vllm import RemoteVLLMClientConfig
from place_layers.server.app import create_app


def _mock_config() -> MockBackendConfig:
    return MockBackendConfig(
        num_layers=2,
        mixer_names=["a", "b"],
        max_concurrent_requests=99,
        replay=ReplayConfig(
            generate_chat=[],
            generate_text=[],
            score_tokens=[],
            score_text=[],
            score_chat=[],
        ),
    )


def test_uses_per_request_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    async def generate_text(self: MockBackend, requests):
        async for request in self._iter_requests(requests):
            if request is None:
                continue
            placement = self.placement
            assert placement == request.placement
            yield GenerationResult(
                key=request.key,
                samples=[
                    GeneratedSample(
                        text="/".join(placement or []),
                        token_ids=[],
                        token_logprobs=[],
                        truncated=False,
                    )
                ],
                prompt_token_ids=[],
            )

    monkeypatch.setattr(MockBackend, "generate_text", generate_text)

    async def run() -> list[GenerationResult]:
        config = PlacementSerialBackendConfig(
            worker_backend=_mock_config(),
            num_workers=1,
        )

        async with config.create_backend() as backend:
            requests = [
                PlacedGenerateTextRequest(
                    key="r0",
                    prompt="x",
                    placement=["a", "a"],
                ),
                PlacedGenerateTextRequest(
                    key="r1",
                    prompt="x",
                    placement=["b", "a"],
                ),
            ]
            return [result async for result in backend.generate_text(requests)]

    results = asyncio.run(run())
    assert [result.samples[0].text for result in results] == ["a/a", "b/a"]


def test_requests_without_placement_use_default(monkeypatch: pytest.MonkeyPatch) -> None:
    async def generate_text(self: MockBackend, requests):
        async for request in self._iter_requests(requests):
            if request is None:
                continue
            assert self.placement == ["b", "b"]
            yield GenerationResult(
                key=request.key,
                samples=[
                    GeneratedSample(
                        text="ok",
                        token_ids=[],
                        token_logprobs=[],
                        truncated=False,
                    )
                ],
                prompt_token_ids=[],
            )

    monkeypatch.setattr(MockBackend, "generate_text", generate_text)

    async def run() -> list[GenerationResult]:
        config = PlacementSerialBackendConfig(
            worker_backend=_mock_config(),
            num_workers=1,
        )

        async with config.create_backend() as backend:
            await backend.set_placement(["b", "b"])
            requests = [GenerateTextRequest(key="r0", prompt="x")]
            return [result async for result in backend.generate_text(requests)]

    results = asyncio.run(run())
    assert results[0].key == "r0"


def test_batches_requests_with_same_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    batch_sizes: list[int] = []

    async def generate_text(self: MockBackend, requests):
        batch_sizes.append(len(requests))
        async for request in self._iter_requests(requests):
            if request is None:
                continue
            yield GenerationResult(
                key=request.key,
                samples=[
                    GeneratedSample(
                        text=request.key,
                        token_ids=[],
                        token_logprobs=[],
                        truncated=False,
                    )
                ],
                prompt_token_ids=[],
            )

    monkeypatch.setattr(MockBackend, "generate_text", generate_text)

    async def run() -> list[GenerationResult]:
        config = PlacementSerialBackendConfig(
            worker_backend=_mock_config(),
            num_workers=1,
            placement_batch_size=2,
        )

        async with config.create_backend() as backend:
            requests = [
                PlacedGenerateTextRequest(
                    key="r0",
                    prompt="x",
                    placement=["a", "b"],
                ),
                PlacedGenerateTextRequest(
                    key="r1",
                    prompt="x",
                    placement=["a", "b"],
                ),
                PlacedGenerateTextRequest(
                    key="r2",
                    prompt="x",
                    placement=["b", "a"],
                ),
            ]
            return [result async for result in backend.generate_text(requests)]

    results = asyncio.run(run())
    assert [result.key for result in results] == ["r0", "r1", "r2"]
    assert batch_sizes == [2, 1]


def test_no_live_workers_fails_pending_requests_instead_of_hanging() -> None:
    async def run() -> list[BackendError]:
        config = PlacementSerialBackendConfig(
            worker_backend=_mock_config(),
            num_workers=1,
        )
        backend = config.create_backend()
        assert isinstance(backend, PlacementSerialBackend)
        backend._idle_workers = asyncio.Queue()
        backend._live_worker_count = 0
        requests = [GenerateTextRequest(key="r0", prompt="x")]
        return [result async for result in backend.generate_text(requests)]

    results = asyncio.run(run())
    assert len(results) == 1
    assert isinstance(results[0], BackendError)
    assert results[0].key == "r0"
    assert "No live placement-serial workers" in results[0].message


def test_invalid_default_placement_is_rejected() -> None:
    async def run() -> None:
        config = PlacementSerialBackendConfig(
            worker_backend=_mock_config(),
            num_workers=1,
        )

        async with config.create_backend() as backend:
            with pytest.raises(ValueError, match="Placement length"):
                await backend.set_placement(["a"])
            with pytest.raises(ValueError, match="Unknown mixer"):
                await backend.set_placement(["a", "c"])

    asyncio.run(run())


def test_rest_server_rejects_placement_serial_backend() -> None:
    with pytest.raises(ValueError, match="not supported by the REST server"):
        create_app(
            PlacementSerialBackendConfig(
                worker_backend=_mock_config(),
                num_workers=1,
            )
        )


def test_one_worker_serializes_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0

    async def generate_text(self: MockBackend, requests):
        nonlocal active, max_active
        async for request in self._iter_requests(requests):
            if request is None:
                continue
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            yield GenerationResult(
                key=request.key,
                samples=[
                    GeneratedSample(
                        text=request.key,
                        token_ids=[],
                        token_logprobs=[],
                        truncated=False,
                    )
                ],
                prompt_token_ids=[],
            )

    monkeypatch.setattr(MockBackend, "generate_text", generate_text)

    async def run() -> list[GenerationResult]:
        config = PlacementSerialBackendConfig(
            worker_backend=_mock_config(),
            num_workers=1,
        )

        async with config.create_backend() as backend:
            requests = [
                PlacedGenerateTextRequest(
                    key=f"r{i}",
                    prompt="x",
                    placement=["a", "b"],
                )
                for i in range(5)
            ]
            return [result async for result in backend.generate_text(requests)]

    results = asyncio.run(run())
    assert sorted(result.key for result in results) == [f"r{i}" for i in range(5)]
    assert max_active == 1


def test_remote_vllm_worker_is_rejected() -> None:
    with pytest.raises(ValueError, match="only supports local vLLM"):
        PlacementSerialBackendConfig(
            worker_backend=VLLMBackendConfig(
                client=RemoteVLLMClientConfig(
                    base_url="http://localhost:8000",
                    model="model",
                    api_key="EMPTY",
                    timeout_seconds=300,
                    max_concurrent_requests=32,
                )
            ),
            num_workers=1,
        )


def test_default_worker_count_follows_visible_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    async def run() -> int:
        config = PlacementSerialBackendConfig(worker_backend=_mock_config())
        async with config.create_backend() as backend:
            return backend.max_concurrent_requests

    assert asyncio.run(run()) == 2


def test_local_vllm_workers_are_pinned_to_visible_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    config = PlacementSerialBackendConfig(
        worker_backend=VLLMBackendConfig(
            client=LocalVLLMClientConfig(
                model_path="model",
                tokenizer_path=None,
                torch_dtype="bfloat16",
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                max_model_len=None,
                disable_compile=True,
                enable_prefix_caching=False,
                attention_backend=None,
                max_concurrent_requests=99,
            )
        )
    )
    backend = config.create_backend()
    assert isinstance(backend, PlacementSerialBackend)

    slots = backend._cuda_device_slots()
    worker_config = backend._worker_backend_config(slots[1])

    assert isinstance(worker_config, VLLMBackendConfig)
    assert isinstance(worker_config.client, LocalVLLMClientConfig)
    assert worker_config.client.cuda_visible_devices == "5"
    assert worker_config.client.max_concurrent_requests == 1


def test_subprocess_worker_roundtrip_with_logs(tmp_path) -> None:
    async def run() -> tuple[str, str]:
        worker_config = _mock_config().model_copy(
            update={
                "replay": ReplayConfig(
                    generate_chat=[],
                    generate_text=[
                        GenerationResult(
                            key="r0",
                            samples=[
                                GeneratedSample(
                                    text="ok",
                                    token_ids=[],
                                    token_logprobs=[],
                                    truncated=False,
                                )
                            ],
                            prompt_token_ids=[],
                        )
                    ],
                    score_tokens=[],
                    score_text=[],
                    score_chat=[],
                )
            }
        )
        worker = _SubprocessWorker(
            index=0,
            worker_backend_config=worker_config,
            log_path=tmp_path / "worker_0.log",
        )
        try:
            structure = await worker.start()
            result = await worker.run(
                operation="generate_text",
                request=GenerateTextRequest(key="r0", prompt="x"),
                placement=["a", "b"],
            )
        finally:
            await worker.shutdown(timeout_seconds=5.0)
        return structure.structure_type, result.samples[0].text

    assert asyncio.run(run()) == ("supernet", "ok")
    assert (tmp_path / "worker_0.log").exists()
