"""Tests for REST backend failover and health tracking.

Focuses on real failure modes:
- Concurrent failover under load
- Flapping servers (rapid fail/recover cycles)
- Partial batch failures
- Recovery timing correctness
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest
from aioresponses import aioresponses

from place_layers.backend.rest import (
    RESTBackend,
    RESTBackendConfig,
    RESTBackendError,
    ServerPool,
)
from place_layers.backend.types import (
    BackendError,
    ScoreChatRequest,
    ScoreTextRequest,
    ScoreTokensRequest,
)
from place_layers.server.schemas import (
    GenerateRequest as ServerGenerateRequest,
)

pytestmark = pytest.mark.asyncio(loop_scope="function")


def mock_server_info() -> dict:
    """Standard mock /info response."""
    return {
        "model_path": "/models/test",
        "batch_size": 8,
        "supports_placement": True,
        "current_placement": None,
        "num_layers": 12,
        "mixer_names": ["attention", "mamba"],
    }


@asynccontextmanager
async def backend_with_mocks(
    config: RESTBackendConfig, mocked: aioresponses
) -> AsyncIterator[RESTBackend]:
    """Setup backend with /info mocked for all servers."""
    # Mock /info for server validation
    for server in config.servers:
        mocked.get(f"{server}/info", payload=mock_server_info())

    backend = RESTBackend(config)
    async with backend:
        yield backend


class TestConcurrentFailover:
    """Test failover behavior under concurrent load."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://s1:8080", "http://s2:8080"],
            base_recovery_interval=0.1,
            max_recovery_interval=1.0,
            max_retries=3,
            timeout_seconds=5,
        )

    async def test_concurrent_requests_failover_to_healthy_server(
        self, config: RESTBackendConfig
    ):
        """Multiple concurrent requests all fail over when first server dies."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Server 1 returns 503 for all requests
                for _ in range(10):
                    m.post("http://s1:8080/generate", status=503)
                # Server 2 handles all requests
                for i in range(10):
                    m.post(
                        "http://s2:8080/generate",
                        payload={"samples": [f"response_{i}"], "prompt_token_ids": []},
                    )

                # Fire 10 concurrent requests
                tasks = [
                    backend._call_with_failover(
                        "/generate",
                        ServerGenerateRequest(key=f"req_{i}", messages=[]),
                    )
                    for i in range(10)
                ]
                results = await asyncio.gather(*tasks)

                assert len(results) == 10
                # Server 1 should be marked unhealthy
                assert backend._server_pool._health["http://s1:8080"].healthy is False
                # Server 2 should be healthy
                assert backend._server_pool._health["http://s2:8080"].healthy is True

    async def test_all_servers_fail_exhausts_retries(self, config: RESTBackendConfig):
        """When all servers fail, retries are exhausted and error is raised."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Both servers fail repeatedly
                for _ in range(20):
                    m.post("http://s1:8080/generate", status=503)
                    m.post("http://s2:8080/generate", status=500)

                with pytest.raises(RESTBackendError) as exc_info:
                    await backend._call_with_failover(
                        "/generate",
                        ServerGenerateRequest(key="test", messages=[]),
                    )

                assert "All available servers failed" in str(exc_info.value)
                # Both servers should be unhealthy
                assert backend._server_pool._health["http://s1:8080"].healthy is False
                assert backend._server_pool._health["http://s2:8080"].healthy is False


class TestFlappingServers:
    """Test behavior when servers rapidly fail and recover."""

    def test_backoff_increases_on_repeated_failures(self):
        """Consecutive failures increase backoff time exponentially."""
        pool = ServerPool(
            servers=["http://s1:8080"],
            base_recovery_interval=10.0,
            max_recovery_interval=300.0,
        )

        # Fail multiple times and verify exponential backoff
        for i in range(5):
            pool.mark_failure("http://s1:8080")
            health = pool._health["http://s1:8080"]
            expected_wait = min(10.0 * (2 ** i), 300.0)
            actual_wait = health.next_check_time(10.0, 300.0) - health.last_failure_time
            assert abs(actual_wait - expected_wait) < 0.01

    def test_success_resets_backoff(self):
        """Successful request resets failure count and backoff."""
        pool = ServerPool(
            servers=["http://s1:8080"],
            base_recovery_interval=10.0,
            max_recovery_interval=300.0,
        )

        # Fail several times
        for _ in range(5):
            pool.mark_failure("http://s1:8080")

        assert pool._health["http://s1:8080"].consecutive_failures == 5

        # Success resets
        pool.mark_success("http://s1:8080")
        assert pool._health["http://s1:8080"].consecutive_failures == 0
        assert pool._health["http://s1:8080"].healthy is True

    def test_immediate_refailure_after_recovery_starts_fresh(self):
        """If server fails again after recovery, backoff starts from base."""
        pool = ServerPool(
            servers=["http://s1:8080"],
            base_recovery_interval=10.0,
            max_recovery_interval=300.0,
        )

        # Fail, recover, fail again
        pool.mark_failure("http://s1:8080")
        pool.mark_success("http://s1:8080")
        pool.mark_failure("http://s1:8080")

        # Should be back to first failure backoff (10s), not accumulated
        health = pool._health["http://s1:8080"]
        wait = health.next_check_time(10.0, 300.0) - health.last_failure_time
        assert abs(wait - 10.0) < 0.01


class TestRecoveryTiming:
    """Test that recovery checks happen at correct times."""

    def test_unhealthy_server_excluded_before_backoff_expires(self):
        """Unhealthy server is excluded until backoff time passes."""
        pool = ServerPool(
            servers=["http://s1:8080", "http://s2:8080"],
            base_recovery_interval=100.0,  # Long backoff
            max_recovery_interval=300.0,
        )

        pool.mark_failure("http://s1:8080")

        # Immediately after failure, only healthy server returned
        servers = pool.get_servers_to_try()
        assert servers == ["http://s2:8080"]

    def test_unhealthy_server_included_after_backoff_expires(self):
        """Unhealthy server is included for retry after backoff expires."""
        pool = ServerPool(
            servers=["http://s1:8080", "http://s2:8080"],
            base_recovery_interval=10.0,
            max_recovery_interval=300.0,
        )

        pool.mark_failure("http://s1:8080")

        # Simulate time passing beyond backoff
        pool._health["http://s1:8080"].last_failure_time = time.monotonic() - 15.0

        servers = pool.get_servers_to_try()
        # Healthy first, then unhealthy due for check
        assert servers == ["http://s2:8080", "http://s1:8080"]


class TestPartialBatchFailure:
    """Test batch operations when some servers fail."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://s1:8080", "http://s2:8080"],
            base_recovery_interval=0.1,
            max_recovery_interval=1.0,
            max_retries=3,
            timeout_seconds=5,
        )

    async def test_batch_succeeds_with_one_server_down(
        self, config: RESTBackendConfig
    ):
        """Batch completes when one server is down, using failover."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Server 1 is down for all requests
                for _ in range(10):
                    m.post("http://s1:8080/score_tokens", status=503)
                # Server 2 handles all requests
                for i in range(4):
                    m.post(
                        "http://s2:8080/score_tokens",
                        payload={"key": f"req_{i}", "log_prob": -1.0 - i},
                    )

                requests = [
                    ScoreTokensRequest(key=f"req_{i}", prompt_tokens=(i,), completion_tokens=(i + 1,))
                    for i in range(4)
                ]
                results = [r async for r in backend.score_tokens(requests)]

                assert len(results) == 4
                assert all(hasattr(r, "log_prob") for r in results)

    async def test_batch_fails_when_all_servers_down(
        self, config: RESTBackendConfig
    ):
        """Batch returns BackendError when all servers are unreachable."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Both servers fail repeatedly
                for _ in range(20):
                    m.post("http://s1:8080/score_tokens", status=503)
                    m.post("http://s2:8080/score_tokens", status=500)

                requests = [
                    ScoreTokensRequest(key=f"req_{i}", prompt_tokens=(i,), completion_tokens=(i + 1,))
                    for i in range(2)
                ]

                # Now returns BackendError in stream instead of raising
                results = [r async for r in backend.score_tokens(requests)]
                assert len(results) == 2
                assert all(isinstance(r, BackendError) for r in results)
                assert all(r.error_type == "server_error" for r in results)


class TestClientErrorsNotRetried:
    """Test that client errors (4xx) fail immediately without failover."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://s1:8080", "http://s2:8080"],
            base_recovery_interval=0.1,
            max_retries=3,
        )

    async def test_400_fails_immediately(self, config: RESTBackendConfig):
        """400 Bad Request fails immediately, no failover attempted."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                m.post("http://s1:8080/generate", status=400, body="Bad request")

                with pytest.raises(RESTBackendError) as exc_info:
                    await backend._call_with_failover(
                        "/generate",
                        ServerGenerateRequest(key="test", messages=[]),
                    )

                assert "400" in str(exc_info.value)

    async def test_404_fails_immediately(self, config: RESTBackendConfig):
        """404 Not Found fails immediately."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                m.post("http://s1:8080/generate", status=404, body="Not found")

                with pytest.raises(RESTBackendError) as exc_info:
                    await backend._call_with_failover(
                        "/generate",
                        ServerGenerateRequest(key="test", messages=[]),
                    )

                assert "404" in str(exc_info.value)


class TestServerRecovery:
    """Test server recovery after failure."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://s1:8080", "http://s2:8080"],
            base_recovery_interval=0.1,
            max_recovery_interval=1.0,
            max_retries=5,
        )

    async def test_server_marked_healthy_after_recovery(
        self, config: RESTBackendConfig
    ):
        """Server is restored to healthy after successful recovery check."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # First request: s1 fails, s2 succeeds
                m.post("http://s1:8080/generate", status=503)
                m.post(
                    "http://s2:8080/generate",
                    payload={"samples": ["ok"], "prompt_token_ids": []},
                )

                await backend._call_with_failover(
                    "/generate",
                    ServerGenerateRequest(key="test", messages=[]),
                )

                assert backend._server_pool._health["http://s1:8080"].healthy is False
                assert backend._server_pool._health["http://s2:8080"].healthy is True

                # Simulate time passing for recovery check to be due
                backend._server_pool._health["http://s1:8080"].last_failure_time = (
                    time.monotonic() - 10.0
                )

                # Next request: s2 fails, s1 recovers
                m.post("http://s2:8080/generate", status=503)
                m.post(
                    "http://s1:8080/generate",
                    payload={"samples": ["recovered"], "prompt_token_ids": []},
                )

                await backend._call_with_failover(
                    "/generate",
                    ServerGenerateRequest(key="test2", messages=[]),
                )

                # s1 should be healthy again
                assert backend._server_pool._health["http://s1:8080"].healthy is True


class TestConnectionErrors:
    """Test handling of network-level failures."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://s1:8080", "http://s2:8080"],
            base_recovery_interval=0.1,
            max_retries=3,
        )

    async def test_connection_error_triggers_failover(
        self, config: RESTBackendConfig
    ):
        """Connection errors trigger failover to next server."""
        import aiohttp

        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Server 1 raises connection error
                m.post(
                    "http://s1:8080/generate",
                    exception=aiohttp.ClientConnectionError("Connection refused"),
                )
                # Server 2 works
                m.post(
                    "http://s2:8080/generate",
                    payload={"samples": ["ok"], "prompt_token_ids": []},
                )

                result = await backend._call_with_failover(
                    "/generate",
                    ServerGenerateRequest(key="test", messages=[]),
                )

                assert result["samples"] == ["ok"]
                assert backend._server_pool._health["http://s1:8080"].healthy is False


class TestEdgeCases:
    """Edge cases that could cause subtle bugs."""

    def test_single_server_pool_works(self):
        """Pool with single server handles failure correctly."""
        pool = ServerPool(
            servers=["http://s1:8080"],
            base_recovery_interval=10.0,
        )

        assert pool.get_servers_to_try() == ["http://s1:8080"]

        pool.mark_failure("http://s1:8080")
        assert pool.get_servers_to_try() == []  # Not due for check yet
        assert pool.all_unhealthy() is True

    async def test_backend_requires_context_manager(self):
        """Backend raises if used without async context manager."""
        config = RESTBackendConfig(servers=["http://s1:8080"])
        backend = RESTBackend(config)

        with pytest.raises(RuntimeError, match="context manager"):
            backend._ensure_session()

    async def test_empty_requests_returns_empty(self):
        """Empty input to batch operations returns empty without server calls."""
        config = RESTBackendConfig(servers=["http://s1:8080"])

        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                results = [r async for r in backend.score_tokens([])]
                assert results == []


class TestScoreText:
    """Tests for score_text with failover."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://s1:8080", "http://s2:8080"],
            base_recovery_interval=0.1,
            max_recovery_interval=1.0,
            max_retries=3,
        )

    async def test_distributes_across_servers(self, config: RESTBackendConfig):
        """Text scoring requests are distributed across healthy servers."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Both servers respond successfully with full LogProbResult format
                for i in range(4):
                    server = "s1" if i % 2 == 0 else "s2"
                    m.post(
                        f"http://{server}:8080/score_text",
                        payload={
                            "key": f"req_{i}",
                            "results": [
                                {
                                    "token_ids": [i * 2, i * 2 + 1],
                                    "logprobs": [None, -1.0 - i],
                                    "answer_range": [1, 2],
                                    "total_log_prob": -1.0 - i,
                                }
                            ],
                        },
                    )

                requests = [
                    ScoreTextRequest(key=f"req_{i}", context=f"ctx{i}", choices=(f"choice_{i}",))
                    for i in range(4)
                ]
                results = [r async for r in backend.score_text(requests)]

                assert len(results) == 4
                # Each result is a ScoreTextResult with results containing LogProbResult
                assert all(hasattr(r, "results") for r in results)
                assert all(len(r.results) == 1 for r in results)

    async def test_failover_on_server_error(self, config: RESTBackendConfig):
        """Failing server triggers failover for text scoring."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Server 1 fails
                m.post("http://s1:8080/score_text", status=503)
                # Server 2 handles all
                m.post(
                    "http://s2:8080/score_text",
                    payload={
                        "key": "req_0",
                        "results": [
                            {
                                "token_ids": [1, 2, 3],
                                "logprobs": [None, -0.5, -1.0],
                                "answer_range": [2, 3],
                                "total_log_prob": -1.5,
                            }
                        ],
                    },
                )

                requests = [
                    ScoreTextRequest(key="req_0", context="The capital of France is", choices=(" Paris",))
                ]
                results = [r async for r in backend.score_text(requests)]

                assert len(results) == 1
                assert results[0].results[0].total_log_prob == -1.5
                assert backend._server_pool._health["http://s1:8080"].healthy is False

    async def test_empty_input_returns_empty(self, config: RESTBackendConfig):
        """Empty requests returns empty without server calls."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                results = [r async for r in backend.score_text([])]
                assert results == []


class TestScoreChat:
    """Tests for score_chat with failover."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://s1:8080", "http://s2:8080"],
            base_recovery_interval=0.1,
            max_recovery_interval=1.0,
            max_retries=3,
        )

    async def test_parses_response_correctly(self, config: RESTBackendConfig):
        """Chat scoring results are parsed correctly."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                m.post(
                    "http://s1:8080/score_chat",
                    payload={
                        "key": "req_0",
                        "results": [
                            {
                                "token_ids": [1, 2, 3],
                                "logprobs": [None, -0.5, -1.0],
                                "answer_range": [1, 3],
                                "total_log_prob": -1.5,
                            }
                        ],
                    },
                )

                requests = [
                    ScoreChatRequest(
                        key="req_0",
                        context_messages=[{"role": "user", "content": "What is 2+2?"}],
                        completion_messages=[{"role": "assistant", "content": "4"}],
                    )
                ]
                results = [r async for r in backend.score_chat(requests)]

                assert len(results) == 1
                assert results[0].results[0].token_ids == [1, 2, 3]
                assert results[0].results[0].logprobs == [None, -0.5, -1.0]
                assert results[0].results[0].answer_range == (1, 3)
                assert results[0].results[0].total_log_prob == -1.5

    async def test_failover_on_server_error(self, config: RESTBackendConfig):
        """Failing server triggers failover for chat scoring."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Server 1 fails
                m.post("http://s1:8080/score_chat", status=500)
                # Server 2 handles request
                m.post(
                    "http://s2:8080/score_chat",
                    payload={
                        "key": "req_0",
                        "results": [
                            {
                                "token_ids": [10, 20],
                                "logprobs": [None, -2.0],
                                "answer_range": [1, 2],
                                "total_log_prob": -2.0,
                            }
                        ],
                    },
                )

                requests = [
                    ScoreChatRequest(
                        key="req_0",
                        context_messages=[{"role": "user", "content": "Hi"}],
                        completion_messages=[{"role": "assistant", "content": "Hello"}],
                    )
                ]
                results = [r async for r in backend.score_chat(requests)]

                assert len(results) == 1
                assert results[0].results[0].total_log_prob == -2.0
                assert backend._server_pool._health["http://s1:8080"].healthy is False

    async def test_empty_input_returns_empty(self, config: RESTBackendConfig):
        """Empty requests returns empty without server calls."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                results = [r async for r in backend.score_chat([])]
                assert results == []
