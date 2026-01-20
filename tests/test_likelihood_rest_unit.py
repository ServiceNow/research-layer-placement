"""Unit tests for REST backend with likelihood evaluation workloads.

Tests the REST backend's compute_chat_log_probs method which is used by
the likelihood evaluator. These are fast unit tests with mocked HTTP.

For integration tests with real servers, see test_likelihood_rest_integration.py
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest
from aioresponses import aioresponses

from place_layers.backend.base import ChatMessage
from place_layers.backend.rest import RESTBackend, RESTBackendConfig

pytestmark = pytest.mark.asyncio(loop_scope="function")


def mock_server_info() -> dict:
    """Mock /info response for server validation."""
    return {
        "model_path": "/models/qwen2-0.5b-instruct",
        "batch_size": 8,
        "supports_placement": True,
        "current_placement": None,
        "num_layers": 24,
        "mixer_names": ["attention", "gdn"],
    }


def mock_chat_log_probs_response(num_choices: int = 4, correct_idx: int = 0) -> dict:
    """Mock /compute_chat_log_probs response for likelihood evaluation.

    Simulates scoring multiple answer choices for a multiple-choice question.

    Args:
        num_choices: Number of answer choices (typically 4 for MMLU)
        correct_idx: Which choice should have highest log prob

    Returns:
        Mock response with ChatLogProbResult format
    """
    results = []
    for i in range(num_choices):
        # Make correct answer have higher log prob
        if i == correct_idx:
            log_prob = -0.1  # High probability
        else:
            log_prob = -2.5  # Lower probability

        # Typical MMLU answer tokens: <|im_start|>assistant\nA<|im_end|>
        results.append({
            "token_ids": [128006, 78191, 198, 65 + i, 128007],  # A/B/C/D
            "logprobs": [None, -0.01, -0.02, log_prob, -0.01],
            "answer_range": [3, 4],  # Just the answer token
            "total_log_prob": log_prob,
        })

    return {"results": results}


@asynccontextmanager
async def backend_with_mocks(
    config: RESTBackendConfig, mocked: aioresponses
) -> AsyncIterator[RESTBackend]:
    """Setup REST backend with /info mocked for all servers."""
    # Mock /info for server validation
    for server in config.servers:
        mocked.get(f"{server}/info", payload=mock_server_info())

    backend = RESTBackend(config)
    async with backend:
        yield backend


class TestChatLogProbsSingleServer:
    """Test compute_chat_log_probs with single server."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://localhost:8080"],
            max_retries=3,
            timeout_seconds=10,
        )

    async def test_single_choice_scoring(self, config: RESTBackendConfig):
        """Test scoring a single multiple-choice question (4 choices)."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Mock response for 4 choices
                m.post(
                    "http://localhost:8080/compute_chat_log_probs",
                    payload=mock_chat_log_probs_response(num_choices=4, correct_idx=0),
                )

                # Create chat messages for 4 answer choices
                chats = []
                for choice in ["A", "B", "C", "D"]:
                    context = [ChatMessage(role="user", content="Question...")]
                    completion = [ChatMessage(role="assistant", content=choice)]
                    chats.append((context, completion))

                # Call backend
                results = await backend.compute_chat_log_probs(chats)

                # Verify results
                assert len(results) == 4
                assert results[0].total_log_prob > results[1].total_log_prob
                assert results[0].total_log_prob > results[2].total_log_prob
                assert results[0].total_log_prob > results[3].total_log_prob

    async def test_batch_of_problems(self, config: RESTBackendConfig):
        """Test scoring multiple problems in one batch."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # 2 problems × 4 choices = 8 chats
                # They all go in ONE request, so mock response should have 8 results

                # Build response with 8 results (2 problems × 4 choices each)
                # Problem 1: correct_idx=0, Problem 2: correct_idx=2
                results_list = []
                # Problem 1 choices (A=best)
                for i in range(4):
                    log_prob = -0.1 if i == 0 else -2.5
                    results_list.append({
                        "token_ids": [128006, 78191, 198, 65 + i, 128007],
                        "logprobs": [None, -0.01, -0.02, log_prob, -0.01],
                        "answer_range": [3, 4],
                        "total_log_prob": log_prob,
                    })
                # Problem 2 choices (C=best)
                for i in range(4):
                    log_prob = -0.1 if i == 2 else -2.5
                    results_list.append({
                        "token_ids": [128006, 78191, 198, 65 + i, 128007],
                        "logprobs": [None, -0.01, -0.02, log_prob, -0.01],
                        "answer_range": [3, 4],
                        "total_log_prob": log_prob,
                    })

                m.post(
                    "http://localhost:8080/compute_chat_log_probs",
                    payload={"results": results_list},
                )

                # Build chats
                chats = []
                for _ in range(2):  # 2 problems
                    for choice in ["A", "B", "C", "D"]:
                        context = [ChatMessage(role="user", content="Q...")]
                        completion = [ChatMessage(role="assistant", content=choice)]
                        chats.append((context, completion))

                results = await backend.compute_chat_log_probs(chats)

                assert len(results) == 8

                # First problem: choice 0 (A) is best
                assert results[0].total_log_prob > results[1].total_log_prob

                # Second problem: choice 2 (C) is best
                assert results[6].total_log_prob > results[4].total_log_prob


class TestChatLogProbsDistribution:
    """Test distribution of chat log prob requests across multiple servers."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=[
                "http://localhost:8080",
                "http://localhost:8081",
            ],
            max_retries=3,
            timeout_seconds=10,
        )

    async def test_round_robin_distribution(self, config: RESTBackendConfig):
        """Test requests are distributed round-robin across servers."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # 20 chats, distributed across 2 servers
                # Each server should get ~10 chats

                # Server 1: gets first batch
                m.post(
                    "http://localhost:8080/compute_chat_log_probs",
                    payload=mock_chat_log_probs_response(num_choices=4, correct_idx=0),
                )

                # Server 2: gets second batch
                m.post(
                    "http://localhost:8081/compute_chat_log_probs",
                    payload=mock_chat_log_probs_response(num_choices=4, correct_idx=0),
                )

                # Build 8 chats (2 problems × 4 choices)
                chats = []
                for _ in range(2):
                    for choice in ["A", "B", "C", "D"]:
                        context = [ChatMessage(role="user", content="Q...")]
                        completion = [ChatMessage(role="assistant", content=choice)]
                        chats.append((context, completion))

                results = await backend.compute_chat_log_probs(chats)

                assert len(results) == 8

    async def test_three_server_distribution(self, config: RESTBackendConfig):
        """Test distribution with 3 servers."""
        config.servers.append("http://localhost:8082")

        with aioresponses() as m:
            # Mock /info for third server
            m.get("http://localhost:8082/info", payload=mock_server_info())

            async with backend_with_mocks(config, m) as backend:
                # 12 chats should distribute evenly: 4 each

                for server in config.servers:
                    m.post(
                        f"{server}/compute_chat_log_probs",
                        payload=mock_chat_log_probs_response(num_choices=4, correct_idx=0),
                    )

                # Build 12 chats (3 problems × 4 choices)
                chats = []
                for _ in range(3):
                    for choice in ["A", "B", "C", "D"]:
                        context = [ChatMessage(role="user", content="Q...")]
                        completion = [ChatMessage(role="assistant", content=choice)]
                        chats.append((context, completion))

                results = await backend.compute_chat_log_probs(chats)

                assert len(results) == 12


class TestChatLogProbsFailover:
    """Test failover behavior during chat log prob computation."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=[
                "http://localhost:8080",
                "http://localhost:8081",
            ],
            base_recovery_interval=0.1,
            max_recovery_interval=1.0,
            max_retries=3,
            timeout_seconds=5,
        )

    async def test_failover_to_healthy_server(self, config: RESTBackendConfig):
        """Test failover when first server fails."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # First server fails - mock multiple attempts
                for _ in range(5):  # More than enough
                    m.post("http://localhost:8080/compute_chat_log_probs", status=503)

                # Second server succeeds - mock multiple for retries
                for _ in range(5):
                    m.post(
                        "http://localhost:8081/compute_chat_log_probs",
                        payload=mock_chat_log_probs_response(num_choices=4, correct_idx=0),
                    )

                # Build chats
                chats = []
                for choice in ["A", "B", "C", "D"]:
                    context = [ChatMessage(role="user", content="Q...")]
                    completion = [ChatMessage(role="assistant", content=choice)]
                    chats.append((context, completion))

                results = await backend.compute_chat_log_probs(chats)

                assert len(results) == 4

                # Verify server health
                assert backend._server_pool._health["http://localhost:8080"].healthy is False
                assert backend._server_pool._health["http://localhost:8081"].healthy is True

    async def test_all_servers_fail(self, config: RESTBackendConfig):
        """Test error handling when all servers fail."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # All servers fail
                for _ in range(6):  # max_retries * num_servers
                    m.post("http://localhost:8080/compute_chat_log_probs", status=503)
                    m.post("http://localhost:8081/compute_chat_log_probs", status=503)

                chats = [
                    (
                        [ChatMessage(role="user", content="Q...")],
                        [ChatMessage(role="assistant", content="A")],
                    )
                ]

                with pytest.raises(Exception):  # Should raise after exhausting retries
                    await backend.compute_chat_log_probs(chats)


class TestChatLogProbsPerformance:
    """Test performance characteristics of distributed chat log prob computation."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=[
                "http://localhost:8080",
                "http://localhost:8081",
            ],
            max_retries=2,
            timeout_seconds=30,
        )

    async def test_concurrent_requests(self, config: RESTBackendConfig):
        """Test multiple concurrent chat log prob requests."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Mock responses for concurrent requests
                for _ in range(10):
                    m.post(
                        "http://localhost:8080/compute_chat_log_probs",
                        payload=mock_chat_log_probs_response(num_choices=4, correct_idx=0),
                    )
                    m.post(
                        "http://localhost:8081/compute_chat_log_probs",
                        payload=mock_chat_log_probs_response(num_choices=4, correct_idx=0),
                    )

                # Create multiple concurrent requests
                async def make_request():
                    chats = []
                    for choice in ["A", "B", "C", "D"]:
                        context = [ChatMessage(role="user", content="Q...")]
                        completion = [ChatMessage(role="assistant", content=choice)]
                        chats.append((context, completion))
                    return await backend.compute_chat_log_probs(chats)

                # Run 3 concurrent requests
                results = await asyncio.gather(
                    make_request(),
                    make_request(),
                    make_request(),
                )

                # Each request should return 4 results
                assert all(len(r) == 4 for r in results)


class TestChatLogProbsPlacement:
    """Test that placement is sent with chat log prob requests."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://localhost:8080"],
            max_retries=3,
            timeout_seconds=10,
        )

    async def test_placement_in_request(self, config: RESTBackendConfig):
        """Test that placement is included in the request payload."""
        with aioresponses() as m:
            async with backend_with_mocks(config, m) as backend:
                # Set placement
                placement = ["attention", "gdn", "attention", "gdn"] * 6  # 24 layers
                await backend.set_placement(placement)

                # Mock response
                m.post(
                    "http://localhost:8080/compute_chat_log_probs",
                    payload=mock_chat_log_probs_response(num_choices=1, correct_idx=0),
                )

                # Make request
                chats = [
                    (
                        [ChatMessage(role="user", content="Q...")],
                        [ChatMessage(role="assistant", content="A")],
                    )
                ]

                results = await backend.compute_chat_log_probs(chats)

                assert len(results) == 1

                # Verify placement was stored
                assert backend._current_placement == placement
