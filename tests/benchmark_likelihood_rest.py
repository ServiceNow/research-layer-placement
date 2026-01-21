"""Performance benchmarks for likelihood evaluator with REST backend.

This script measures the performance characteristics of the REST backend
when used with the likelihood evaluator, focusing on:
- Requests per second
- Problems per second
- Tokens per second
- Speedup from multiple servers
- Impact of batch size

Run with: python -m pytest tests/benchmark_likelihood_rest.py -v -s
"""

from __future__ import annotations

import asyncio
import time
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
        "model_path": "/models/qwen2-0.5b",
        "batch_size": 8,
        "supports_placement": True,
        "current_placement": None,
        "num_layers": 24,
        "mixer_names": ["attention", "gdn"],
    }


def mock_chat_log_probs_response(num_results: int, avg_tokens_per_result: int = 5) -> dict:
    """Mock /compute_chat_log_probs response.

    Args:
        num_results: Number of results to return
        avg_tokens_per_result: Average tokens per result (for token counting)
    """
    results = []
    for i in range(num_results):
        # Simulate realistic token counts (answer is typically 1 token, but full sequence has more)
        results.append({
            "token_ids": list(range(avg_tokens_per_result)),
            "logprobs": [None] + [-0.1] * (avg_tokens_per_result - 1),
            "answer_range": [3, 4],
            "total_log_prob": -0.1,
        })
    return {"results": results}


@asynccontextmanager
async def backend_with_mocks(
    config: RESTBackendConfig, mocked: aioresponses, num_requests: int, results_per_request: int
) -> AsyncIterator[RESTBackend]:
    """Setup REST backend with enough mocked responses."""
    # Mock /info for all servers
    for server in config.servers:
        mocked.get(f"{server}/info", payload=mock_server_info())

    # Mock compute_chat_log_probs responses (enough for all requests)
    for server in config.servers:
        for _ in range(num_requests):
            mocked.post(
                f"{server}/compute_chat_log_probs",
                payload=mock_chat_log_probs_response(results_per_request),
            )

    backend = RESTBackend(config)
    async with backend:
        yield backend


class TestSingleServerPerformance:
    """Benchmark single server performance."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://localhost:8080"],
            max_retries=2,
            timeout_seconds=30,
        )

    async def test_throughput_small_batch(self, config: RESTBackendConfig):
        """Measure throughput with small batches (8 chats per request)."""
        num_problems = 50
        choices_per_problem = 4
        total_chats = num_problems * choices_per_problem  # 200 chats

        with aioresponses() as m:
            # Mock enough responses (200 chats / 8 per request = 25 requests)
            async with backend_with_mocks(config, m, num_requests=30, results_per_request=8) as backend:
                # Build chats
                chats = []
                for _ in range(num_problems):
                    for choice in ["A", "B", "C", "D"]:
                        context = [ChatMessage(role="user", content="What is 2+2?")]
                        completion = [ChatMessage(role="assistant", content=choice)]
                        chats.append((context, completion))

                # Measure performance
                start_time = time.time()
                results = await backend.compute_chat_log_probs(chats)
                elapsed = time.time() - start_time

                # Calculate metrics
                problems_per_sec = num_problems / elapsed
                chats_per_sec = len(results) / elapsed
                tokens_per_sec = len(results) * 5 / elapsed  # Assume 5 tokens per result

                print(f"\n{'='*70}")
                print(f"Single Server - Small Batch (batch_size=8)")
                print(f"{'='*70}")
                print(f"Total problems:     {num_problems}")
                print(f"Total chats:        {total_chats}")
                print(f"Elapsed time:       {elapsed:.3f}s")
                print(f"Problems/sec:       {problems_per_sec:.1f}")
                print(f"Chats/sec:          {chats_per_sec:.1f}")
                print(f"Tokens/sec:         {tokens_per_sec:.0f}")
                print(f"Avg latency:        {elapsed/num_problems*1000:.1f}ms per problem")
                print(f"{'='*70}\n")

                assert len(results) == total_chats

    async def test_throughput_large_batch(self, config: RESTBackendConfig):
        """Measure throughput with larger batches (32 chats per request)."""
        num_problems = 50
        choices_per_problem = 4
        total_chats = num_problems * choices_per_problem

        # Simulate larger server batch size
        mock_info = mock_server_info()
        mock_info["batch_size"] = 32

        with aioresponses() as m:
            # Mock with larger batches
            m.get("http://localhost:8080/info", payload=mock_info)

            # Mock enough responses (200 chats / 32 per request = 7 requests)
            for _ in range(10):
                m.post(
                    "http://localhost:8080/compute_chat_log_probs",
                    payload=mock_chat_log_probs_response(32),
                )

            backend = RESTBackend(config)
            async with backend:
                # Build chats
                chats = []
                for _ in range(num_problems):
                    for choice in ["A", "B", "C", "D"]:
                        context = [ChatMessage(role="user", content="What is 2+2?")]
                        completion = [ChatMessage(role="assistant", content=choice)]
                        chats.append((context, completion))

                # Measure performance
                start_time = time.time()
                results = await backend.compute_chat_log_probs(chats)
                elapsed = time.time() - start_time

                # Calculate metrics
                problems_per_sec = num_problems / elapsed
                chats_per_sec = len(results) / elapsed
                tokens_per_sec = len(results) * 5 / elapsed

                print(f"\n{'='*70}")
                print(f"Single Server - Large Batch (batch_size=32)")
                print(f"{'='*70}")
                print(f"Total problems:     {num_problems}")
                print(f"Total chats:        {total_chats}")
                print(f"Elapsed time:       {elapsed:.3f}s")
                print(f"Problems/sec:       {problems_per_sec:.1f}")
                print(f"Chats/sec:          {chats_per_sec:.1f}")
                print(f"Tokens/sec:         {tokens_per_sec:.0f}")
                print(f"Avg latency:        {elapsed/num_problems*1000:.1f}ms per problem")
                print(f"{'='*70}\n")


class TestMultiServerPerformance:
    """Benchmark multi-server performance and speedup."""

    async def test_speedup_two_servers(self):
        """Measure speedup from using two servers vs one."""
        num_problems = 100
        choices_per_problem = 4
        total_chats = num_problems * choices_per_problem

        # Baseline: Single server
        config_1 = RESTBackendConfig(
            servers=["http://localhost:8080"],
            max_retries=2,
            timeout_seconds=30,
        )

        with aioresponses() as m:
            async with backend_with_mocks(config_1, m, num_requests=60, results_per_request=8) as backend:
                chats = []
                for _ in range(num_problems):
                    for choice in ["A", "B", "C", "D"]:
                        context = [ChatMessage(role="user", content="Q?")]
                        completion = [ChatMessage(role="assistant", content=choice)]
                        chats.append((context, completion))

                start_time = time.time()
                results_1 = await backend.compute_chat_log_probs(chats)
                elapsed_1 = time.time() - start_time

        # Two servers
        config_2 = RESTBackendConfig(
            servers=["http://localhost:8080", "http://localhost:8081"],
            max_retries=2,
            timeout_seconds=30,
        )

        with aioresponses() as m:
            async with backend_with_mocks(config_2, m, num_requests=60, results_per_request=8) as backend:
                chats = []
                for _ in range(num_problems):
                    for choice in ["A", "B", "C", "D"]:
                        context = [ChatMessage(role="user", content="Q?")]
                        completion = [ChatMessage(role="assistant", content=choice)]
                        chats.append((context, completion))

                start_time = time.time()
                results_2 = await backend.compute_chat_log_probs(chats)
                elapsed_2 = time.time() - start_time

        # Calculate speedup
        speedup = elapsed_1 / elapsed_2
        efficiency = speedup / 2 * 100  # Percentage of ideal speedup

        print(f"\n{'='*70}")
        print(f"Multi-Server Speedup Analysis")
        print(f"{'='*70}")
        print(f"Total problems:     {num_problems}")
        print(f"Total chats:        {total_chats}")
        print(f"")
        print(f"Single server:")
        print(f"  Time:             {elapsed_1:.3f}s")
        print(f"  Problems/sec:     {num_problems/elapsed_1:.1f}")
        print(f"  Chats/sec:        {total_chats/elapsed_1:.1f}")
        print(f"")
        print(f"Two servers:")
        print(f"  Time:             {elapsed_2:.3f}s")
        print(f"  Problems/sec:     {num_problems/elapsed_2:.1f}")
        print(f"  Chats/sec:        {total_chats/elapsed_2:.1f}")
        print(f"")
        print(f"Speedup:            {speedup:.2f}x")
        print(f"Efficiency:         {efficiency:.1f}% (of ideal 2x)")
        print(f"{'='*70}\n")

    async def test_speedup_scaling(self):
        """Measure how performance scales with number of servers."""
        num_problems = 100
        choices_per_problem = 4
        total_chats = num_problems * choices_per_problem

        results_summary = []

        for num_servers in [1, 2, 3, 4]:
            servers = [f"http://localhost:{8080+i}" for i in range(num_servers)]
            config = RESTBackendConfig(
                servers=servers,
                max_retries=2,
                timeout_seconds=30,
            )

            with aioresponses() as m:
                async with backend_with_mocks(config, m, num_requests=60, results_per_request=8) as backend:
                    chats = []
                    for _ in range(num_problems):
                        for choice in ["A", "B", "C", "D"]:
                            context = [ChatMessage(role="user", content="Q?")]
                            completion = [ChatMessage(role="assistant", content=choice)]
                            chats.append((context, completion))

                    start_time = time.time()
                    results = await backend.compute_chat_log_probs(chats)
                    elapsed = time.time() - start_time

                    results_summary.append({
                        "servers": num_servers,
                        "elapsed": elapsed,
                        "problems_per_sec": num_problems / elapsed,
                        "chats_per_sec": total_chats / elapsed,
                    })

        # Print scaling results
        print(f"\n{'='*70}")
        print(f"Performance Scaling Analysis")
        print(f"{'='*70}")
        print(f"Total problems: {num_problems}")
        print(f"Total chats:    {total_chats}")
        print(f"")
        print(f"Servers | Time (s) | Problems/s | Chats/s | Speedup | Efficiency")
        print(f"{'-'*70}")

        baseline_time = results_summary[0]["elapsed"]
        for r in results_summary:
            speedup = baseline_time / r["elapsed"]
            efficiency = speedup / r["servers"] * 100
            print(
                f"{r['servers']:7d} | {r['elapsed']:8.3f} | "
                f"{r['problems_per_sec']:10.1f} | {r['chats_per_sec']:7.1f} | "
                f"{speedup:7.2f}x | {efficiency:6.1f}%"
            )

        print(f"{'='*70}\n")


class TestConcurrencyPerformance:
    """Benchmark concurrent request handling."""

    @pytest.fixture
    def config(self) -> RESTBackendConfig:
        return RESTBackendConfig(
            servers=["http://localhost:8080", "http://localhost:8081"],
            max_retries=2,
            timeout_seconds=30,
        )

    async def test_concurrent_requests_throughput(self, config: RESTBackendConfig):
        """Measure throughput with concurrent evaluation requests."""
        num_concurrent = 5
        problems_per_request = 20
        choices_per_problem = 4

        with aioresponses() as m:
            # Mock plenty of responses
            async with backend_with_mocks(config, m, num_requests=100, results_per_request=8) as backend:

                async def evaluate_batch():
                    """Simulate one evaluation batch."""
                    chats = []
                    for _ in range(problems_per_request):
                        for choice in ["A", "B", "C", "D"]:
                            context = [ChatMessage(role="user", content="Q?")]
                            completion = [ChatMessage(role="assistant", content=choice)]
                            chats.append((context, completion))
                    return await backend.compute_chat_log_probs(chats)

                # Run concurrent evaluations
                start_time = time.time()
                results = await asyncio.gather(*[evaluate_batch() for _ in range(num_concurrent)])
                elapsed = time.time() - start_time

                # Calculate metrics
                total_problems = num_concurrent * problems_per_request
                total_chats = sum(len(r) for r in results)
                problems_per_sec = total_problems / elapsed
                chats_per_sec = total_chats / elapsed

                print(f"\n{'='*70}")
                print(f"Concurrent Request Performance")
                print(f"{'='*70}")
                print(f"Concurrent requests: {num_concurrent}")
                print(f"Problems/request:    {problems_per_request}")
                print(f"Total problems:      {total_problems}")
                print(f"Total chats:         {total_chats}")
                print(f"Elapsed time:        {elapsed:.3f}s")
                print(f"Problems/sec:        {problems_per_sec:.1f}")
                print(f"Chats/sec:           {chats_per_sec:.1f}")
                print(f"Avg latency:         {elapsed/num_concurrent*1000:.1f}ms per batch")
                print(f"{'='*70}\n")


class TestBatchSizeOptimization:
    """Find optimal batch size for throughput."""

    async def test_batch_size_comparison(self):
        """Compare performance across different batch sizes."""
        num_problems = 50
        choices_per_problem = 4
        total_chats = num_problems * choices_per_problem

        batch_sizes = [4, 8, 16, 32, 64]
        results_summary = []

        for batch_size in batch_sizes:
            config = RESTBackendConfig(
                servers=["http://localhost:8080", "http://localhost:8081"],
                max_retries=2,
                timeout_seconds=30,
            )

            mock_info = mock_server_info()
            mock_info["batch_size"] = batch_size

            with aioresponses() as m:
                # Mock /info with this batch size
                for server in config.servers:
                    m.get(f"{server}/info", payload=mock_info)

                # Mock enough responses
                num_requests = (total_chats // batch_size) + 2
                for server in config.servers:
                    for _ in range(num_requests):
                        m.post(
                            f"{server}/compute_chat_log_probs",
                            payload=mock_chat_log_probs_response(batch_size),
                        )

                backend = RESTBackend(config)
                async with backend:
                    chats = []
                    for _ in range(num_problems):
                        for choice in ["A", "B", "C", "D"]:
                            context = [ChatMessage(role="user", content="Q?")]
                            completion = [ChatMessage(role="assistant", content=choice)]
                            chats.append((context, completion))

                    start_time = time.time()
                    results = await backend.compute_chat_log_probs(chats)
                    elapsed = time.time() - start_time

                    results_summary.append({
                        "batch_size": batch_size,
                        "elapsed": elapsed,
                        "problems_per_sec": num_problems / elapsed,
                        "chats_per_sec": total_chats / elapsed,
                    })

        # Print comparison
        print(f"\n{'='*70}")
        print(f"Batch Size Optimization")
        print(f"{'='*70}")
        print(f"Total problems: {num_problems}")
        print(f"Servers:        2")
        print(f"")
        print(f"Batch | Time (s) | Problems/s | Chats/s | Relative")
        print(f"{'-'*70}")

        baseline = results_summary[0]["problems_per_sec"]
        for r in results_summary:
            relative = r["problems_per_sec"] / baseline
            print(
                f"{r['batch_size']:5d} | {r['elapsed']:8.3f} | "
                f"{r['problems_per_sec']:10.1f} | {r['chats_per_sec']:7.1f} | "
                f"{relative:7.2f}x"
            )

        # Find optimal
        optimal = max(results_summary, key=lambda x: x["problems_per_sec"])
        print(f"\nOptimal batch size: {optimal['batch_size']} "
              f"({optimal['problems_per_sec']:.1f} problems/sec)")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
