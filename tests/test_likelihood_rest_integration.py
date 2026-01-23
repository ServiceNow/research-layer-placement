"""Integration tests for likelihood evaluator with REST backend.

These tests spin up actual local Transformers backend servers and run real evaluations.
Purpose:
- Test end-to-end evaluation workflow with real servers
- Measure evaluation performance (problems/sec, throughput)
- Verify failover behavior under real conditions
- Validate accuracy computation with actual models

Usage:
    # Run with real local servers (CPU is fine, GPU recommended for speed)
    RUN_INTEGRATION_TESTS=1 pytest tests/test_likelihood_rest_integration.py -v -s

    # Optionally specify device map for GPU
    RUN_INTEGRATION_TESTS=1 TEST_DEVICE_MAP=auto pytest tests/test_likelihood_rest_integration.py -v -s

By default, these tests are skipped unless RUN_INTEGRATION_TESTS=1 is set.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from place_layers.backend.rest import RESTBackendConfig
from place_layers.evaluator.likelihood import LikelihoodEvaluatorConfig

# Skip integration tests by default unless environment variable is set
_integration_skip_marker = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Integration tests require RUN_INTEGRATION_TESTS=1 environment variable",
)


class ServerManager:
    """Manages lifecycle of local Transformers backend REST servers for testing."""

    def __init__(self, model_path: str, ports: list[int], device_map: str = "cpu"):
        """Initialize server manager.

        Args:
            model_path: Path to HuggingFace model (e.g., "Qwen/Qwen2-0.5B-Instruct")
            ports: List of ports to run servers on
            device_map: Device map for model loading (default: "cpu", use "auto" for GPU)
        """
        self.model_path = model_path
        self.ports = ports
        self.device_map = device_map
        self.processes: list[subprocess.Popen] = []

    async def start(self) -> list[str]:
        """Start all servers and wait for them to be ready.

        Returns:
            List of server URLs
        """
        # Get path to helper script
        script_path = Path(__file__).parent / "start_test_server.py"

        for port in self.ports:
            # Start Transformers backend server
            cmd = [
                sys.executable,  # Use same Python as test runner
                str(script_path),
                "--model-path",
                self.model_path,
                "--port",
                str(port),
                "--host",
                "localhost",
                "--device-map",
                self.device_map,
                "--batch-size",
                "8",
            ]

            print(f"Starting server on port {port} (device_map={self.device_map})...")
            print(f"  Command: {' '.join(cmd)}")
            # Redirect output to /dev/null to avoid blocking - we'll check health via HTTP
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.processes.append(process)
            print(f"  Process PID: {process.pid}")

        # Wait for servers to be ready
        urls = [f"http://localhost:{port}" for port in self.ports]
        await self._wait_for_servers(urls)

        print(f"All {len(urls)} servers ready!")
        return urls

    async def _wait_for_servers(self, urls: list[str], timeout: float = 180.0):
        """Wait for all servers to respond to /info endpoint.

        Args:
            urls: List of server URLs to wait for
            timeout: Timeout in seconds (default: 180s for model loading)
        """
        import aiohttp

        start_time = time.time()
        last_error = None

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                all_ready = True
                for url in urls:
                    try:
                        async with session.get(
                            f"{url}/info", timeout=aiohttp.ClientTimeout(total=5.0)
                        ) as resp:
                            if resp.status != 200:
                                all_ready = False
                                last_error = f"{url} returned status {resp.status}"
                                break
                    except Exception as e:
                        all_ready = False
                        last_error = f"{url} error: {e}"
                        break

                if all_ready:
                    print(f"All servers ready after {time.time() - start_time:.1f}s")
                    return

                # Check if any processes have died
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        print(f"  WARNING: Server process {i} (PID {process.pid}) has exited with code {process.returncode}")

                # Print progress every 10 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    print(f"  Still waiting ({elapsed:.0f}s elapsed)... {last_error}")

                await asyncio.sleep(2.0)

        raise TimeoutError(
            f"Servers not ready after {timeout}s. Last error: {last_error}"
        )

    def stop(self):
        """Stop all servers."""
        for process in self.processes:
            if process.poll() is None:  # Process still running
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"Process {process.pid} didn't terminate, killing it")
                    process.kill()
                    process.wait(timeout=5)
        self.processes = []


@pytest.fixture(scope="module")
def local_servers():
    """Fixture that provides two local REST servers for testing.

    Yields:
        Tuple of (server URLs, model_path)
    """
    # Use a small model for fast testing
    model_path = os.getenv("TEST_MODEL_PATH", "Qwen/Qwen2-0.5B-Instruct")
    device_map = os.getenv("TEST_DEVICE_MAP", "cpu")
    ports = [8080, 8081]

    print(f"\nSetting up test servers:")
    print(f"  Model: {model_path}")
    print(f"  Device map: {device_map}")
    print(f"  Ports: {ports}")

    manager = ServerManager(model_path, ports, device_map)

    # Run startup in sync context
    urls = asyncio.run(manager.start())

    try:
        yield urls, model_path
    finally:
        print("\nShutting down test servers...")
        manager.stop()


@_integration_skip_marker
class TestLikelihoodIntegration:
    """Integration tests with real local servers."""

    @pytest.mark.asyncio
    async def test_single_server_evaluation(self, local_servers, tmp_path):
        """Test basic evaluation with single real server."""
        urls, model_path = local_servers

        config = LikelihoodEvaluatorConfig(
            backend=RESTBackendConfig(
                servers=[urls[0]],
                max_retries=3,
                timeout_seconds=30,
            ),
            tasks=["mmlu_abstract_algebra"],
            num_fewshot=0,
            batch_size=8,
            max_problems_per_task=50,
            output_dir=str(tmp_path / "output"),
        )

        from place_layers.evaluator.likelihood import LikelihoodEvaluator

        evaluator = LikelihoodEvaluator(config)

        async with evaluator:
            start_time = time.time()
            accuracy = await evaluator.evaluate(None)
            elapsed = time.time() - start_time

            # Basic validation
            assert 0.0 <= accuracy <= 1.0, "Accuracy must be in [0, 1]"

            # Performance metrics
            problems_per_sec = 50 / elapsed
            print(f"\n=== Single Server Performance ===")
            print(f"Model: {model_path}")
            print(f"Problems: 50")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Throughput: {problems_per_sec:.1f} problems/sec")

    @pytest.mark.asyncio
    async def test_distributed_evaluation(self, local_servers, tmp_path):
        """Test evaluation distributed across two real servers."""
        urls, model_path = local_servers

        config = LikelihoodEvaluatorConfig(
            backend=RESTBackendConfig(
                servers=urls,  # Both servers
                max_retries=3,
                timeout_seconds=30,
            ),
            tasks=["mmlu_abstract_algebra"],
            num_fewshot=0,
            batch_size=16,  # Larger batch for distribution
            max_problems_per_task=50,
            output_dir=str(tmp_path / "output"),
        )

        from place_layers.evaluator.likelihood import LikelihoodEvaluator

        evaluator = LikelihoodEvaluator(config)

        async with evaluator:
            start_time = time.time()
            accuracy = await evaluator.evaluate(None)
            elapsed = time.time() - start_time

            assert 0.0 <= accuracy <= 1.0

            # Performance metrics
            problems_per_sec = 50 / elapsed
            print(f"\n=== Distributed Performance (2 servers) ===")
            print(f"Model: {model_path}")
            print(f"Problems: 50")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Throughput: {problems_per_sec:.1f} problems/sec")

    @pytest.mark.asyncio
    async def test_performance_measurement(self, local_servers, tmp_path):
        """Measure detailed performance metrics for optimization."""
        urls, model_path = local_servers

        # Test with different batch sizes
        batch_sizes = [8, 16, 32]
        results = []

        for batch_size in batch_sizes:
            config = LikelihoodEvaluatorConfig(
                backend=RESTBackendConfig(
                    servers=urls,
                    max_retries=3,
                    timeout_seconds=60,
                ),
                tasks=["mmlu_abstract_algebra"],
                num_fewshot=0,
                batch_size=batch_size,
                max_problems_per_task=50,
                output_dir=str(tmp_path / f"output_batch{batch_size}"),
            )

            from place_layers.evaluator.likelihood import LikelihoodEvaluator

            evaluator = LikelihoodEvaluator(config)

            async with evaluator:
                start_time = time.time()
                accuracy = await evaluator.evaluate(None)
                elapsed = time.time() - start_time

                throughput = 50 / elapsed
                chats_per_sec = (50 * 4) / elapsed  # 4 choices per problem

                results.append({
                    "batch_size": batch_size,
                    "elapsed": elapsed,
                    "throughput": throughput,
                    "chats_per_sec": chats_per_sec,
                    "accuracy": accuracy,
                })

        # Print comparison
        print(f"\n=== Performance Optimization Results ===")
        print(f"Model: {model_path}")
        print(f"Servers: {len(urls)}")
        print(f"Problems: 50 each")
        print("\nBatch Size | Time (s) | Problems/s | Chats/s | Accuracy")
        print("-" * 60)
        for r in results:
            print(
                f"{r['batch_size']:10d} | {r['elapsed']:8.2f} | "
                f"{r['throughput']:10.1f} | {r['chats_per_sec']:7.1f} | "
                f"{r['accuracy']:.3f}"
            )

        # Find optimal batch size
        best = max(results, key=lambda x: x["throughput"])
        print(f"\nOptimal batch size: {best['batch_size']} "
              f"({best['throughput']:.1f} problems/sec)")


class TestLikelihoodQuick:
    """Quick smoke tests that can run without integration flag.

    These tests just verify that the config and basic setup work,
    without actually running evaluations.
    """

    def test_config_creation(self):
        """Test that config can be created successfully."""
        config = LikelihoodEvaluatorConfig(
            backend=RESTBackendConfig(
                servers=["http://localhost:8080"],
                max_retries=3,
                timeout_seconds=10,
            ),
            tasks=["mmlu_abstract_algebra"],
            num_fewshot=0,
            batch_size=8,
            max_problems_per_task=10,
        )

        assert config.backend.servers == ["http://localhost:8080"]
        assert config.tasks == ["mmlu_abstract_algebra"]
        assert config.batch_size == 8

    def test_multi_server_config(self):
        """Test configuration with multiple servers."""
        config = LikelihoodEvaluatorConfig(
            backend=RESTBackendConfig(
                servers=[
                    "http://localhost:8080",
                    "http://localhost:8081",
                    "http://localhost:8082",
                ],
                max_retries=3,
                timeout_seconds=10,
                base_recovery_interval=0.5,
                max_recovery_interval=10.0,
            ),
            tasks=["mmlu_abstract_algebra", "mmlu_anatomy"],
            num_fewshot=5,
            batch_size=32,
            max_problems_per_task=100,
        )

        assert len(config.backend.servers) == 3
        assert len(config.tasks) == 2
        assert config.num_fewshot == 5
