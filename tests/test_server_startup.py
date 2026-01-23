#!/usr/bin/env python3
"""Simple test to verify the Transformers backend server can start."""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import aiohttp


async def test_server_startup():
    """Test that a single server can start and respond to /info."""
    model_path = "Qwen/Qwen2-0.5B-Instruct"
    port = 8080
    script_path = Path(__file__).parent / "start_test_server.py"

    print(f"Starting server...")
    print(f"  Model: {model_path}")
    print(f"  Port: {port}")
    print(f"  Device: cpu")

    # Start server
    cmd = [
        sys.executable,
        str(script_path),
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--device-map",
        "cpu",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to be ready
    url = f"http://localhost:{port}"
    start_time = time.time()
    timeout = 180  # 3 minutes for model loading

    print("Waiting for server to be ready...")

    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < timeout:
            try:
                async with session.get(
                    f"{url}/info", timeout=aiohttp.ClientTimeout(total=5.0)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        elapsed = time.time() - start_time
                        print(f"\n✓ Server ready after {elapsed:.1f}s")
                        print(f"  Model path: {data.get('model_path')}")
                        print(f"  Batch size: {data.get('batch_size')}")
                        print(f"  Supports placement: {data.get('supports_placement')}")

                        # Test /health endpoint
                        async with session.get(f"{url}/health") as health_resp:
                            health_data = await health_resp.json()
                            print(f"  Health: {health_data.get('status')}")

                        # Cleanup
                        process.terminate()
                        process.wait(timeout=5)
                        print("\n✓ Test passed!")
                        return True
            except Exception as e:
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    print(f"  Still waiting ({elapsed:.0f}s)...")
                await asyncio.sleep(2.0)

    # Timeout
    print(f"\n✗ Server not ready after {timeout}s")
    process.terminate()
    process.wait(timeout=5)
    return False


if __name__ == "__main__":
    result = asyncio.run(test_server_startup())
    sys.exit(0 if result else 1)
