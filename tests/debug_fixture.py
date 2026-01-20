#!/usr/bin/env python3
"""Debug script to test the fixture directly."""

import asyncio
import os
from pathlib import Path

# Simulate what the fixture does
model_path = os.getenv("TEST_MODEL_PATH", "Qwen/Qwen2-0.5B-Instruct")
device_map = os.getenv("TEST_DEVICE_MAP", "cpu")
ports = [8080, 8081]

print(f"Setting up test servers:")
print(f"  Model: {model_path}")
print(f"  Device map: {device_map}")
print(f"  Ports: {ports}")

# Import the ServerManager
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_likelihood_rest_integration import ServerManager

manager = ServerManager(model_path, ports, device_map)

print("\nStarting servers (this may take 30-60 seconds)...")
urls = asyncio.run(manager.start())

print(f"\n✓ Servers started successfully!")
print(f"  URLs: {urls}")

print("\nShutting down...")
manager.stop()
print("✓ Done!")
