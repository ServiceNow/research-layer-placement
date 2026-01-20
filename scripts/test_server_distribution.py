#!/usr/bin/env python3
"""Test script to verify server distribution behavior.

This script will show which servers actually handle requests when using
_distribute_batch with multiple servers.
"""

import asyncio
import logging

from place_layers.backend.rest import RESTBackendConfig
from place_layers.evaluator.likelihood import LikelihoodEvaluatorConfig

# Set up logging to see our diagnostic messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)

async def test_distribution():
    """Test distribution with 4 servers."""

    # Use the same servers as in outputs1
    servers = [
        "http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-0:8080",
        "http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-1:8080",
        "http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-2:8080",
        "http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-3:8080",
    ]

    config = LikelihoodEvaluatorConfig(
        backend=RESTBackendConfig(
            servers=servers,
            max_retries=3,
            timeout_seconds=300,
        ),
        tasks=["mmlu_abstract_algebra"],
        num_fewshot=5,
        batch_size=100,  # Single batch to see distribution clearly
        max_problems_per_task=20,  # Just 20 problems for quick test
    )

    from place_layers.evaluator.likelihood import LikelihoodEvaluator

    evaluator = LikelihoodEvaluator(config)

    print("\n" + "="*80)
    print("Testing server distribution with 4 servers and batch_size=100")
    print("Expected: 4 assignments, 20 items each (20 problems × 4 choices = 80 chats)")
    print("Expected: 4 'handled by' messages, one for each server")
    print("="*80 + "\n")

    async with evaluator:
        await evaluator.evaluate(None)

    print("\n" + "="*80)
    print("Check the logs above:")
    print("- Look for '_distribute_batch: Assigned X items to <server>'")
    print("- Look for 'Request to /compute_chat_log_probs handled by <server>'")
    print("- If all requests go to id-0, the bug is confirmed!")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_distribution())
