"""Test that REST backend correctly distributes requests across servers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from place_layers.backend.rest import RESTBackend, RESTBackendConfig


@pytest.mark.asyncio
async def test_distribute_batch_uses_assigned_servers():
    """Verify that _distribute_batch sends requests to assigned servers, not all to first server."""

    config = RESTBackendConfig(
        servers=[
            "http://server-0:8080",
            "http://server-1:8080",
            "http://server-2:8080",
            "http://server-3:8080",
        ],
        max_concurrent_per_server=8,
    )

    backend = RESTBackend(config)

    # Track which server each request goes to
    request_log = []

    async def mock_call_server(server, endpoint, payload):
        """Mock _call_server to track which server is called."""
        request_log.append(server)
        return {"results": [{"token_ids": [], "logprobs": [], "answer_range": [0, 1], "total_log_prob": 0.0}] * len(payload.get("chats", []))}

    # Patch _call_server
    with patch.object(backend, "_call_server", side_effect=mock_call_server):
        # Simulate entering context manager
        backend._session = MagicMock()
        backend._server_semaphores = {
            server: asyncio.Semaphore(8) for server in config.servers
        }

        # Create 8 dummy items to distribute
        items = [(i, ([], [])) for i in range(8)]

        # Call _distribute_batch
        await backend._distribute_batch(
            items=items,
            endpoint="/compute_chat_log_probs",
            make_payload=lambda chunk_items: {
                "chats": [chat for _, chat in chunk_items],
            },
            parse_response=lambda chunk_items, data: [
                (idx, data["results"][i])
                for i, (idx, _) in enumerate(chunk_items)
            ],
        )

    # Verify requests were distributed across all 4 servers
    assert len(request_log) == 4, f"Expected 4 requests, got {len(request_log)}"

    # Count requests per server
    server_counts = {server: request_log.count(server) for server in config.servers}

    print(f"\nRequests per server: {server_counts}")

    # Each server should get exactly 1 request (8 items / 4 servers = 2 items per server, 1 request per chunk)
    for server, count in server_counts.items():
        assert count == 1, f"Server {server} got {count} requests, expected 1"

    # Verify we hit all 4 different servers
    assert len(set(request_log)) == 4, f"Expected 4 unique servers, got {set(request_log)}"

    print("✓ Requests correctly distributed across all servers!")


@pytest.mark.asyncio
async def test_distribute_batch_failover_when_server_fails():
    """Verify that _distribute_batch falls back to failover if assigned server fails."""

    config = RESTBackendConfig(
        servers=[
            "http://server-0:8080",
            "http://server-1:8080",
        ],
    )

    backend = RESTBackend(config)

    call_log = []

    async def mock_call_server(server, endpoint, payload):
        """Mock that fails for server-1."""
        call_log.append(("call_server", server))
        if server == "http://server-1:8080":
            from place_layers.backend.rest import RESTBackendError
            raise RESTBackendError(f"Server {server} failed")
        return {"results": [{"token_ids": [], "logprobs": [], "answer_range": [0, 1], "total_log_prob": 0.0}] * len(payload.get("chats", []))}

    async def mock_call_with_failover(endpoint, payload):
        """Mock failover."""
        call_log.append(("call_with_failover", "fallback"))
        return {"results": [{"token_ids": [], "logprobs": [], "answer_range": [0, 1], "total_log_prob": 0.0}] * len(payload.get("chats", []))}

    with patch.object(backend, "_call_server", side_effect=mock_call_server):
        with patch.object(backend, "_call_with_failover", side_effect=mock_call_with_failover):
            backend._session = MagicMock()
            backend._server_semaphores = {
                server: asyncio.Semaphore(8) for server in config.servers
            }

            # 4 items distributed across 2 servers
            items = [(i, ([], [])) for i in range(4)]

            await backend._distribute_batch(
                items=items,
                endpoint="/compute_chat_log_probs",
                make_payload=lambda chunk_items: {"chats": [chat for _, chat in chunk_items]},
                parse_response=lambda chunk_items, data: [
                    (idx, data["results"][i])
                    for i, (idx, _) in enumerate(chunk_items)
                ],
            )

    print(f"\nCall log: {call_log}")

    # Verify server-0 was called directly and succeeded
    assert ("call_server", "http://server-0:8080") in call_log

    # Verify server-1 was tried directly but failed, then failover was used
    assert ("call_server", "http://server-1:8080") in call_log
    assert ("call_with_failover", "fallback") in call_log

    print("✓ Failover works correctly when assigned server fails!")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_distribute_batch_uses_assigned_servers())
    asyncio.run(test_distribute_batch_failover_when_server_fails())
