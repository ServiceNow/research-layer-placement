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

    # Track which server each request goes to (track preferred_server from _call_with_failover)
    request_log = []

    original_call_with_failover = backend._call_with_failover

    async def mock_call_with_failover(endpoint, payload, preferred_server=None):
        """Mock _call_with_failover to track preferred server."""
        request_log.append(preferred_server)
        return {"results": [{"token_ids": [], "logprobs": [], "answer_range": [0, 1], "total_log_prob": 0.0}] * len(payload.get("chats", []))}

    # Patch _call_with_failover
    with patch.object(backend, "_call_with_failover", side_effect=mock_call_with_failover):
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
async def test_distribute_batch_uses_preferred_server():
    """Verify that _distribute_batch passes preferred_server to _call_with_failover."""

    config = RESTBackendConfig(
        servers=[
            "http://server-0:8080",
            "http://server-1:8080",
        ],
    )

    backend = RESTBackend(config)

    preferred_servers_log = []

    async def mock_call_with_failover(endpoint, payload, preferred_server=None):
        """Track which preferred_server is passed."""
        preferred_servers_log.append(preferred_server)
        return {"results": [{"token_ids": [], "logprobs": [], "answer_range": [0, 1], "total_log_prob": 0.0}] * len(payload.get("chats", []))}

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

    print(f"\nPreferred servers: {preferred_servers_log}")

    # Verify both servers were used as preferred servers
    assert "http://server-0:8080" in preferred_servers_log
    assert "http://server-1:8080" in preferred_servers_log
    assert len(preferred_servers_log) == 2

    print("✓ Preferred servers correctly passed to _call_with_failover!")


@pytest.mark.asyncio
async def test_call_with_failover_ignores_unhealthy_preferred():
    """Verify that unhealthy preferred servers are ignored."""
    from place_layers.backend.rest import RESTBackend, RESTBackendConfig

    config = RESTBackendConfig(
        servers=[
            "http://server-0:8080",
            "http://server-1:8080",
        ],
    )

    backend = RESTBackend(config)

    # Mark server-1 as unhealthy
    backend._server_pool._health["http://server-1:8080"].mark_unhealthy()

    called_servers = []

    async def mock_do_http_request(server, endpoint, payload):
        """Track which server is actually called."""
        called_servers.append(server)
        return {"result": "success"}

    with patch.object(backend, "_do_http_request", side_effect=mock_do_http_request):
        backend._session = MagicMock()
        backend._server_semaphores = {
            server: asyncio.Semaphore(8) for server in config.servers
        }

        # Try to call with unhealthy server-1 as preferred
        await backend._call_with_failover(
            "/test",
            {"data": "test"},
            preferred_server="http://server-1:8080"
        )

    print(f"\nCalled servers: {called_servers}")

    # Should only call server-0 (the only healthy server)
    # server-1 should be ignored even though it was preferred
    assert called_servers == ["http://server-0:8080"]
    print("✓ Unhealthy preferred server correctly ignored!")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_distribute_batch_uses_assigned_servers())
    asyncio.run(test_distribute_batch_uses_preferred_server())
    asyncio.run(test_call_with_failover_ignores_unhealthy_preferred())
