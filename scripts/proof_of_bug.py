#!/usr/bin/env python3
"""Minimal proof that _distribute_batch doesn't actually distribute.

This script shows the logical flaw without needing to run actual servers.
"""

import asyncio


# Simplified version of the problematic code
class ServerPool:
    def __init__(self, servers):
        self.servers = servers

    def get_servers_to_try(self):
        """Returns ALL servers in order."""
        return self.servers.copy()  # Returns all servers


class SimplifiedRESTBackend:
    def __init__(self, servers):
        self.server_pool = ServerPool(servers)
        self.request_log = []  # Track which server handled each request

    async def _call_with_failover(self, endpoint, payload):
        """This is the problematic method - it tries servers in order."""
        servers_to_try = self.server_pool.get_servers_to_try()

        for server in servers_to_try:
            # Simulate successful request to first server
            self.request_log.append(f"{endpoint} -> {server}")
            return {"result": "success"}  # Returns immediately on first success

    async def _distribute_batch(self, items, endpoint):
        """This TRIES to distribute but fails due to _call_with_failover."""
        servers = self.server_pool.get_servers_to_try()

        # Distribute items across servers (round-robin)
        server_items = {s: [] for s in servers}
        for i, item in enumerate(items):
            server = servers[i % len(servers)]
            server_items[server].append(item)
            print(f"ASSIGNED: Item {i} -> {server}")

        # Process each server's chunk
        async def process_chunk(server, chunk_items):
            if not chunk_items:
                return []
            # BUG: Calls _call_with_failover WITHOUT specifying which server to use!
            # The 'server' parameter is IGNORED
            await self._call_with_failover(endpoint, {"items": chunk_items})
            return []

        tasks = [
            process_chunk(server, chunk_items)
            for server, chunk_items in server_items.items()
            if chunk_items
        ]

        await asyncio.gather(*tasks)


async def main():
    servers = [
        "http://server-0:8080",
        "http://server-1:8080",
        "http://server-2:8080",
        "http://server-3:8080",
    ]

    backend = SimplifiedRESTBackend(servers)

    print("=" * 60)
    print("Testing distribution with 4 servers and 8 items")
    print("=" * 60)
    print()

    items = list(range(8))  # 8 items to distribute
    await backend._distribute_batch(items, "/compute")

    print()
    print("=" * 60)
    print("ACTUAL REQUEST LOG:")
    print("=" * 60)
    for log in backend.request_log:
        print(f"  {log}")

    print()
    print("=" * 60)
    print("BUG CONFIRMED!" if all("server-0" in log for log in backend.request_log) else "Bug not present")
    print("=" * 60)
    print()
    print("Expected behavior: 4 requests distributed across 4 servers")
    print(f"Actual behavior: {len(backend.request_log)} requests all to server-0")
    print()
    print("Why: _call_with_failover() always tries server-0 first,")
    print("     and returns immediately on success, ignoring the")
    print("     'server' parameter passed to process_chunk().")


if __name__ == "__main__":
    asyncio.run(main())
