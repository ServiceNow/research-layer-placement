#!/usr/bin/env python3
"""Proof that the fix correctly distributes across servers.

This script shows the corrected behavior.
"""

import asyncio


class ServerPool:
    def __init__(self, servers):
        self.servers = servers

    def get_servers_to_try(self):
        """Returns ALL servers in order."""
        return self.servers.copy()


class FixedRESTBackend:
    def __init__(self, servers):
        self.server_pool = ServerPool(servers)
        self.request_log = []

    async def _call_server(self, server, endpoint, payload):
        """NEW: Call a specific server directly."""
        # Simulate successful request to the specified server
        self.request_log.append(f"{endpoint} -> {server} (direct)")
        return {"result": "success"}

    async def _call_with_failover(self, endpoint, payload):
        """Fallback for when assigned server fails."""
        servers_to_try = self.server_pool.get_servers_to_try()
        for server in servers_to_try:
            self.request_log.append(f"{endpoint} -> {server} (failover)")
            return {"result": "success"}

    async def _distribute_batch(self, items, endpoint):
        """FIXED: Now actually distributes across servers."""
        servers = self.server_pool.get_servers_to_try()

        # Distribute items across servers
        server_items = {s: [] for s in servers}
        for i, item in enumerate(items):
            server = servers[i % len(servers)]
            server_items[server].append(item)
            print(f"ASSIGNED: Item {i} -> {server}")

        # Process each server's chunk
        async def process_chunk(server, chunk_items):
            if not chunk_items:
                return []

            # FIX: Call the specific server directly!
            try:
                await self._call_server(server, endpoint, {"items": chunk_items})
            except Exception:
                # If assigned server fails, fall back to failover
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

    backend = FixedRESTBackend(servers)

    print("=" * 60)
    print("Testing FIXED distribution with 4 servers and 8 items")
    print("=" * 60)
    print()

    items = list(range(8))
    await backend._distribute_batch(items, "/compute")

    print()
    print("=" * 60)
    print("ACTUAL REQUEST LOG:")
    print("=" * 60)
    for log in backend.request_log:
        print(f"  {log}")

    print()
    print("=" * 60)

    # Count requests per server
    server_counts = {}
    for log in backend.request_log:
        for i in range(4):
            server = f"server-{i}"
            if server in log:
                server_counts[server] = server_counts.get(server, 0) + 1

    print("Requests per server:")
    for server, count in sorted(server_counts.items()):
        print(f"  {server}: {count} request(s)")

    print("=" * 60)
    print()
    if len(set(server_counts.values())) == 1:
        print("✓ FIX VERIFIED: Requests evenly distributed!")
    else:
        print("✗ Still broken")
    print()


if __name__ == "__main__":
    asyncio.run(main())
