# Verification Plan: Server Distribution Bug

## The Bug Hypothesis

When `_distribute_batch()` distributes work across N servers:
1. It assigns chunks to specific servers (round-robin)
2. It launches N async tasks via `asyncio.gather()`
3. **BUT**: Each task calls `_call_with_failover()` which tries ALL servers starting with server-0
4. **Result**: All N tasks send their requests to server-0 (the first healthy server)
5. Servers 1,2,3,...,N-1 remain idle

## Code Evidence

### Evidence 1: _distribute_batch assigns servers but doesn't use them

```python
# Line 629-630: Assign items to servers
for i, item in enumerate(items):
    server = servers[i % len(servers)]  # <- Assigns to specific server
    server_items[server].append(item)

# Line 634-646: process_chunk RECEIVES server parameter
async def process_chunk(server: str, chunk_items: list[tuple[int, Any]]):
    # ...
    data = await self._call_with_failover(endpoint, payload)  # <- IGNORES server parameter!
```

### Evidence 2: _call_with_failover tries all servers

```python
# Line 342: Get ALL healthy servers
servers_to_try = self._server_pool.get_servers_to_try()

# Line 350: Try servers in order until one succeeds
for server in servers_to_try:
    # Try server, if success, return immediately
```

### Evidence 3: get_servers_to_try returns all servers

```python
# Line 122-140: Returns ALL healthy servers in order
def get_servers_to_try(self) -> list[str]:
    healthy: list[str] = []
    for server in self.servers:
        if health.healthy:
            healthy.append(server)  # <- All healthy servers
    return healthy  # <- Returns [server-0, server-1, server-2, ...]
```

## Verification Methods

### Method 1: Add logging (DONE)

We added:
- Log in `process_chunk`: Shows which server was ASSIGNED
- Log in `_call_with_failover`: Shows which server ACTUALLY handled request

Run any evaluation and grep logs for:
```
grep "Assigned.*items to" app.log
grep "handled by" app.log
```

Expected if bug exists:
```
Assigned 100 items to http://...id-0:8080
Assigned 100 items to http://...id-1:8080
Assigned 100 items to http://...id-2:8080
Assigned 100 items to http://...id-3:8080
Request to /compute_chat_log_probs handled by http://...id-0:8080
Request to /compute_chat_log_probs handled by http://...id-0:8080  <- BUG: All go to id-0!
Request to /compute_chat_log_probs handled by http://...id-0:8080
Request to /compute_chat_log_probs handled by http://...id-0:8080
```

### Method 2: Server metrics

If servers expose metrics (request count), check:
- Server-0: Should have N requests (where N = number of servers)
- Server-1,2,3: Should have 0 requests (if bug exists)

### Method 3: Temporarily break server-0

If we make server-0 unavailable, `_call_with_failover` should failover to server-1.
Expected result if bug exists:
- All requests now go to server-1 instead
- Still no distribution across servers 1,2,3,4

## Conclusion

The bug is **architecturally certain** based on code inspection:
1. `process_chunk(server, ...)` receives a `server` parameter
2. But it calls `_call_with_failover()` which **ignores** that parameter
3. `_call_with_failover()` always tries servers in order: [0, 1, 2, 3, ...]
4. First healthy server (server-0) handles all requests

**Confidence: 99%** (The 1% is for the possibility of some async timing I'm missing, but it's highly unlikely)
