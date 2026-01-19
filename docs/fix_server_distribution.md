# Fix: Server Distribution in REST Backend

## Problem

The REST backend's `_distribute_batch()` method was not actually distributing requests across multiple servers. All requests were going to the first healthy server, resulting in **zero speedup** when scaling from 1 to 4 servers.

## Root Cause

In [src/place_layers/backend/rest.py](../src/place_layers/backend/rest.py):

1. `_distribute_batch()` (line 654-709) distributed items across servers using round-robin
2. For each server's chunk, it called `process_chunk(server, chunk_items)`
3. **BUG**: `process_chunk` received the assigned `server` parameter but **ignored it**
4. Instead, it called `_call_with_failover(endpoint, payload)` which tried all servers in order
5. All requests went to the first healthy server and returned immediately

### Before Fix

```python
async def process_chunk(server: str, chunk_items: list[tuple[int, Any]]):
    payload = make_payload(chunk_items)
    # BUG: Ignores 'server' parameter!
    data = await self._call_with_failover(endpoint, payload)
    return parse_response(chunk_items, data)
```

**Result**: All requests go to server-0, servers 1-3 idle.

### After Fix

```python
async def process_chunk(server: str, chunk_items: list[tuple[int, Any]]):
    payload = make_payload(chunk_items)
    try:
        # FIX: Call the assigned server directly
        data = await self._call_server(server, endpoint, payload)
        return parse_response(chunk_items, data)
    except RESTBackendError:
        # If assigned server fails, fall back to other servers
        data = await self._call_with_failover(endpoint, payload)
        return parse_response(chunk_items, data)
```

**Result**: Requests distributed across all servers, with failover if a server fails.

## Changes Made

### 1. Added `_call_server()` method (lines 397-444)

New method that calls a specific server directly without trying all servers:

```python
async def _call_server(
    self,
    server: str,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Call a specific server directly without failover."""
    # ... implementation ...
```

Key differences from `_call_with_failover`:
- Takes explicit `server` parameter
- Calls only that server
- Raises error if server fails (no automatic failover)
- Still respects per-server semaphore for concurrency control

### 2. Updated `_distribute_batch()` (lines 682-705)

Modified `process_chunk` to:
1. Try the assigned server first using `_call_server()`
2. Only fall back to `_call_with_failover()` if the assigned server fails
3. Added logging to track assignments and failovers

## Expected Performance Impact

### Before Fix
- 1 server: 100 problems in ~20s
- 4 servers: 100 problems in ~20s (**no speedup!**)
- All requests go to server-0

### After Fix
- 1 server: 100 problems in ~20s
- 4 servers: 100 problems in ~5s (**4x speedup**)
- Requests distributed: server-0 gets 25%, server-1 gets 25%, etc.

With N independent servers (each on separate GPU), expected speedup is **~Nx** (accounting for network overhead).

## Failover Behavior

The fix preserves fault tolerance:

1. **Normal operation**: Each chunk goes directly to its assigned server
2. **Server failure**: If assigned server fails, chunk is retried via `_call_with_failover()` which tries all other servers
3. **Health tracking**: Failed servers are marked unhealthy and avoided in future assignments

## Testing

### Unit Tests
- `tests/test_rest_distribution.py`: Verifies requests are distributed across servers
- `tests/test_rest_distribution.py`: Verifies failover works when assigned server fails

### Verification Scripts
- `scripts/proof_of_bug.py`: Demonstrates the original bug
- `scripts/proof_of_fix.py`: Demonstrates the fix works

### Integration Test
Run with real servers to verify:
```bash
# Check logs for distribution
grep "Assigned.*items to" outputs/latest/app.log
grep "handled by" outputs/latest/app.log

# Should see:
# - N "Assigned" lines (one per server)
# - N "handled by" lines (one per server, not all to server-0)
```

## Backward Compatibility

✅ **Fully backward compatible**

- Same API surface
- Same failover guarantees
- Only changes **which** server handles each request, not **how** it's handled
- Existing configs work unchanged

## Related Issues

This fix resolves the issue where:
- Multiple REST servers were configured but only one was used
- No performance benefit from horizontal scaling
- Server resources underutilized (servers 1-N idle while server-0 overloaded)
