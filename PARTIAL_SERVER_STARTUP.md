# Partial Server Availability at Startup

The REST backend now supports starting evaluation with only a subset of configured servers available.

## What Changed

Previously, the REST backend required **all** configured servers to be healthy during initialization. If any server failed to respond to the `/info` health check, the entire initialization would fail with an error like:

```
Failed to connect to 2/3 servers: http://server1:8080: Connection refused, http://server2:8080: Connection refused
```

Now, the backend will:
1. Start successfully if **at least one** server is reachable
2. Mark unreachable servers as unhealthy
3. Log a warning about unavailable servers
4. Allow those servers to be retried and recovered during evaluation using the existing retry mechanism

## Example

### Configuration
```yaml
backend:
  backend_type: rest
  servers:
    - http://server1:8080
    - http://server2:8080
    - http://server3:8080
```

### Scenario: Two Servers Down at Start

```
2026-01-20 10:00:00 WARNING Server http://server1:8080 marked unhealthy (failures: 1, next check in 10.0s)
2026-01-20 10:00:00 WARNING Server http://server3:8080 marked unhealthy (failures: 1, next check in 10.0s)
2026-01-20 10:00:00 WARNING Failed to connect to 2/3 servers during initialization:
    http://server1:8080: Connection refused,
    http://server3:8080: HTTP 503.
    These servers will be retried during evaluation.
2026-01-20 10:00:00 INFO REST backend initialized with 1/3 healthy servers, max 24 concurrent requests
2026-01-20 10:00:00 INFO Model structure: 24 layers, mixers: ('attention', 'gdn')
```

The evaluation will proceed with only `server2`, and the backend will automatically retry the unhealthy servers according to the recovery schedule (exponential backoff from `base_recovery_interval` to `max_recovery_interval`).

## Benefits

1. **More robust startup**: Temporary network issues or slow server startup won't block evaluation
2. **Graceful degradation**: Evaluation can proceed with reduced capacity rather than failing completely
3. **Automatic recovery**: Servers that become available later are automatically integrated
4. **No configuration changes needed**: Works with existing backend configurations

## Error Handling

The backend will still fail during initialization if:
- **All** servers are unreachable
- Reachable servers have inconsistent configurations (different models, different layer counts, etc.)

This ensures that evaluation only proceeds when at least one server is available and all reachable servers are compatible.

## Configuration Options

The recovery behavior is controlled by existing configuration options:

```yaml
backend:
  backend_type: rest
  servers: [...]
  base_recovery_interval: 10.0   # Initial retry delay (seconds)
  max_recovery_interval: 300.0   # Maximum retry delay (seconds)
  max_retries: 10                # Max attempts when all servers fail
```

Unhealthy servers are retried with exponential backoff: 10s, 20s, 40s, 80s, 160s, 300s (capped), 300s, ...
