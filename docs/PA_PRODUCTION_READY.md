# PA Orchestrator - Production Ready Configuration

## Summary

The PA orchestrator is now configured for production use with the correct cluster settings and behavior.

## Key Production Features

### 1. Preemptable + Restartable Children ✅

**Configuration:**
```python
job_spec = {
    "preemptable": True,   # Allow cluster to preempt when resources needed
    "restartable": True,   # Cluster auto-restarts when preempted
}
```

**Behavior:**
- Cluster can interrupt (preempt) child jobs when resources are needed elsewhere
- Cluster automatically restarts interrupted jobs
- Same job continues with new IP address
- PA orchestrator transparently handles IP changes
- No manual intervention required

**Benefits:**
- Efficient resource utilization across cluster
- Automatic recovery from preemption
- No duplicate job spawning (cluster handles restarts)
- Seamless to clients (PA updates server list automatically)

### 2. Failure State Handling ✅

**PA Orchestrator Handles:**
- `FAILED` - Job crashed → Spawn new replacement
- `CANCELLED` - Job was cancelled → Spawn new replacement
- `SUCCEEDED` - Job exited unexpectedly → Spawn new replacement

**Cluster Handles Automatically:**
- `INTERRUPTED` - Job was preempted → Cluster restarts same job
- `CANCELLING` - Transient state → Wait for CANCELLED

**Implementation:**
```python
FAILED_STATES = {"CANCELLED", "FAILED", "SUCCEEDED"}
```

### 2.1. Rate Limiting Protection ✅

**Problem:** If a child job fails immediately (e.g., bad image, wrong command), PA could spawn unlimited replacements in a tight loop, exhausting cluster resources.

**Solution:** Rate limiting with multiple safeguards:

1. **No duplicate spawning**: Before spawning a replacement, check if we already have a job at this index in QUEUING/QUEUED/RUNNING state
   - Checks both tracked jobs (in orchestrator's memory)
   - AND queries PA API for all jobs (including untracked ones without IDs yet)
2. **Job discovery**: Automatically discovers jobs that were spawned without IDs initially by parsing command strings
3. **Minimum spawn interval**: Enforce minimum time (default: 60 seconds) between spawn attempts for the same child index

**Configuration:**
```bash
python scripts/submit_pa_orchestrator.py \
    --min-spawn-interval 120  # 2 minutes between spawn attempts
```

**Behavior:**
```
1. Child 0 fails (FAILED state)
2. PA spawns replacement immediately
3. Replacement fails again within 60 seconds
4. PA detects failure but waits (rate limited)
5. After 60 seconds, PA spawns another replacement
6. This prevents rapid loop: job → fail → spawn → fail → spawn...
```

**Logs:**
```
2024-01-15 10:00:00 - Child 0 (job abc-123) failed with state: FAILED
2024-01-15 10:00:00 - Spawning replacement for child 0
2024-01-15 10:00:30 - Child 0 (job def-456) failed with state: FAILED
2024-01-15 10:00:30 - Not spawning replacement for child 0 yet - rate limited (min interval: 60s)
2024-01-15 10:01:00 - Child 0 (job def-456) failed with state: FAILED
2024-01-15 10:01:00 - Spawning replacement for child 0
```

**Benefits:**
- Prevents resource exhaustion from misconfigured jobs
- Gives time to investigate failures
- Prevents overwhelming the cluster scheduler
- Configurable per deployment

**Important PA API Behaviors:**
1. **Jobs without IDs**: When you spawn a job via `POST /api/v1/jobs`, it might not have an `id` field until it reaches QUEUING state
2. **PA never deletes jobs**: The `GET /api/v1/` endpoint returns ALL jobs ever spawned (including FAILED, CANCELLED, SUCCEEDED)
3. **Multiple jobs per index**: After respawns, PA API will show n (requested) + m (respawned) total jobs
4. **Job discovery**: PA orchestrator discovers untracked jobs by parsing port numbers from command strings

**Example:**
```
Initial: Request 4 children
PA API shows: 4 jobs (indices 0-3)

Child 0 fails, PA spawns replacement
PA API shows: 5 jobs (original 4 + 1 replacement)
  - job-abc (index 0, FAILED)
  - job-xyz (index 0, QUEUING)  ← replacement
  - job-123 (index 1, RUNNING)
  - job-456 (index 2, RUNNING)
  - job-789 (index 3, RUNNING)
```

### 3. Correct Cluster Image ✅

**Default Image:**
```
registry.toolkit-sp.yul201.service-now.com/snow.research.afm/interactive-toolkit:25.12-py3-vllm014rc1
```

**Contents:**
- Python 3
- vLLM 0.14 RC1
- All required dependencies for serving
- Tested and validated on cluster

### 4. Smart Configuration Merge ✅

**Supports Both Formats:**

**PA API Format (used directly):**
```yaml
environmentVars:
  - HOME=/home/toolkit
  - RESEARCH_INTERACTIVE_TOOLKIT=1
resources:
  cpu: 4
  mem: 16
  gpu: 2
  gpuMem: 12
```

**User-Friendly Format (converted automatically):**
```yaml
env:
  PYTHONPATH: /path
  LOG_LEVEL: INFO
cpu: 4
mem: 16
```

**Mixed Format (both respected):**
```yaml
# PA API format (used directly)
environmentVars:
  - HOME=/home/toolkit
resources:
  cpu: 4
  mem: 16

# User-friendly format (appended/converted)
env:
  PYTHONPATH: /path
```

**Merge Logic:**
- If PA API fields exist, use them directly
- If user-friendly fields exist, convert them
- If both exist, append/merge appropriately
- PA config (from `pa_defaults.yaml`) applied last

## Production Deployment

### 1. Create User Job Spec

```yaml
# my_job_spec.yaml
data:
  - snow.home.myuser:/home/toolkit:rw
  - snow.research.fml.datasets:/mnt/datasets:rw

environmentVars:
  - HOME=/home/toolkit
  - RESEARCH_INTERACTIVE_TOOLKIT=1

image: registry.toolkit-sp.yul201.service-now.com/snow.research.afm/interactive-toolkit:25.12-py3-vllm014rc1

resources:
  cpu: 4
  mem: 16
  gpu: 2
  gpuMem: 12
```

### 2. Submit PA Orchestrator

```bash
python scripts/submit_pa_orchestrator.py \
    --job-spec my_job_spec.yaml \
    --model-name apriel2_0_5b_dev \
    --num-children 4 \
    --child-mem 32 \
    --child-gpu 1 \
    --health-check-interval 30 \
    --min-spawn-interval 60
```

**Key Parameters:**
- `--health-check-interval`: How often (seconds) PA checks child status (default: 30)
- `--min-spawn-interval`: Minimum seconds between spawn attempts for same child (default: 60)

### 3. Wait for PA URL

Script will output:
```
============================================================
PA Orchestrator is ready!
============================================================
Job ID: abc-123-def
PA URL: http://10.20.30.40:8080
============================================================
```

### 4. Use in Evaluation

```bash
python -m place_layers \
    +command=evaluate \
    +backend=pa_rest \
    backend.pa_url='http://10.20.30.40:8080' \
    +evaluator=generative \
    +task=gsm8k \
    +generation=greedy
```

## Operational Scenarios

### Scenario 1: Normal Operation

```
PA Orchestrator: RUNNING
  ├─ Child 0: RUNNING on 10.20.30.41:8080
  ├─ Child 1: RUNNING on 10.20.30.42:8081
  ├─ Child 2: RUNNING on 10.20.30.43:8082
  └─ Child 3: RUNNING on 10.20.30.44:8083

GET /servers → ["http://10.20.30.41:8080", ...]
```

**Actions:** None - everything healthy

### Scenario 2: Cluster Preempts Child

```
1. Cluster needs resources
2. Child 2 transitions: RUNNING → INTERRUPTED
3. Cluster automatically restarts: INTERRUPTED → QUEUING → QUEUED → RUNNING
4. Child 2 now on different IP: 10.20.30.50:8082
5. PA polls, sees new IP, updates tracking
6. Next GET /servers includes new IP
7. Clients fetch updated list and continue
```

**Actions:**
- PA: Update IP in tracking
- Cluster: Auto-restart job
- Clients: Fetch updated server list

### Scenario 3: Child Crashes

```
1. Child 1 crashes (out of memory, bug, etc.)
2. Job transitions: RUNNING → FAILED
3. PA detects FAILED state
4. PA spawns NEW child job at index 1
5. New job: QUEUING → QUEUED → RUNNING on 10.20.30.51:8081
6. Next GET /servers includes new IP
```

**Actions:**
- PA: Spawn replacement job
- Clients: Fetch updated server list

### Scenario 4: Child Exits Cleanly

```
1. Child 3 exits with code 0 (bug in serve code)
2. Job transitions: RUNNING → SUCCEEDED
3. PA detects SUCCEEDED state (should never happen for serve jobs)
4. PA spawns NEW child job at index 3
5. New job: QUEUING → QUEUED → RUNNING
```

**Actions:**
- PA: Spawn replacement job
- Investigate why serve job exited

### Scenario 5: User Cancels Child

```
1. User runs: eai job cancel <child-job-id>
2. Job transitions: RUNNING → CANCELLING → CANCELLED
3. PA detects CANCELLED state
4. PA spawns NEW child job
```

**Actions:**
- PA: Spawn replacement job
- Note: If intentional, may want to reduce num_children instead

### Scenario 6: Child Fails Repeatedly (Rate Limiting)

```
1. Child 2 has bad configuration (e.g., wrong image)
2. Job spawned: QUEUING → FAILED (image pull error)
3. PA detects FAILED, spawns replacement immediately
4. Replacement also fails within 10 seconds: QUEUING → FAILED
5. PA detects FAILED but rate limiting applies
6. PA logs: "Not spawning replacement for child 2 yet - rate limited"
7. PA waits min_spawn_interval seconds (default: 60)
8. After 60 seconds, PA spawns another replacement
9. Admin investigates logs, fixes configuration
10. Next spawn succeeds
```

**Actions:**
- PA: Apply rate limiting to prevent resource exhaustion
- Admin: Check PA logs for rate limit warnings
- Admin: Check child logs to identify root cause
- Admin: Fix configuration (image, resources, etc.)
- PA: Auto-recover once configuration is fixed

## Monitoring and Debugging

### Check PA Orchestrator Status

```bash
# Get PA health
curl http://PA_IP:8080/health

# Response:
{
  "status": "ok",
  "active_children": 4,
  "target_children": 4
}
```

### Check Active Servers

```bash
# Get server list
curl http://PA_IP:8080/servers

# Response:
{
  "servers": [
    "http://10.20.30.41:8080",
    "http://10.20.30.42:8081",
    "http://10.20.30.43:8082",
    "http://10.20.30.44:8083"
  ]
}
```

### Check PA Logs

```bash
# PA orchestrator logs
eai job logs <pa-job-id>

# Look for:
# - "Spawning child X" - New children being created
# - "Child X (job Y) state: A -> B" - State transitions
# - "Child X (job Y) failed with state: Z" - Failures detected
# - "Spawning replacement for child X" - Replacements being spawned
```

### Check Child Job Status

```bash
# Via PA API (inside PA job)
curl http://0.0.0.0:8666/api/v1/jobs/<child-job-id>

# Via eai
eai job show <child-job-id>
```

## Troubleshooting

### Issue: Server List Empty

**Symptoms:**
```bash
curl http://PA_IP:8080/servers
{"servers": []}
```

**Possible Causes:**
1. Children still starting (QUEUING/QUEUED state)
2. All children failed
3. Image not available
4. Resource constraints

**Debug:**
```bash
# Check PA logs
eai job logs <pa-job-id>

# Check if children exist
curl http://PA_IP:8080/health
# If target_children > 0 but active_children = 0, children are failing

# Check child logs
eai job logs <child-job-id>
```

### Issue: Frequent Restarts

**Symptoms:**
- PA logs show many "Spawning replacement" messages
- Children transition to FAILED/SUCCEEDED frequently

**Possible Causes:**
1. Out of memory (increase child_mem)
2. Model loading failures (check image/model path)
3. GPU memory insufficient (check child GPU allocation)
4. Serve command incorrect

**Debug:**
```bash
# Check child logs for errors
eai job logs <child-job-id>

# Check resource allocation
eai job show <child-job-id>
```

### Issue: IP Not Updating After Preemption

**Symptoms:**
- Child shows RUNNING but wrong IP in /servers

**Possible Causes:**
- Health check interval too long
- PA not polling properly

**Debug:**
```bash
# Force health check by waiting health_check_interval seconds
# Default is 30 seconds

# Check PA logs
eai job logs <pa-job-id>
# Look for "Child X (job Y) state: INTERRUPTED -> RUNNING"
```

### Issue: Rate Limiting Triggered

**Symptoms:**
- PA logs show "Not spawning replacement for child X yet - rate limited"
- Server list incomplete for extended period
- Specific child index keeps failing

**Possible Causes:**
1. Bad configuration causing immediate failures (wrong image, bad command)
2. Insufficient resources (out of memory, GPU allocation too small)
3. Model loading errors

**Debug:**
```bash
# Check why the child is failing
eai job logs <child-job-id>

# Look for:
# - Image pull errors
# - Out of memory errors
# - Model loading failures
# - Command syntax errors
```

**Actions:**
1. **Fix the root cause** (don't just increase spawn interval):
   - Verify image is correct and accessible
   - Check resource allocation matches model requirements
   - Verify model path and accessibility
   - Check command syntax

2. **Adjust interval if needed**:
   ```bash
   # If legitimate transient failures, increase interval
   python scripts/submit_pa_orchestrator.py \
       --min-spawn-interval 300  # 5 minutes
   ```

3. **Temporary workaround**:
   ```bash
   # Cancel PA job and resubmit with fixed configuration
   eai job cancel <pa-job-id>

   # Fix configuration and resubmit
   python scripts/submit_pa_orchestrator.py \
       --job-spec fixed_job_spec.yaml \
       ...
   ```

## Production Checklist

Before deploying to production:

- [ ] PA defaults config (`config/pa_defaults.yaml`) is correct
- [ ] User job spec has correct data mounts
- [ ] User job spec has correct environment variables
- [ ] Image is accessible from cluster nodes
- [ ] Model path is correct and accessible
- [ ] Resource allocation is appropriate (mem, GPU)
- [ ] Health check interval is appropriate for workload (default: 30s)
- [ ] Rate limiting interval is appropriate for failure patterns (default: 60s)
- [ ] Tested with single child first
- [ ] Tested preemption scenario (manually cancel child)
- [ ] Tested failure scenario (manually kill child process)
- [ ] Tested rate limiting (provide bad config, verify PA doesn't spawn unlimited jobs)
- [ ] Verified client can connect to PA URL
- [ ] Verified client can use servers from /servers endpoint

## Support

For issues:

1. Check PA orchestrator logs
2. Check child job logs
3. Verify PA health endpoint
4. Verify server list endpoint
5. Check cluster resource availability
6. Consult this document for common scenarios
