# PA Orchestrator Changelog

## Latest Changes (Based on Production Requirements)

### Rate Limiting Protection (New Feature)

#### 4. Spawn Rate Limiting
**Added**: Protection against rapid respawning of failed children

**Problem**: If a child job has a bad configuration (e.g., wrong image, insufficient memory), it can fail immediately after spawning. Without protection, PA could spawn unlimited replacements in a tight loop, exhausting cluster resources.

**Solution**: Two-layer rate limiting:

1. **Duplicate prevention**: Don't spawn if we already have a job at this index in QUEUING/QUEUED state
2. **Time-based limiting**: Enforce minimum interval (default: 60 seconds) between spawn attempts for same child index

**Configuration:**
```python
# In PAOrchestratorCommandConfig
min_spawn_interval: int = Field(
    default=60,
    description="Minimum seconds between spawn attempts for same child index",
    ge=1,
)
```

**Usage:**
```bash
python scripts/submit_pa_orchestrator.py \
    --min-spawn-interval 120  # 2 minutes between spawns for same child
```

**Implementation:**
```python
def _should_spawn_replacement(self, index: int) -> bool:
    """Check if we should spawn a replacement for the given index."""
    # 1. Check for existing QUEUING/QUEUED jobs
    for child in self._children.values():
        if child["index"] == index and child["state"] in {"QUEUING", "QUEUED"}:
            return False

    # 2. Check time since last spawn
    last_spawn = self._last_spawn_time.get(index, 0)
    time_since_spawn = time.time() - last_spawn

    if time_since_spawn < self._min_spawn_interval:
        return False

    return True
```

**Example Scenario:**
```
10:00:00 - Child 0 fails (FAILED state)
10:00:00 - PA spawns replacement immediately
10:00:05 - Replacement also fails (bad config)
10:00:05 - PA wants to spawn but is rate limited
10:00:05 - Log: "Not spawning replacement for child 0 yet - rate limited (min interval: 60s)"
10:01:00 - 60 seconds elapsed
10:01:00 - PA spawns replacement again
```

**Benefits:**
- Prevents resource exhaustion from misconfigured jobs
- Gives time for admin to investigate failures
- Prevents overwhelming cluster scheduler
- Configurable per deployment needs

**Files Updated:**
- `src/place_layers/command/pa_orchestrator.py` - Added rate limiting logic
- `scripts/submit_pa_orchestrator.py` - Added `--min-spawn-interval` argument
- `docs/PA_PRODUCTION_READY.md` - Documented rate limiting behavior
- `docs/CHANGELOG_PA.md` - Added this entry

### 4.1. Robust Job Tracking for PA API Edge Cases

#### Additional Rate Limiting Improvements

**Issues Addressed:**
1. **Jobs without IDs**: PA API might not return job `id` field until job reaches QUEUING state
2. **PA never deletes jobs**: `GET /api/v1/` returns ALL jobs (including FAILED/CANCELLED/SUCCEEDED), not just active ones
3. **Multiple jobs per index**: After respawns, PA API shows n (requested) + m (respawned) total jobs
4. **Untracked job handling**: Jobs spawned without IDs need to be discovered later

**Solutions Implemented:**

1. **Graceful handling of missing job IDs:**
```python
job_id = job_info.get("id")  # Might be None if not QUEUING yet

# Record spawn time immediately (even if no job_id yet)
self._last_spawn_time[index] = time.time()

# Track only if we have a job_id
if job_id:
    self._children[job_id] = {...}
else:
    logger.info(f"Spawned child {index}: job not yet in QUEUING state (no ID)")
```

2. **Job discovery from PA API:**
```python
# In health check: discover untracked jobs by parsing command strings
for job in all_pa_jobs:
    if job_id not in self._children:
        # Parse command for port number
        if f"port={expected_port}" in command_str:
            # Found untracked job - add to tracking
            self._children[job_id] = {...}
```

3. **Enhanced rate limiting checks:**
```python
def _should_spawn_replacement(self, index: int, all_pa_jobs: list) -> bool:
    # Check tracked jobs
    for child in self._children.values():
        if child["index"] == index and child["state"] in {"QUEUING", "QUEUED", "RUNNING"}:
            return False

    # Check ALL PA jobs (including untracked ones)
    for job in all_pa_jobs:
        if f"port={expected_port}" in command_str and job_state not in FAILED_STATES:
            return False  # Found untracked job still active

    # Check time-based rate limit
    ...
```

**Behavior:**
- Jobs spawned without IDs are tracked by timestamp only initially
- Health monitor discovers them in next poll and adds to tracking
- Rate limiter checks both tracked and ALL PA jobs to prevent duplicates
- Handles n+m jobs in PA API (n requested + m respawned)

**Example Scenario:**
```
10:00:00 - Spawn child 0, no job_id yet (pre-QUEUING)
10:00:00 - Record spawn time for index 0
10:00:05 - Health check runs
10:00:05 - Discover job-abc for index 0 (now has ID, in QUEUING state)
10:00:05 - Add job-abc to tracking
10:00:10 - Child 0 wants to spawn (misconfiguration)
10:00:10 - Rate limiter checks: found job-abc in QUEUING → Don't spawn
10:00:10 - Also checks: time since spawn = 10s < 60s → Don't spawn
```

**Files Updated:**
- `src/place_layers/command/pa_orchestrator.py`:
  - Modified `_spawn_child()` to handle missing job IDs
  - Added job discovery logic in `_check_and_restart_failed_children()`
  - Updated `_should_spawn_replacement()` to check all PA jobs
  - Added comments explaining PA API behavior

## Previous Changes

### Configuration Updates

#### 1. Default Image Updated
**Changed**: Default child image to actual cluster registry

**Before:**
```yaml
image: ubuntu
```

**After:**
```yaml
image: registry.toolkit-sp.yul201.service-now.com/snow.research.afm/interactive-toolkit:25.12-py3-vllm014rc1
```

**Reason**: Use the actual image available in the cluster that has all required dependencies (Python 3, vLLM 0.14rc1, etc.)

**Files Updated:**
- `src/place_layers/conf/command/pa_orchestrator.yaml`
- `examples/pa_user_job_spec.yaml`
- `scripts/submit_pa_orchestrator.py` (default argument)

#### 2. Default Child Count Reduced
**Changed**: Default `num_children` from 2 to 1

**Reason**: More conservative default to avoid over-allocating resources

**Files Updated:**
- `src/place_layers/conf/command/pa_orchestrator.yaml`
- `scripts/submit_pa_orchestrator.py` (default argument)

### Behavioral Changes

#### 3. Child Job Preemption and Restart Handling
**Updated**: `FAILED_STATES` to only include truly failed states

**Before:**
```python
FAILED_STATES = {"CANCELLED", "FAILED", "INTERRUPTED", "CANCELLING"}
```

**After:**
```python
FAILED_STATES = {"CANCELLED", "FAILED", "SUCCEEDED"}
```

**Reason**: Child jobs are now spawned with `preemptable: true` and `restartable: true`. With these settings:
- When cluster needs resources, it can interrupt (preempt) child jobs
- The cluster automatically restarts interrupted jobs (same job ID, different IP)
- PA orchestrator doesn't need to spawn new jobs for interruptions
- Only truly failed states (FAILED, CANCELLED, SUCCEEDED) require PA intervention

**Child Job Configuration:**
```python
job_spec = {
    ...
    "preemptable": True,   # Allow cluster to preempt when resources needed
    "restartable": True,   # Cluster auto-restarts when preempted
}
```

**Impact:**
- **INTERRUPTED** - Cluster handles restart automatically → PA does nothing
- **CANCELLING** - Transient state → PA waits for CANCELLED
- **FAILED** - Actual failure → PA spawns replacement
- **CANCELLED** - User cancelled → PA spawns replacement
- **SUCCEEDED** - Unexpected exit → PA spawns replacement

**Example Scenarios:**
- Cluster preempts job → Cluster restarts it with new IP → PA sees new IP in next poll ✓
- Server crashes (exit code 1) → FAILED state → PA spawns replacement ✓
- User cancels job → CANCELLED state → PA spawns replacement ✓
- Server exits cleanly (exit code 0) → SUCCEEDED state → PA spawns replacement ✓
- Server runs forever → RUNNING state → No action needed ✓

**Files Updated:**
- `src/place_layers/command/pa_orchestrator.py`
- `docs/PA_ORCHESTRATOR.md`
- `docs/PA_API_FIXES.md`

### Documentation Updates

All documentation has been updated to reflect:
1. Actual cluster image usage
2. SUCCEEDED state behavior
3. Simplified examples (no need to specify `--child-image`)
4. Updated default values

**Files Updated:**
- `docs/PA_ORCHESTRATOR.md` - Added explanation of SUCCEEDED state
- `docs/PA_API_FIXES.md` - Updated FAILED_STATES documentation
- `scripts/README.md` - Simplified examples

## Migration Guide

### For Existing Users

If you were using the PA orchestrator before these changes:

#### No Action Required If:
- You didn't specify `--child-image` → You'll automatically get the new default
- You specified `--num-children` explicitly → Your value is still used
- Your serve jobs run indefinitely → No change in behavior

#### Action Required If:
- You relied on serve jobs exiting successfully → They will now be restarted. If this is not desired, you may need to modify the health monitoring logic.

### For New Users

Simply use the script with the new defaults:

```bash
python scripts/submit_pa_orchestrator.py \
    --job-spec my_spec.yaml \
    --model-name apriel2_0_5b_dev \
    --num-children 4
```

The correct image and behavior are now built-in.

## Technical Details

### Preemptable + Restartable Architecture

**Key Insight**: With `preemptable: true` and `restartable: true`, the cluster handles interruptions transparently:

**Normal Operation:**
```
QUEUING → QUEUED → RUNNING → (runs indefinitely)
```

**Cluster Preemption (Handled by Cluster):**
```
RUNNING → INTERRUPTED → (cluster restarts) → QUEUING → QUEUED → RUNNING
                                                                    ↓
                                                          (possibly new IP)
```

**Actual Failure (Handled by PA):**
```
RUNNING → FAILED/CANCELLED/SUCCEEDED
    ↓
[PA detects & spawns NEW job]
    ↓
QUEUING (new job ID) → QUEUED → RUNNING
```

### Why SUCCEEDED is a "Failed" State

From a process supervision perspective:

**Expected behavior**: Serve job enters RUNNING state and stays there indefinitely

**Actual behavior if job exits**:
- Job transitions: RUNNING → SUCCEEDED (or FAILED)
- This is an **abnormal** state transition for a long-running server
- PA orchestrator detects this and spawns a **new** replacement job

**Alternative considered**: Only restart on FAILED/CANCELLED
- **Rejected** because: Servers can exit cleanly (code 0) due to bugs or misconfigurations
- Would result in reduced server count without detection
- Silent failures are worse than explicit restarts

### IP Address Handling

**Cluster Restarts (INTERRUPTED):**
- Same job ID
- Possibly different IP address
- PA fetches updated IP from PA API in next health check
- Clients fetch updated server list from PA orchestrator's `/servers` endpoint

**PA Restarts (FAILED/CANCELLED/SUCCEEDED):**
- New job ID
- New IP address
- New job appears in PA tracking
- Clients fetch updated server list from PA orchestrator's `/servers` endpoint

### Production Readiness Checklist

- [x] Image updated to cluster registry
- [x] SUCCEEDED state handling for server restarts
- [x] Default child count reduced for safety
- [x] Documentation updated
- [x] Examples updated to match production config
- [x] PA API alignment verified

## Testing Recommendations

Before deploying to production, test:

1. **Normal operation**: Verify serve jobs stay in RUNNING state
2. **Clean exit**: Kill serve job with `exit 0` → Should restart
3. **Error exit**: Kill serve job with `exit 1` → Should restart
4. **Cancellation**: Cancel serve job → Should restart
5. **Image availability**: Verify image is accessible from cluster nodes
6. **Resource allocation**: Verify child jobs get scheduled with specified resources

## Support

If you encounter issues:

1. Check job logs: `eai job logs <job-id>`
2. Check PA orchestrator logs: `eai job logs <pa-job-id>`
3. Verify PA health: `curl http://<pa-ip>:8080/health`
4. Check server list: `curl http://<pa-ip>:8080/servers`
