# PA API Fixes - Alignment with OpenAPI Spec

This document summarizes the fixes made to align the PA orchestrator implementation with the official Process Agent OpenAPI specification.

## Key API Field Corrections

### 1. Job State Field

**Issue**: Code used `status` field
**Fix**: PA API uses `state` field

**Changes:**
- `pa_orchestrator.py`: Updated all references from `status` to `state`
- `submit_pa_orchestrator.py`: Updated job polling to check `state` instead of `status`

### 2. Job State Values

**Issue**: Code used lowercase state values (`failed`, `canceled`, etc.)
**Fix**: PA API uses uppercase enum values

**Valid JobState enum values:**
- `QUEUING` - Job is queuing
- `QUEUED` - Job is queued
- `RUNNING` - Job is running
- `CANCELLING` - Job is being cancelled
- `CANCELLED` - Job was cancelled
- `INTERRUPTED` - Job was interrupted
- `FAILED` - Job failed
- `SUCCEEDED` - Job succeeded

**Changes:**
- Updated `FAILED_STATES` constant to: `{"CANCELLED", "FAILED", "SUCCEEDED"}`
- Updated all state comparisons to use uppercase values (e.g., `"RUNNING"` instead of `"running"`)
- Added `SUCCEEDED` to failed states because serve jobs should run indefinitely; if they exit cleanly, they need to be restarted
- Removed `INTERRUPTED` and `CANCELLING` from failed states because child jobs are spawned with `preemptable: true` and `restartable: true`, so the cluster handles these states automatically

### 3. Stats Response Structure

**Issue**: Code assumed `GET /api/v1/` returned dict keyed by job_id
**Fix**: PA API returns `Stats` object with `jobs` array

**Response structure:**
```json
{
  "isReady": true,
  "isShutdown": false,
  "id": "...",
  "user": "...",
  "jobs": [
    {"id": "...", "state": "RUNNING", "ip": "...", ...},
    {"id": "...", "state": "QUEUED", "ip": null, ...}
  ]
}
```

**Changes:**
- Modified `_check_and_restart_failed_children()` to:
  1. Extract `jobs` array from `Stats` object
  2. Build dict keyed by job_id for lookup
  3. Access job info via dict lookup

### 4. Job Spec Field Names

**Issue**: Code used incorrect field names
**Fix**: Aligned with PA API schema

| User-Friendly | PA API Field | Notes |
|---------------|--------------|-------|
| `env` (dict) | `environmentVars` (array) | Convert `{KEY: val}` to `["KEY=val"]` |
| `workingDir` | `workdir` | Direct rename |
| `nonpreemptible` | `preemptable` | Inverted boolean |
| top-level `cpu`, `mem`, `gpu` | `resources` object | Wrap in object |

**Changes in `submit_pa_orchestrator.py`:**
```python
# Convert env dict to environmentVars array
if "env" in merged:
    env_dict = merged.pop("env")
    merged["environmentVars"] = [f"{k}={v}" for k, v in env_dict.items()]

# Convert workingDir to workdir
if "workingDir" in merged:
    merged["workdir"] = merged.pop("workingDir")

# Wrap resources
resources = {}
for key in ["cpu", "mem", "gpu"]:
    if key in merged:
        resources[key] = merged.pop(key)
if resources:
    merged["resources"] = resources

# Use preemptable instead of nonpreemptible
merged["preemptable"] = False
```

## Configuration Improvements

### PA Defaults Config File

PA-specific configuration has been extracted to a separate YAML file for easier maintenance:

**Location:** `config/pa_defaults.yaml`

**Contents:**
```yaml
isProcessAgent: true
options:
  sidecars:
    process-agent:
      version: 3  # PA API version
preemptable: false
```

**Benefits:**
- Single source of truth for PA settings
- Easy to update when PA API version changes
- Can be overridden with `--pa-config` flag
- No need to modify Python code for PA updates

**Usage:**
```bash
# Use default config
python scripts/submit_pa_orchestrator.py --job-spec my_spec.yaml ...

# Use custom config
python scripts/submit_pa_orchestrator.py \
    --job-spec my_spec.yaml \
    --pa-config /path/to/custom_pa_config.yaml \
    ...
```

## Files Modified

### 1. `src/place_layers/command/pa_orchestrator.py`

**Line 37**: Updated `FAILED_STATES` to only include truly failed states
```python
FAILED_STATES = {"CANCELLED", "FAILED", "SUCCEEDED"}
```

Notes:
- `SUCCEEDED` is included because serve jobs should run indefinitely. If a serve job completes successfully, it means the server stopped when it should still be running, so PA spawns a replacement.
- `INTERRUPTED` and `CANCELLING` are NOT included because child jobs are spawned with `preemptable: true` and `restartable: true`. When jobs are interrupted (preempted), the cluster automatically restarts them. PA only needs to handle truly failed states.

**Line 104**: Check for `"RUNNING"` state
```python
if child["state"] == "RUNNING"
```

**Line 114**: Check for `"RUNNING"` state in health endpoint
```python
if child["state"] == "RUNNING"
```

**Line 181**: Store `state` field from API response
```python
"state": job_info.get("state", "QUEUING"),
```

**Lines 148-161**: Added preemption and restart settings to child job spec
```python
job_spec = {
    ...
    "preemptable": True,   # Allow cluster to preempt when resources needed
    "restartable": True,   # Cluster auto-restarts when preempted
}
```

This enables the cluster to handle interruptions automatically, reducing PA orchestrator's responsibilities.

**Lines 201-250**: Rewrote `_check_and_restart_failed_children()` to:
- Parse `Stats` object correctly
- Extract `jobs` array
- Use `state` field
- Use uppercase state values
- Only restart jobs in truly failed states (FAILED, CANCELLED, SUCCEEDED)
- Let cluster handle INTERRUPTED state automatically

### 2. `scripts/submit_pa_orchestrator.py`

**Lines 46-83**: Rewrote `merge_specs()` to:
- Convert `env` dict to `environmentVars` array
- Convert `workingDir` to `workdir`
- Wrap `cpu`, `mem`, `gpu` in `resources` object
- Use `preemptable` instead of `nonpreemptible`

**Lines 241-265**: Updated `get_job_info()` to:
- Look for `state` field (with fallback to `status`)
- Normalize response to always have `state` field

**Lines 286-301**: Updated `wait_for_job_ip()` to:
- Check `state` field
- Use uppercase state values (`FAILED`, `CANCELLED`, `INTERRUPTED`)

### 3. `examples/pa_user_job_spec.yaml`

Added comments explaining automatic field conversions:
```yaml
# Note: The helper script will automatically convert:
#   - env dict -> environmentVars array
#   - workingDir -> workdir
#   - top-level cpu/mem/gpu -> resources object
```

### 4. `docs/PA_ORCHESTRATOR.md`

Updated documentation to reflect:
- Correct field names and conversions
- Uppercase state values
- Stats response structure

## Testing Recommendations

1. **Test job submission**: Verify PA accepts the generated job spec
2. **Test state tracking**: Verify orchestrator correctly parses job states
3. **Test failover**: Kill a child job and verify replacement spawn
4. **Test environment variables**: Verify env vars are passed correctly to children
5. **Verify /servers endpoint**: Check running jobs are correctly identified

## API Reference

For complete API details, see:
- `src/place_layers/command/PA_openapi.json`
- PA API endpoints:
  - `POST /api/v1/jobs` - Create jobs (accepts array of JobSpec-Input)
  - `GET /api/v1/` - Get stats (returns Stats with jobs array)
  - `GET /api/v1/jobs/{id}` - Get job info (returns PAJob or PAJobSpec)
