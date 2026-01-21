# Process Agent API Reference

Compact reference for PA API v1 endpoints used by the orchestrator.

## Base URL

```
http://0.0.0.0:8666/api/v1
```

---

## Endpoints

### GET /api/v1/

Get Process Agent stats including all jobs.

**Response:** `Stats`
```json
{
  "isReady": true,
  "isShutdown": false,
  "id": "string",
  "user": "string",
  "jobs": [
    {
      "id": "string",
      "index": 0,
      "state": "RUNNING",
      "ip": "10.20.30.40",
      "name": "job-name",
      ...
    }
  ]
}
```

**Used by:** Health monitor to check all job states

---

### POST /api/v1/jobs

Create one or more jobs.

**Request Body:** Array of `JobSpec-Input`
```json
[
  {
    "name": "serve-port-8080",
    "image": "registry.example.com/image:tag",
    "command": ["bash", "-c", "python -m place_layers +command=serve ..."],
    "resources": {
      "cpu": 2,
      "mem": 16,
      "gpu": 1,
      "gpuMem": 12
    },
    "data": [
      "snow.home.user:/home/toolkit:rw",
      "snow.datasets:/mnt/datasets:ro"
    ],
    "environmentVars": [
      "HOME=/home/toolkit",
      "PYTHONPATH=/path"
    ],
    "workdir": "/home/toolkit",
    "preemptable": true,
    "restartable": true
  }
]
```

**Response:** Array of job specs with assigned indices
```json
[
  {
    "index": 0,
    "name": "serve-port-8080",
    ...
  }
]
```

**Used by:** `_spawn_child()` to create new serve jobs

---

### GET /api/v1/jobs/{index_or_job_id}

Get information about a specific job.

**Parameters:**
- `index_or_job_id`: Job index (0, 1, 2, ...) or job ID (UUID)

**Response:** `PAJob` or `PAJobSpec`
```json
{
  "id": "abc-123-def",
  "index": 0,
  "state": "RUNNING",
  "ip": "10.20.30.40",
  "name": "serve-port-8080",
  "resources": {
    "cpu": 2,
    "mem": 16,
    "gpu": 1
  },
  "runs": [
    {
      "id": "run-uuid",
      "state": "RUNNING",
      "ip": "10.20.30.40",
      "startedOn": "2026-01-20T10:00:00Z"
    }
  ]
}
```

**Used by:** Job status queries

---

### GET /api/v1/jobs/{index_or_job_id}/rerun

Restart a job at the given index.

**Parameters:**
- `index_or_job_id`: Job index (0, 1, 2, ...) or job ID (UUID)

**Response:** `RerunJob`
```json
{
  "message": "Job rerun initiated",
  "index": 0
}
```

**Used by:** `_restart_child(index)` to restart failed jobs

---

### POST /api/v1/jobs/{index_or_job_id}/kill

Kill a running job.

**Parameters:**
- `index_or_job_id`: Job index or job ID

**Response:** `KillJob`
```json
{
  "message": "Job kill initiated",
  "index": 0
}
```

**Used by:** Manual job termination (not used by orchestrator)

---

### GET /api/v1/shutdown

Initiate PA shutdown.

**Response:**
```json
{
  "message": "Shutdown initiated"
}
```

**Used by:** Clean shutdown (not used by orchestrator)

---

## Data Models

### Stats

Process Agent statistics and job list.

```typescript
{
  isReady: boolean          // PA is ready to accept jobs
  isShutdown: boolean       // PA is shutting down
  id: string                // PA instance ID
  user: string              // User running PA
  jobs: PAJob[]             // All jobs (including FAILED, CANCELLED)
}
```

**Note:** `jobs` array contains ALL jobs ever spawned, including terminated ones.

---

### PAJob

Job information returned by PA.

```typescript
{
  id: string                // Job UUID (may be null for newly spawned jobs)
  index: number             // Job index (0, 1, 2, ...)
  state: JobState           // Current state (see JobState enum)
  ip: string | null         // Job IP address (null if not running)
  name: string              // Job name
  resources: {
    cpu: number             // CPU cores
    mem: number             // Memory in GB
    gpu: number             // Number of GPUs
    gpuMem?: number         // Min GPU memory in GB (optional)
  }
  runs: JobRun[]            // All runs of this job
  created: string           // Creation timestamp
  updated: string           // Last update timestamp
}
```

---

### JobState

Job state enum (uppercase values).

```typescript
enum JobState {
  QUEUING = "QUEUING"       // Job is being queued
  QUEUED = "QUEUED"         // Job is queued, waiting for resources
  RUNNING = "RUNNING"       // Job is running
  CANCELLING = "CANCELLING" // Job is being cancelled
  CANCELLED = "CANCELLED"   // Job was cancelled
  INTERRUPTED = "INTERRUPTED" // Job was preempted by cluster
  FAILED = "FAILED"         // Job failed
  SUCCEEDED = "SUCCEEDED"   // Job completed successfully
}
```

**PA Orchestrator Handling:**
- `FAILED`, `CANCELLED`, `SUCCEEDED` → Spawn replacement
- `INTERRUPTED` → Cluster auto-restarts (PA monitors IP change)
- `CANCELLING` → Transient state, wait for `CANCELLED`

---

### JobSpec-Input

Job specification for creating jobs.

```typescript
{
  name: string              // Job name (required)
  image: string             // Container image (required)
  command: string[]         // Command to run (required)
  resources: {              // Resource requirements (required)
    cpu: number             // CPU cores
    mem: number             // Memory in GB
    gpu: number             // Number of GPUs
    gpuMem?: number         // Min GPU memory in GB (optional)
  }
  data?: string[]           // Data volume mounts (format: "source:dest:mode")
  environmentVars?: string[] // Environment variables (format: "KEY=value")
  workdir?: string          // Working directory
  preemptable?: boolean     // Allow cluster preemption (default: false)
  restartable?: boolean     // Auto-restart on interruption (default: false)
  isProcessAgent?: boolean  // Is this a PA job? (default: false)
  options?: {               // PA-specific options
    sidecars?: {
      "process-agent"?: {
        version?: number    // PA API version (default: 3)
      }
    }
  }
}
```

**PA Orchestrator Settings for Children:**
```json
{
  "preemptable": true,
  "restartable": true
}
```

**PA Orchestrator Settings for Self:**
```json
{
  "preemptable": false,
  "isProcessAgent": true,
  "options": {
    "sidecars": {
      "process-agent": {
        "version": 3
      }
    }
  }
}
```

---

### JobRun

Information about a single run of a job (used for restarts after preemption).

```typescript
{
  id: string                // Run UUID
  jobId: string             // Parent job UUID
  state: JobState           // Run state
  ip: string | null         // Run IP address
  createdOn: string         // When run was created
  queuedOn?: string         // When run was queued
  startedOn?: string        // When run started
  finishedOn?: string       // When run finished
  waitTime?: string         // Time spent waiting (duration string)
  startupTime?: string      // Time spent starting (duration string)
  runTime?: string          // Time spent running (duration string)
  totalTime?: string        // Total time (duration string)
}
```

**Note:** When a job is preempted and restarted, a new run is created with the same job ID but different run ID and potentially different IP.

---

## Usage Examples

### 1. Create Child Jobs

```python
import httpx

PA_BASE_URL = "http://0.0.0.0:8666/api/v1"

job_spec = {
    "name": "serve-port-8080",
    "image": "registry.example.com/image:tag",
    "command": ["bash", "-c", "python -m place_layers +command=serve port=8080"],
    "resources": {"cpu": 2, "mem": 16, "gpu": 1},
    "data": ["snow.home.user:/home/toolkit:rw"],
    "environmentVars": ["HOME=/home/toolkit"],
    "preemptable": True,
    "restartable": True,
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{PA_BASE_URL}/jobs",
        json=[job_spec],
        timeout=30.0,
    )
    data = response.json()
    job_index = data[0]["index"]
    print(f"Created job at index {job_index}")
```

---

### 2. Monitor All Jobs

```python
async with httpx.AsyncClient() as client:
    response = await client.get(f"{PA_BASE_URL}/", timeout=10.0)
    stats = response.json()

    for job in stats["jobs"]:
        print(f"Job {job['index']}: {job['state']} @ {job.get('ip')}")
```

---

### 3. Restart Failed Job

```python
index = 2  # Job index to restart

async with httpx.AsyncClient() as client:
    response = await client.get(
        f"{PA_BASE_URL}/jobs/{index}/rerun",
        timeout=30.0,
    )
    print(f"Restarted job at index {index}")
```

---

### 4. Get Specific Job Info

```python
index = 0  # Job index

async with httpx.AsyncClient() as client:
    response = await client.get(
        f"{PA_BASE_URL}/jobs/{index}",
        timeout=10.0,
    )
    job = response.json()
    print(f"Job state: {job['state']}, IP: {job.get('ip')}")
```

---

## Important Behaviors

### 1. Jobs Without IDs

Newly spawned jobs might not have an `id` field until they reach `QUEUING` state. The PA orchestrator discovers such jobs by parsing command strings to identify port numbers.

### 2. PA Never Deletes Jobs

`GET /api/v1/` returns ALL jobs ever spawned, including `FAILED`, `CANCELLED`, and `SUCCEEDED` jobs. The orchestrator filters by state to find active jobs.

### 3. Multiple Jobs Per Index

After respawns, PA API shows original + replacement jobs:
```
Initial: 4 children requested
PA API shows: 4 jobs (indices 0-3)

Child 0 fails, orchestrator spawns replacement
PA API shows: 5 jobs total
  - job-abc (index 0, FAILED)      ← Original
  - job-xyz (index 0, QUEUING)     ← Replacement
  - job-123 (index 1, RUNNING)
  - job-456 (index 2, RUNNING)
  - job-789 (index 3, RUNNING)
```

The orchestrator tracks "current job at index" by filtering for the latest job at each index.

### 4. Job Index Assignment

PA assigns job indices sequentially starting from 0. The orchestrator enforces `assert child_index == pa_job_index` to ensure consistent index mapping.

### 5. IP Changes After Preemption

When cluster preempts a job (`RUNNING` → `INTERRUPTED`) and restarts it:
- Same job ID
- New run ID
- Potentially different IP address
- State transitions: `INTERRUPTED` → `QUEUING` → `QUEUED` → `RUNNING`

The orchestrator polls PA API to detect IP changes and updates the server list accordingly.

---

## Field Naming Conventions

### User-Friendly → PA API Transformations

| User Input | PA API Field | Transformation |
|------------|--------------|----------------|
| `env: {KEY: val}` | `environmentVars: ["KEY=val"]` | Dict → Array of "KEY=val" strings |
| `workingDir: /path` | `workdir: /path` | Rename field |
| `cpu: 4` (top-level) | `resources: {cpu: 4}` | Wrap in `resources` object |
| `mem: 16` (top-level) | `resources: {mem: 16}` | Wrap in `resources` object |
| `gpu: 1` (top-level) | `resources: {gpu: 1}` | Wrap in `resources` object |

The helper script `submit_pa_orchestrator.py` handles these transformations automatically.

---

## Error Handling

### HTTP Status Codes

- `200 OK` - Request successful
- `422 Unprocessable Entity` - Validation error (check request body)
- `404 Not Found` - Job not found (invalid index/ID)
- `500 Internal Server Error` - PA internal error

### Common Errors

**Job not found:**
```json
{
  "detail": "Job with index 5 not found"
}
```

**Validation error:**
```json
{
  "detail": [
    {
      "loc": ["body", "resources", "cpu"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Complete OpenAPI Spec

For the full OpenAPI 3.1.0 specification with all endpoints, schemas, and details, see the original JSON file (not recommended for reading):

```
docs/PA_openapi.json (3,744 lines)
```

This compact reference covers the subset of PA API used by the orchestrator.
