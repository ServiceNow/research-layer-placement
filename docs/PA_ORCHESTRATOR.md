# PA Orchestrator Guide

The PA (Process Agent) orchestrator provides self-healing, distributed inference for the place_layers project. It automatically spawns and manages multiple serve jobs, providing a single endpoint for dynamic server discovery with automatic failover.

## Architecture

```
┌─────────────────────┐
│  PA Orchestrator    │  ← Single entry point
│  (Process Agent)    │
│                     │
│  GET /servers       │  ← Returns active server list
│  GET /health        │  ← Health status
└──────────┬──────────┘
           │
           │ spawns & monitors
           │
    ┌──────┴──────┬──────────┬──────────┐
    │             │          │          │
    ▼             ▼          ▼          ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ Serve  │  │ Serve  │  │ Serve  │  │ Serve  │
│ (GPU 0)│  │ (GPU 1)│  │ (GPU 2)│  │ (GPU 3)│
└────────┘  └────────┘  └────────┘  └────────┘

Auto-restart on failure ↻
```

## Features

- **Dynamic Server Discovery**: Clients query PA for current server list
- **Auto-Recovery**: PA monitors child jobs and restarts failed ones
- **Rate Limiting Protection**: Prevents resource exhaustion from misconfigured jobs
- **Load Balancing**: Client-side load balancing via `ServerPool`
- **Health Tracking**: Exponential backoff for unhealthy servers
- **Minimal Configuration**: Just point to PA URL instead of listing all servers

## Configuration Files

The PA orchestrator uses two configuration files for maximum flexibility:

### 1. PA Defaults (`config/pa_defaults.yaml`)

Contains PA-specific settings that rarely change:
```yaml
isProcessAgent: true
options:
  sidecars:
    process-agent:
      version: 3  # PA API version
preemptable: false
```

**When to modify:**
- PA API version changes
- Need different preemption settings
- Additional PA options required

**Location:** `config/pa_defaults.yaml` (can be overridden with `--pa-config`)

### 2. User Job Spec

Contains your job-specific requirements (resources, mounts, environment).

---

**For detailed configuration architecture**, see [PA_CONFIG_ARCHITECTURE.md](PA_CONFIG_ARCHITECTURE.md)

## Quick Start

### 1. Create User Job Spec

Create a file `my_job_spec.yaml`:

```yaml
cpu: 4
mem: 16  # GB
image: ubuntu
data:
  - servicenow.data.home:/home/toolkit
env:
  PYTHONPATH: /home/toolkit/code/research-layer-placement/src
```

### 2. Submit PA Orchestrator

```bash
python scripts/submit_pa_orchestrator.py \
    --job-spec my_job_spec.yaml \
    --model-name apriel2_0_5b_dev \
    --num-children 4 \
    --child-mem 32 \
    --child-gpu 1
```

The script will:
1. Merge your job spec with PA configuration
2. Submit the job to the cluster
3. Wait for the job to start
4. Print the PA URL

Example output:
```
============================================================
PA Orchestrator is ready!
============================================================
Job ID: abc123
PA URL: http://10.20.30.40:8080

Use this backend config:
  +backend=pa_rest backend.pa_url='http://10.20.30.40:8080'
============================================================
```

### 3. Use PA Backend in Evaluation

```bash
python -m place_layers \
    +command=evaluate \
    +backend=pa_rest \
    backend.pa_url='http://10.20.30.40:8080' \
    +evaluator=generative \
    +task=gsm8k \
    +generation=greedy
```

## Configuration Options

### PA Orchestrator Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-children` | 2 | Number of serve jobs to maintain |
| `--child-mem` | 32 | Memory per child (GB) |
| `--child-gpu` | 1 | GPUs per child |
| `--child-image` | ubuntu | Container image for children |
| `--child-port-start` | 8080 | Starting port (increments per child) |
| `--health-check-interval` | 30 | Health check interval (seconds) |
| `--min-spawn-interval` | 60 | Minimum seconds between spawn attempts for same child (rate limiting) |
| `--pa-port` | 8080 | PA orchestrator API port |

### User Job Spec Fields

Your job spec can include:

- `cpu`: CPU cores for PA container
- `mem`: Memory (GB) for PA container
- `image`: Container image
- `data`: List of data mounts (e.g., `["servicenow.data.home:/home/toolkit"]`)
- `env`: Environment variables as dict (e.g., `{PYTHONPATH: /path, LOG_LEVEL: INFO}`)
- `name`: Job name (optional)
- `workingDir`: Working directory (optional)

The script automatically:
- Converts `env` dict to `environmentVars` array (e.g., `["PYTHONPATH=/path", "LOG_LEVEL=INFO"]`)
- Converts `workingDir` to `workdir`
- Wraps `cpu`, `mem`, `gpu` into `resources` object
- Adds `isProcessAgent: true`
- Sets `preemptable: false`
- Adds `options.sidecars.process-agent.version: 3`
- Adds `command`: Orchestrator startup command

## PA API Endpoints

The PA orchestrator exposes:

### GET /servers

Returns list of active server URLs.

**Response:**
```json
{
  "servers": [
    "http://10.20.30.41:8080",
    "http://10.20.30.42:8081",
    "http://10.20.30.43:8082",
    "http://10.20.30.44:8083"
  ]
}
```

### GET /health

Returns orchestrator health status.

**Response:**
```json
{
  "status": "ok",
  "active_children": 4,
  "target_children": 4
}
```

## How It Works

### Server Discovery

1. Client creates `pa_rest` backend with PA URL
2. On `__aenter__`, backend calls `GET /servers` to get initial server list
3. Backend validates all servers and discovers model structure
4. Requests are load-balanced across healthy servers

### Failure Recovery

1. When a server fails, `ServerPool.handle_server_failure()` is called
2. In PA mode, this triggers `refresh_servers_from_pa()`
3. Backend fetches updated server list from PA
4. New servers are added to the pool
5. Client continues with refreshed server list

### PA Monitoring

1. PA orchestrator runs a background health monitor loop
2. Every N seconds (default: 30), PA queries Process Agent API (`GET /api/v1/`)
3. For each child job, PA checks state field
4. If state is in `{CANCELLED, FAILED, SUCCEEDED}`:
   - Remove job from tracking
   - Spawn replacement with same index (port)
5. New server appears in next `/servers` response

**Child Job Configuration:**
- Children are spawned with `preemptable: true` and `restartable: true`
- When cluster needs resources, it can interrupt (preempt) child jobs
- The cluster automatically restarts interrupted jobs (same job ID, possibly different IP)
- PA orchestrator handles IP changes transparently via `/servers` endpoint

**States PA Orchestrator Handles:**
- `FAILED` - Job crashed or encountered an error → Spawn replacement
- `CANCELLED` - Job was manually cancelled → Spawn replacement
- `SUCCEEDED` - Job exited cleanly when it should run indefinitely → Spawn replacement

**States Cluster Handles Automatically:**
- `INTERRUPTED` - Job was preempted by cluster → Cluster restarts it automatically
- `CANCELLING` - Job is being cancelled → Transient state, wait for CANCELLED

**PA API JobState enum values**: `QUEUING`, `QUEUED`, `RUNNING`, `CANCELLING`, `CANCELLED`, `INTERRUPTED`, `FAILED`, `SUCCEEDED`

### Rate Limiting Protection

To prevent resource exhaustion from misconfigured jobs, PA implements multi-layer rate limiting:

**Problem**: If a child job fails immediately (e.g., wrong image, insufficient memory), PA could spawn unlimited replacements in a tight loop.

**Solution**:
1. **Duplicate prevention**: Don't spawn if we already have a job at this index in `QUEUING`, `QUEUED`, or `RUNNING` state
   - Checks both tracked jobs (in orchestrator memory)
   - Checks all PA API jobs (including untracked ones without IDs yet)
2. **Job discovery**: Automatically discovers jobs spawned without IDs by parsing command strings for port numbers
3. **Time-based limiting**: Enforce minimum interval (default: 60 seconds) between spawn attempts for same child index

**PA API Edge Cases Handled:**
- **Jobs without IDs**: Newly spawned jobs might not have `id` field until QUEUING state
- **PA never deletes jobs**: `GET /api/v1/` returns ALL jobs (including FAILED, CANCELLED, SUCCEEDED)
- **Multiple jobs per index**: After respawns, PA API shows n (requested) + m (respawned) total jobs
- **Untracked jobs**: Jobs spawned without IDs are discovered in next health check

**Example with bad configuration:**
```
10:00:00 - Child 0 fails (FAILED state)
10:00:00 - PA spawns replacement immediately
10:00:05 - Replacement also fails (bad config)
10:00:05 - PA detects failure but is rate limited
10:00:05 - Log: "Not spawning replacement for child 0 yet - rate limited (min interval: 60s)"
10:01:00 - 60 seconds elapsed
10:01:00 - PA spawns another replacement
10:01:00 - Admin investigates logs and fixes configuration
```

**Configuration**:
```bash
# Increase interval for environments with slower startup
python scripts/submit_pa_orchestrator.py \
    --min-spawn-interval 120  # 2 minutes
```

**Benefits**:
- Prevents cluster resource exhaustion
- Gives time to investigate failures
- Prevents overwhelming the scheduler
- Configurable per deployment

## Comparison: REST vs PA REST

### Static REST Backend

```bash
# Manual server management
CUDA_VISIBLE_DEVICES=0 python -m place_layers +command=serve ... port=8080 &
CUDA_VISIBLE_DEVICES=1 python -m place_layers +command=serve ... port=8081 &
CUDA_VISIBLE_DEVICES=2 python -m place_layers +command=serve ... port=8082 &

# Hardcoded server list
python -m place_layers \
    +backend=rest \
    'backend.servers=["http://host1:8080","http://host2:8081","http://host3:8082"]'
```

**Issues:**
- Manual server startup
- No auto-recovery
- Must restart evaluation if server fails
- Static configuration

### PA REST Backend

```bash
# Single command to start orchestrator
python scripts/submit_pa_orchestrator.py --job-spec my_spec.yaml ...

# Single PA URL
python -m place_layers \
    +backend=pa_rest \
    backend.pa_url='http://pa:8080'
```

**Benefits:**
- Automatic server management
- Auto-recovery on failure
- Dynamic server discovery
- Seamless failover

## Troubleshooting

### Job doesn't start

Check job logs:
```bash
eai job logs <job-id>
```

Common issues:
- Insufficient resources (CPU/mem/GPU)
- Image not available
- Data mount errors

### No servers returned

Check PA orchestrator logs:
```bash
eai job logs <job-id>
```

Common issues:
- Children failing to spawn (check PA logs for spawn errors)
- Model loading failures (check child job logs)
- Network issues

### Servers frequently failing

Check health monitoring:
```bash
curl http://PA_IP:8080/health
```

If servers are cycling (frequent restarts):
- Reduce `num_children` to match available resources
- Increase `child_mem` if OOM errors
- Check model compatibility with image

## Advanced Usage

### Custom Health Check Interval

For faster recovery:
```bash
python scripts/submit_pa_orchestrator.py \
    ... \
    --health-check-interval 10
```

### Multiple PA Orchestrators

Run multiple PA orchestrators for different models:

```bash
# PA 1: Apriel2-0.5B
python scripts/submit_pa_orchestrator.py \
    --job-spec spec1.yaml \
    --model-name apriel2_0_5b_dev \
    --num-children 4

# PA 2: Qwen2-0.5B
python scripts/submit_pa_orchestrator.py \
    --job-spec spec2.yaml \
    --model-name qwen2_0_5b \
    --num-children 4
```

Then use different PA URLs in your evaluations.

### Debugging

Enable debug logging:
```yaml
# In user job spec
env:
  LOG_LEVEL: DEBUG
```

Check PA API manually:
```bash
# Get servers
curl http://PA_IP:8080/servers

# Get health
curl http://PA_IP:8080/health
```

Check Process Agent API:
```bash
# Inside PA job
curl http://0.0.0.0:8666/api/v1

# Get specific job
curl http://0.0.0.0:8666/api/v1/jobs/<job-id>
```
