# Process Agent (PA) Orchestrator Guide

Comprehensive guide for using PA orchestrator to manage distributed inference with automatic failover.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [How It Works](#how-it-works)
6. [Operational Scenarios](#operational-scenarios)
7. [Troubleshooting](#troubleshooting)

---

## Overview

PA orchestrator provides self-healing distributed inference by automatically spawning and managing multiple serve jobs. It exposes a single endpoint for dynamic server discovery with automatic failover.

### Key Features

- **Dynamic Server Discovery**: Clients query PA for current server list
- **Auto-Recovery**: Monitors child jobs and restarts failed ones
- **Preemption Support**: Handles cluster preemption transparently
- **Rate Limiting**: Prevents resource exhaustion from misconfigured jobs
- **Load Balancing**: Client-side load balancing via `ServerPool`
- **Simple Configuration**: Point to PA URL instead of managing server list

### Architecture

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
```

---

## Quick Start

### 1. Create User Job Spec

Create `my_job_spec.yaml`:

```yaml
cpu: 4
mem: 16  # GB
image: registry.toolkit-sp.yul201.service-now.com/snow.research.afm/interactive-toolkit:25.12-py3-vllm014rc1
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

**Output:**
```
============================================================
PA Orchestrator is ready!
============================================================
Job ID: abc123
PA URL: http://10.20.30.40:8080
============================================================
```

### 3. Run Evaluation

```bash
python -m place_layers \
    +command=evaluate \
    +backend=pa_rest \
    backend.pa_url='http://10.20.30.40:8080' \
    +evaluator=generative \
    +task=gsm8k \
    +generation=greedy
```

---

## Configuration

### PA Orchestrator Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-children` | 2 | Number of serve jobs to maintain |
| `--child-mem` | 32 | Memory per child (GB) |
| `--child-gpu` | 1 | GPUs per child |
| `--child-image` | ubuntu | Container image for children |
| `--health-check-interval` | 30 | Health check interval (seconds) |
| `--min-spawn-interval` | 60 | Minimum seconds between spawn attempts (rate limiting) |
| `--max-retries` | 5 | Maximum retry attempts for failed jobs |

### Configuration Architecture

PA configuration is split into two files:

**1. PA Defaults (`config/pa_defaults.yaml`)** - PA-specific settings (rarely change):
```yaml
isProcessAgent: true
options:
  sidecars:
    process-agent:
      version: 3
preemptable: false
```

**2. User Job Spec** - Your job-specific requirements (resources, mounts, env):
```yaml
cpu: 4
mem: 16
data:
  - servicenow.data.home:/home/toolkit
env:
  PYTHONPATH: /path
```

The helper script automatically:
- Converts `env` dict → `environmentVars` array
- Converts `workingDir` → `workdir`
- Wraps `cpu`, `mem`, `gpu` → `resources` object
- Merges with PA defaults

### PA API Endpoints

**GET /servers** - Returns active server URLs:
```json
{
  "servers": [
    "http://10.20.30.41:8080",
    "http://10.20.30.42:8081"
  ]
}
```

**GET /health** - Returns orchestrator status:
```json
{
  "status": "ok",
  "active_children": 4,
  "target_children": 4
}
```

---

## How It Works

### Server Discovery

1. Client creates `pa_rest` backend with PA URL
2. On `__aenter__`, backend calls `GET /servers` to get initial server list
3. Requests are load-balanced across healthy servers
4. When server fails, backend refreshes server list from PA
5. New servers automatically added to pool

### Child Job Lifecycle

**Child Job Configuration:**
```python
job_spec = {
    "preemptable": True,   # Allow cluster to preempt
    "restartable": True,   # Cluster auto-restarts when preempted
}
```

**PA Orchestrator Handles:**
- `FAILED` - Job crashed → Spawn replacement
- `CANCELLED` - Job was cancelled → Spawn replacement
- `SUCCEEDED` - Job exited when it should run indefinitely → Spawn replacement

**Cluster Handles Automatically:**
- `INTERRUPTED` - Job preempted by cluster → Cluster restarts same job (new IP)
- `CANCELLING` - Transient state → Wait for CANCELLED

**PA JobState enum values:** `QUEUING`, `QUEUED`, `RUNNING`, `CANCELLING`, `CANCELLED`, `INTERRUPTED`, `FAILED`, `SUCCEEDED`

### Health Monitoring

1. PA runs background monitor loop every N seconds (default: 30)
2. Polls Process Agent API (`GET /api/v1/`) for all job states
3. For each child index, checks current job state
4. If state in `{CANCELLED, FAILED, SUCCEEDED}`: spawns replacement using `/jobs/{index}/rerun`
5. New server appears in next `/servers` response
6. Clients fetch updated list and continue

**Invariant:** `child_index == pa_job_index` (enforced by assertion)

### Rate Limiting Protection

**Problem:** Misconfigured jobs (wrong image, insufficient memory) could fail immediately and trigger unlimited spawn loop.

**Solution:** Multi-layer rate limiting:
1. **Duplicate prevention**: Don't spawn if job already exists at this index in QUEUING/QUEUED/RUNNING
2. **Time-based limiting**: Enforce minimum interval (default: 60s) between spawn attempts for same child
3. **Job discovery**: Automatically discovers jobs without IDs by parsing commands

**Example with bad config:**
```
10:00:00 - Child 0 fails (FAILED)
10:00:00 - PA spawns replacement
10:00:05 - Replacement fails (bad config)
10:00:05 - PA detects failure but is rate limited
10:00:05 - Log: "Not spawning replacement for child 0 yet - rate limited (min interval: 60s)"
10:01:00 - 60 seconds elapsed, PA spawns replacement
```

### Comparison: REST vs PA REST

**Static REST Backend:**
```bash
# Manual management - no auto-recovery
CUDA_VISIBLE_DEVICES=0 python -m place_layers +command=serve ... port=8080 &
CUDA_VISIBLE_DEVICES=1 python -m place_layers +command=serve ... port=8081 &

python -m place_layers +backend=rest 'backend.servers=["http://host1:8080","http://host2:8081"]'
```

**PA REST Backend:**
```bash
# Automatic management with auto-recovery
python scripts/submit_pa_orchestrator.py --job-spec my_spec.yaml ...
python -m place_layers +backend=pa_rest backend.pa_url='http://pa:8080'
```

---

## Operational Scenarios

### Normal Operation

```
PA Orchestrator: RUNNING
  ├─ Child 0: RUNNING on 10.20.30.41:8080
  ├─ Child 1: RUNNING on 10.20.30.42:8081
  ├─ Child 2: RUNNING on 10.20.30.43:8082
  └─ Child 3: RUNNING on 10.20.30.44:8083

GET /servers → ["http://10.20.30.41:8080", ...]
```

**Actions:** None - everything healthy

### Cluster Preempts Child

```
1. Cluster needs resources
2. Child 2: RUNNING → INTERRUPTED
3. Cluster auto-restarts: INTERRUPTED → QUEUING → QUEUED → RUNNING
4. Child 2 on new IP: 10.20.30.50:8082
5. PA polls, sees new IP, updates tracking
6. Next GET /servers includes new IP
7. Clients fetch updated list and continue
```

**Actions:**
- PA: Update IP in tracking
- Cluster: Auto-restart job
- Clients: Fetch updated server list

### Child Crashes

```
1. Child 1 crashes (OOM, bug, etc.)
2. Job: RUNNING → FAILED
3. PA detects FAILED state
4. PA calls /jobs/1/rerun to restart child
5. New job: QUEUING → QUEUED → RUNNING on 10.20.30.51:8081
6. Next GET /servers includes new IP
```

**Actions:**
- PA: Restart job via rerun endpoint
- Clients: Fetch updated server list

### Child Fails Repeatedly (Rate Limiting)

```
1. Child 2 has bad config (wrong image)
2. Job: QUEUING → FAILED
3. PA spawns replacement immediately
4. Replacement fails within 10s: QUEUING → FAILED
5. PA detects failure but rate limiting applies
6. PA logs: "Not spawning replacement for child 2 yet - rate limited"
7. PA waits min_spawn_interval (60s)
8. Admin investigates logs, fixes configuration
9. Next spawn succeeds
```

**Actions:**
- PA: Apply rate limiting
- Admin: Check PA logs for warnings
- Admin: Check child logs for root cause
- Admin: Fix configuration
- PA: Auto-recover once fixed

---

## Troubleshooting

### Empty Server List

**Symptoms:**
```bash
curl http://PA_IP:8080/servers
{"servers": []}
```

**Possible Causes:**
- Children still starting (QUEUING/QUEUED)
- All children failed
- Image not available
- Resource constraints

**Debug:**
```bash
# Check PA logs
eai job logs <pa-job-id>

# Check health
curl http://PA_IP:8080/health
# If target_children > 0 but active_children = 0, children are failing

# Check child logs
eai job logs <child-job-id>
```

### Frequent Restarts

**Symptoms:**
- PA logs show many "Spawning replacement" messages
- Children transition to FAILED/SUCCEEDED frequently

**Possible Causes:**
- Out of memory (increase `--child-mem`)
- Model loading failures (check image/model path)
- GPU memory insufficient
- Serve command incorrect

**Debug:**
```bash
# Check child logs for errors
eai job logs <child-job-id>

# Check resource allocation
eai job show <child-job-id>
```

### Rate Limiting Triggered

**Symptoms:**
- PA logs: "Not spawning replacement for child X yet - rate limited"
- Server list incomplete for extended period
- Specific child keeps failing

**Possible Causes:**
- Bad configuration (wrong image, bad command)
- Insufficient resources (OOM, GPU allocation)
- Model loading errors

**Debug:**
```bash
# Check why child is failing
eai job logs <child-job-id>

# Look for:
# - Image pull errors
# - Out of memory errors
# - Model loading failures
# - Command syntax errors
```

**Actions:**
1. Fix root cause (verify image, resources, model path, command)
2. Adjust interval if needed: `--min-spawn-interval 300`
3. Cancel and resubmit with fixed config if necessary

### IP Not Updating After Preemption

**Symptoms:**
- Child shows RUNNING but wrong IP in /servers

**Possible Causes:**
- Health check interval too long
- PA not polling properly

**Debug:**
```bash
# Wait health_check_interval seconds (default: 30)
# Check PA logs for state transitions
eai job logs <pa-job-id>
# Look for "Child X state: INTERRUPTED -> RUNNING"
```

---

## Monitoring

### Check PA Status

```bash
# PA health
curl http://PA_IP:8080/health

# Server list
curl http://PA_IP:8080/servers

# PA logs
eai job logs <pa-job-id>
```

### Check Child Status

```bash
# Via PA API (inside PA job)
curl http://0.0.0.0:8666/api/v1/jobs/<child-job-id>

# Via eai
eai job show <child-job-id>
eai job logs <child-job-id>
```

### Key Log Messages

PA orchestrator logs show:
- `"Spawning child on port X"` - New child created
- `"Child X: A -> B"` - State transitions
- `"Child X failed with state: Z"` - Failures detected
- `"Successfully restarted child at index X"` - Restart completed
- `"Not spawning replacement for child X yet - rate limited"` - Rate limit applied

---

## Production Checklist

Before deploying:

- [ ] PA defaults config (`config/pa_defaults.yaml`) is correct
- [ ] User job spec has correct data mounts and environment
- [ ] Image is accessible from cluster nodes
- [ ] Model path is correct and accessible
- [ ] Resource allocation appropriate (mem, GPU)
- [ ] Health check interval appropriate (default: 30s)
- [ ] Rate limiting interval appropriate (default: 60s)
- [ ] Tested with single child first
- [ ] Tested preemption scenario
- [ ] Tested failure scenario
- [ ] Verified client can connect to PA URL

---

## Advanced Usage

### Multiple PA Orchestrators

Run multiple orchestrators for different models:

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

### Custom Health Check Interval

For faster recovery:
```bash
python scripts/submit_pa_orchestrator.py \
    ... \
    --health-check-interval 10
```

### Debug Logging

Enable in job spec:
```yaml
env:
  LOG_LEVEL: DEBUG
```

---

## API Reference

**PA Orchestrator Endpoints:**
- `GET /servers` - List of active server URLs
- `GET /health` - Orchestrator health status

**Process Agent API (internal):**
- `POST /api/v1/jobs` - Create jobs
- `GET /api/v1/` - Get stats with jobs array
- `GET /api/v1/jobs/{index}/rerun` - Restart job at index

For complete API details, see: `docs/PA_openapi.json`
