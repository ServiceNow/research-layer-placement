# Main Branch Diff Analysis

**Last Updated:** 2026-01-25
**Our Branch:** `feature/unified-backend-api` (current: `eba265e`)
**Main Branch:** `origin/main` (commit `1d842e0`)
**Merge Base:** `2b89c52` (Fix docstring: correct=False not correct=None for truncated samples)
**Total Commits on Main Since Divergence:** 59

---

## 1. Features Already Ported

These have been implemented on our branch.

| Main Commit | Feature | Our Implementation | Status |
|-------------|---------|-------------------|--------|
| `3de051c` | Text-based generation (no chat template) | `generate_chat()`/`generate_text()` methods on all backends, clients, server | ✅ Ported |
| `f7442f9` | `evaluate_truncated_samples` flag | Config field in `BaseGenerativeEvaluatorConfig`, used in `_make_records()` | ✅ Ported |
| `1031248` | Increase `timeout_keep_alive` to 30s | Set in `command/serve.py` | ✅ Ported |
| `ecc59ef`, `6ff46cb`, `8129a1b` | REST backend distribution fix | `ServerPool.select_server()` with round-robin + `preferred_server` in `_call_with_failover()` | ✅ Ported |
| `255fe65` | Separate trace-writer output dirs for placements | Run-length encoded dir names (`a2g3`), placement in `OutputMetadata` | ✅ Ported (improved) |
| `d06b226`, `2a97683` | Batch task loading + docs caching | `load_tasks()` in `__aenter__`, async prefetching with `doc_prefetch_count` | ✅ Ported (improved) |
| *N/A (our addition)* | Streaming evaluation across tasks | Requests streamed across ALL tasks in single stream, not task-by-task | ✅ Our improvement |
| *N/A (our addition)* | Command refactoring | `resolve_placement()` utility, generic result handlers, cleaner evaluate/precompute_traces | ✅ Our improvement |

### Architectural Difference: Text Generation

**Main branch approach:**
- Config flag `use_chat_template` on evaluator
- Dispatch logic: `batch_generate()` vs `batch_generate_text()`
- Methods: `agenerate()` vs `agenerate_text()`

**Our approach:**
- Distinct request types: `GenerateChatRequest`, `GenerateTextRequest`
- Distinct methods: `generate_chat()`, `generate_text()`
- Distinct endpoints: `/generate_chat`, `/generate_text`
- Streaming interface with request/response pattern
- `use_chat_template` config flag dispatches between methods

Our approach is cleaner - the caller explicitly chooses which generation type to use, and the evaluator respects the config flag.

### Architectural Difference: Placement Output Directories

**Main branch approach:**
- Directory naming: MD5 hash of placement (`8f3a2b1c`)
- Writes `placement.txt` file with readable placement
- Adds `placement` field to every `BaseGeneration` and `SkippedProblem` record

**Our approach:**
- Directory naming: Run-length encoding (`a2g3`) - more readable
- Placement stored in `OutputMetadata` (metadata.json) - no duplication
- `backend.placement` property reads current placement from model
- `_current_placement` set in `__aenter__()` before any trace writer init
- Non-supernet models (`placement=None`) use base output directory

Our approach is cleaner - placement is stored once in metadata instead of duplicated on every record, and directory names are human-readable.

### Architectural Difference: Task and Docs Caching

**Main branch approach:**
- Pre-loads ALL docs in `__aenter__()` after batch loading tasks
- ~45s startup overhead for 57 MMLU tasks
- Loads docs even for tasks that might be complete (resumption)
- Task-by-task evaluation (processes all problems for task A, then all for task B, etc.)

**Our approach:**
- `_task_cache`: Batch loaded in `__aenter__()` via `load_tasks()` (~6x faster)
- `_docs_cache`: Lazy populated via async prefetcher, synchronized with `threading.Event`
- `doc_prefetch_count` config controls prefetch concurrency (default: 2)
- Both caches persist across placements (docs are task-specific, not placement-specific)
- **Streaming evaluation**: Requests streamed across ALL tasks in a single stream for optimal server utilization
- Lazy request generation to overlap doc loading with request processing

Our approach is better:
- No upfront doc loading overhead
- Prefetching hides I/O latency (~0.8s per task for HuggingFace metadata)
- Cross-task streaming prevents pipeline stalls between tasks
- Results processed as they arrive (may be out of order)

### Architectural Difference: REST Distribution

**Main branch approach:**
- `preferred_server` passed through call stack
- Health checking via simple list filtering

**Our approach:**
- `ServerPool` class with dual responsibilities:
  1. **Health Tracking**: Tracks healthy/unhealthy servers with exponential backoff recovery
  2. **Server Selection**: `select_server()` provides round-robin load balancing
- `_call_with_failover()` uses `preferred_server` parameter for initial server choice
- Thread-safe design documented (no locks needed due to GIL + no await points)

---

## 2. Features To Port (High Priority)

### 2.1 REST DNS Backend Config

**Commit:** `71574c6`
**Priority:** Medium
**Complexity:** Low

**What it does:**
- Auto-generates server URLs from DNS prefix pattern
- Example: `dns_prefix=my-cluster`, `num_servers=4` generates:
  - `http://my-cluster-id-0:8080`
  - `http://my-cluster-id-1:8080`
  - `http://my-cluster-id-2:8080`
  - `http://my-cluster-id-3:8080`

**Files to add/modify:**
```
src/place_layers/backend/rest.py              # Add RESTDNSBackendConfig class
src/place_layers/conf/backend/rest_dns.yaml   # Add config file
```

**Implementation:**
```python
class RESTDNSBackendConfig(RESTBackendConfig):
    backend_type: Literal["rest_dns"] = "rest_dns"

    dns_prefix: str = Field(description="DNS prefix for server hostnames")
    num_servers: int = Field(description="Number of servers", ge=1)
    port: int = Field(default=8080, ge=1, le=65535)

    # Override servers with empty default
    servers: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def generate_servers(cls, data: dict) -> dict:
        if not data.get("servers"):
            dns_prefix = data.get("dns_prefix")
            num_servers = data.get("num_servers")
            port = data.get("port", 8080)

            if dns_prefix and num_servers:
                data["servers"] = [
                    f"http://{dns_prefix}-id-{i}:{port}"
                    for i in range(num_servers)
                ]
        return data
```

---

## 3. Config Files to Copy

These are simple YAML configs that can be copied directly:

```bash
# Task configs for extraction evaluation
git show origin/main:src/place_layers/conf/task/fda.yaml > src/place_layers/conf/task/fda.yaml
git show origin/main:src/place_layers/conf/task/swde.yaml > src/place_layers/conf/task/swde.yaml

# Generation config for extraction tasks
git show origin/main:src/place_layers/conf/generation/extraction.yaml > src/place_layers/conf/generation/extraction.yaml

# Model configs (if needed for specific experiments)
git show origin/main:src/place_layers/conf/model/apriel2_1p6_0112.yaml > src/place_layers/conf/model/apriel2_1p6_0112.yaml
git show origin/main:src/place_layers/conf/model/apriel2_1p6_annealed_0113.yaml > src/place_layers/conf/model/apriel2_1p6_annealed_0113.yaml

# Placement configs (if needed)
git show origin/main:src/place_layers/conf/placement/9_outof_10_gdn.yaml > src/place_layers/conf/placement/9_outof_10_gdn.yaml
git show origin/main:src/place_layers/conf/placement/9_outof_10_kda.yaml > src/place_layers/conf/placement/9_outof_10_kda.yaml
git show origin/main:src/place_layers/conf/placement/9_outof_10_swa.yaml > src/place_layers/conf/placement/9_outof_10_swa.yaml
```

---

## 4. Refactors We Superseded

These changes exist on main but are obsolete for our branch due to architectural differences:

| Commit | Description | Why We Don't Need It |
|--------|-------------|---------------------|
| `c522df7`, `04d7995` | Refactor REST backend to eliminate code duplication | We rewrote REST backend with unified streaming interface |
| `ab22d04`, `1263df4`, `54c0b11` | Parallel task evaluation + async TraceWriter | **Not needed** - we use sequential tasks, sync append (see below) |
| `9c41f40` | Send all chats from a problem to same server | **Not needed** - our request types bundle all choices together (see below) |

### Why We Don't Need Parallel Task Evaluation (ab22d04)

**Main branch added:**
- `parallel_tasks: bool = True` config for LikelihoodEvaluator
- `asyncio.gather()` to run multiple tasks (e.g., MMLU subtasks) concurrently
- `async def append()` with `asyncio.Lock` to prevent concurrent file writes
- Then `append_sync()` for sync backends, then refactor to reduce duplication

**Why we decided against it:**
- Marginal benefit: async backend already saturates throughput within a single task
- If backend isn't saturated, increase batch size instead
- Added complexity: async locking, sync/async variants, cascading fixes
- Memory pressure from loading multiple task datasets simultaneously

Our approach: sequential task evaluation with sync `append()`. Simpler, same throughput.

**Config compatibility:** No configs on main explicitly set `parallel_tasks` - it's just a code default. Users of `+evaluator=likelihood` get sequential evaluation on our branch automatically.

### Why We Don't Need Problem-to-Server Affinity (9c41f40)

**Main branch's problem:**
```python
# Main's API took individual (context, completion) tuples
compute_chat_log_probs(chats: Sequence[tuple[context, completion]])
# Individual choices distributed across servers - broke diff-based scoring
```

**Our architecture solves this at the API level:**
```python
# Our request types bundle ALL choices for a problem together
ScoreChatRequest(
    key="problem_1",
    context_messages=[...],           # The question
    completion_messages=[A, B, C, D], # ALL choices
)

ScoreTextRequest(
    key="problem_1",
    context="...",
    choices=["A", "B", "C", "D"],  # ALL choices
)
```

The fan-in happens at the **evaluator level** (`_build_choice_chats` bundles choices), not the backend. Each request is atomic (one problem, all choices) and goes to exactly one server.

---

## 5. Low Priority / Nice to Have

*(No remaining items - all useful features ported)*

---

## 6. Test Files Added on Main

These tests may be useful to review but likely need adaptation for our architecture:

```
tests/benchmark_likelihood_rest.py
tests/debug_fixture.py
tests/start_test_server.py
tests/test_likelihood_rest_integration.py
tests/test_likelihood_rest_unit.py
tests/test_rest_distribution.py
tests/test_server_startup.py
```

---

## 7. Commands to View Specific Changes

```bash
# View full diff for a specific file
git diff 2b89c52..origin/main -- path/to/file.py

# View a specific commit
git show <commit_hash>

# View commit with full diff
git show <commit_hash> -p

# List all commits affecting a file
git log --oneline 2b89c52..origin/main -- path/to/file.py
```

---

## 8. Porting Checklist

### Completed ✅

- [x] `use_chat_template` flag (ported 2026-01-25) - dispatches between generate_chat/generate_text
- [x] `evaluate_truncated_samples` flag (ported 2026-01-25) - config field + usage in evaluators
- [x] `build_text_prompt()` function in lm_eval_utils.py (ported 2026-01-25)
- [x] REST round-robin distribution fix (ported 2026-01-25) - ServerPool.select_server() + preferred_server
- [x] Problem-to-server affinity (analyzed 2026-01-25) - **not needed**: our request types bundle all choices together
- [x] Separate output dirs per placement (ported 2026-01-25) - run-length encoded dir names (e.g., `a2g3a2`)
- [x] Batch task loading + docs caching (ported 2026-01-25) - `load_tasks()` in `__aenter__`, async prefetching
- [x] Streaming evaluation across tasks (added 2026-01-25) - requests across all tasks in single stream, not task-by-task
- [x] Command refactoring (added 2026-01-25) - `resolve_placement()` utility, generic result handlers

### To Do 📋

- [ ] REST DNS backend config - convenience for Kubernetes deployments
- [ ] Copy task configs (fda.yaml, swde.yaml) - extraction evaluation support
- [ ] Copy extraction.yaml generation config - extraction evaluation support

---

## 9. Notes

### TraceWriter Changes
Main has `append_sync()` vs `append()` distinction. Our TraceWriter uses sync writing. If we move to async evaluation loops, we may need similar handling.

### use_chat_template Integration
We added `use_chat_template` config field that dispatches between `generate_chat()` and `generate_text()` at evaluation time. This matches main's behavior while using our cleaner method-based API.

### ServerPool Design
Our `ServerPool` combines health tracking with load balancing (Option B from design discussion). Key design decisions:
- Round-robin counter operates on FULL server list, skipping unhealthy servers
- This maintains consistent distribution even when servers fail/recover
- Thread-safe without locks (GIL + no await points between read/write)
- Returns `None` when no servers available, letting `_call_with_failover` handle recovery

### Doc Prefetching Design
Docs are prefetched in background to hide ~0.8s I/O latency per task (HuggingFace metadata fetch):
- `_prefetch_docs()` runs as async task in `__aenter__`, bounded by `asyncio.Semaphore(prefetch_ahead)`
- Uses `asyncio.to_thread()` for blocking I/O so event loop isn't blocked
- `threading.Event` allows sync request generator to wait on async prefetcher (GIL released during I/O)
- Fallback: if prefetcher hasn't loaded docs yet, `_prepare_task()` loads synchronously

### Streaming Evaluation Design
Evaluation streams requests across ALL tasks rather than processing task-by-task:
- Prevents pipeline stalls when transitioning between tasks
- Lazy `request_generator()` yields requests across all tasks
- Results keyed by `"{task_name}:{idx}"` and looked up in `key_to_context` dict
- Docs retrieved from `_docs_cache[task_name]` (populated by prefetcher)
- Benefits: better server utilization, no idle time between tasks

### Command Refactoring
Cleaned up evaluate and precompute_traces commands:
- `placement_utils.py`: Shared `resolve_placement()` for cyclic extension, validation, name-to-index conversion
- Result handlers now accept any `BaseModel`, not just `SearchResult`
- `EvaluateCommandConfig` uses `result_handler: ResultHandlerConfig` (defaults via Hydra YAML)
- No defaults in Python configs - all defaults provided by Hydra YAML files
