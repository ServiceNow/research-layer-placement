# Main Branch Diff Analysis

**Last Updated:** 2026-01-25
**Our Branch:** `feature/unified-backend-api` (current: `fea9d8c`)
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

### 2.1 Dataset Docs Caching

**Commit:** `2a97683`
**Priority:** High
**Complexity:** Low

**What it does:**
- Caches loaded documents per task in `_docs_cache` dict
- Pre-loads all docs in `__aenter__()` after batch loading tasks
- Avoids re-reading HuggingFace metadata on every `_prepare_task()` call

**Performance impact:**
- Before: ~45s per allocation for 57 MMLU tasks
- After: <1s per allocation
- ~18x speedup for optimization runs

**Files to modify:**
```
src/place_layers/evaluator/task_based_evaluator.py
```

**Implementation:**
```python
class TaskBasedEvaluator(BackendEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._docs_cache: dict[str, Docs] = {}

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        # ... after loading tasks ...

        # Pre-load all docs
        for task_name, task in self._task_cache.items():
            task.download()
            self._docs_cache[task_name] = get_docs(task, self.config.split)

        return self

    def _prepare_task(self, ...):
        # Use cached docs instead of calling get_docs()
        task_name = get_task_name(task_spec)
        if task_name in self._docs_cache:
            docs = self._docs_cache[task_name]
        else:
            docs = get_docs(task, self.config.split)
```

---

### 2.2 Batch Load Tasks in `__aenter__`

**Commit:** `d06b226`
**Priority:** Medium
**Complexity:** Medium

**What it does:**
- Loads all lm-eval tasks upfront in `__aenter__()` instead of on-demand
- Uses `load_tasks()` (batch) instead of `load_task()` (individual)
- ~6x faster task loading

**Files to modify:**
```
src/place_layers/evaluator/task_based_evaluator.py
src/place_layers/evaluator/lm_eval_utils.py  # May need load_tasks() function
```

**Note:** Check if we already have `load_tasks()` or need to add it.

---

### 2.3 REST DNS Backend Config

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
| `54c0b11`, `1263df4` | TraceWriter refactors (append_sync vs append) | Different architecture - check if sync handling needed |
| `9c41f40` | Send all chats from a problem to same server | **Consider porting** - may improve diff-based scoring consistency |
| `255fe65` | Separate trace-writer output dirs for placements | Potentially useful - review for our trace output structure |

---

## 5. Low Priority / Nice to Have

### 5.1 Problem-to-Server Affinity

**Commit:** `9c41f40`

Sends all chats from a given problem to the same server. This may improve consistency for diff-based answer range calculation since all choices for a multiple-choice question are scored together.

**Consideration:** Our `select_server()` uses pure round-robin. For scoring operations where all choices for a problem should go to the same server, we might want to add an optional `affinity_key` parameter.

### 5.2 Separate Output Dirs per Placement

**Commit:** `255fe65`

Uses separate trace-writer output directories for different placements. May be useful for organization when running evaluations across many placements.

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

### To Do 📋

- [ ] Dataset docs caching (`_docs_cache`) - ~18x speedup for multi-task evaluations
- [ ] Batch task loading (`load_tasks()` in `__aenter__`) - ~6x faster task loading
- [ ] REST DNS backend config - convenience for Kubernetes deployments
- [ ] Copy task configs (fda.yaml, swde.yaml) - extraction evaluation support
- [ ] Copy extraction.yaml generation config - extraction evaluation support
- [ ] Review problem-to-server affinity for scoring consistency

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
