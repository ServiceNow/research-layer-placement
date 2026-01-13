# Layer Placement

A toolkit for evaluating and optimizing layer placements in supernet models (models with multiple mixer types per layer, e.g., attention, GDN, KDA).

## Installation

> ⚠️ **Environment setup is a work in progress.** Team setup for Superpod is in progress.

**For developers with Docker access:** Use the devcontainer (`.devcontainer/`) - this is the reference setup.

**Required dependencies** (CUDA kernel builds for mixer types):
```bash
# causal-conv1d and mamba-ssm (for Mamba)
pip install --no-build-isolation \
    "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d@v1.5.4" \
    mamba-ssm==2.2.6.post3

# flash-linear-attention (for GDN and KDA)
pip install --no-build-isolation \
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention@67eee20"
```

**This package:**
```bash
pip install -e .
```

See `.devcontainer/Dockerfile` for exact versions, base images, and CUDA architecture settings.

## Quick Start

Evaluate a checkpoint on GSM8K with all-attention placement:

```bash
python -m place_layers +experiments=eval_gsm8k_attention
```

## Backend Support

The key differentiator between backends is **placement switching** - the ability to dynamically change which mixer each layer uses. This is required for supernet evaluation and optimization.

| Backend | Placement Switching | Best For |
|---------|---------------------|----------|
| **Transformers** | ✅ Yes | Single-GPU supernet evaluation/optimization |
| **REST** | ✅ Yes | Multi-GPU distributed supernet evaluation/optimization |
| **vLLM** | ❌ No | Fast trace generation from standard (non-supernet) teachers |

**Why vLLM doesn't support placement switching:**

vLLM is a standard inference server - it doesn't know about supernet architecture. The model runs with its default/compiled placement. This is fine for:
- Generating traces from standard teachers (e.g., `Qwen/Qwen2-0.5B-Instruct`)
- Evaluating non-supernet models

But it cannot be used for supernet optimization or comparing different placements.

## Commands

### 1. Evaluate (`command=evaluate`)

#### Generative Evaluation

Generate completions and check correctness. Accurate but slow.

**Single placement:**
```bash
python -m place_layers \
    +command=evaluate \
    +evaluator=generative \
    +backend=transformers \
    +model=apriel2_0_5b_dev \
    +task=gsm8k \
    +generation=greedy \
    +placement=all_attention
```

**Sweep over placements:**
```bash
python -m place_layers \
    +command=evaluate \
    +evaluator=generative \
    +backend=transformers \
    +model=apriel2_0_5b_dev \
    +task=gsm8k \
    +generation=greedy \
    +placement=all_attention,every_2nd_gdn,every_4th_gdn \
    --multirun
```

**Pre-configured experiments:**
```bash
# Apriel2-0.5B-dev with all-attention
python -m place_layers +experiments=eval_gsm8k_attention

# Apriel2-0.5B-dev with GDN at every 4th layer
python -m place_layers +experiments=eval_gsm8k_every4th_gdn

# Qwen2-0.5B-Instruct baseline (non-supernet)
python -m place_layers +experiments=eval_qwen2_gsm8k
```

**Pre-configured placements:**
- `all_attention` - baseline, all attention layers
- `every_2nd_gdn` - GDN at layers 1,3,5,... (12 GDN layers for 24-layer model)
- `every_3rd_gdn` - GDN at layers 2,5,8,... (8 GDN layers)
- `every_4th_gdn` - GDN at layers 3,7,11,... (6 GDN layers)
- `every_2nd_kda`, `every_3rd_kda`, `every_4th_kda` - KDA variants

#### IS Evaluation

Fast evaluation using importance sampling. Requires precomputed traces (see [Precompute Traces](#2-precompute-traces-commandprecompute_traces) below).

**Step 1:** Precompute traces (once) — see [Section 2](#2-precompute-traces-commandprecompute_traces)

**Step 2:** Run IS evaluation (fast, forward passes only):
```bash
TRACE_PATH=/path/to/traces python -m place_layers +experiments=eval_gsm8k_is

# Compare placements quickly
TRACE_PATH=/path/to/traces python -m place_layers +experiments=eval_gsm8k_is \
    placement=all_attention,every_4th_gdn --multirun
```

**Multi-GPU IS evaluation:**
```bash
# Start servers (one per GPU)
CUDA_VISIBLE_DEVICES=0 python -m place_layers +command=serve +backend=transformers +model=apriel2_0_5b_dev port=8080
CUDA_VISIBLE_DEVICES=1 python -m place_layers +command=serve +backend=transformers +model=apriel2_0_5b_dev port=8081

# Run IS evaluation over REST
TRACE_PATH=/path/to/traces python -m place_layers +experiments=eval_gsm8k_is \
    backend=rest 'backend.servers=["http://localhost:8080","http://localhost:8081"]'
```

IS evaluation reports ESS (effective sample size) alongside the score estimate.

### 2. Precompute Traces (`command=precompute_traces`)

Generate teacher completions with log-probabilities. Required for IS evaluation and placement optimization.

**With vLLM (faster):**
```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-0.5B-Instruct \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8

# Generate traces
VLLM_MODEL="Qwen/Qwen2-0.5B-Instruct" python -m place_layers \
    command=precompute_traces \
    backend=vllm \
    task=gsm8k \
    generation=sampling
```

**With Transformers (slower but consistent):**
```bash
python -m place_layers \
    command=precompute_traces \
    backend=transformers \
    model=qwen2_0_5b \
    task=gsm8k \
    generation=sampling
```

> **Note:** Using Transformers for trace generation ensures numerical consistency between teacher and student log-probabilities, which may improve IS sample efficiency. vLLM is significantly faster but may introduce minor log-prob discrepancies.

**Output:** `validation.jsonl` with traces containing `teacher_log_prob` for each completion.

### 3. Placement Optimization (`command=optimize`)

Find optimal placements within a latency budget using cost-constrained search with IS evaluation.

#### Step 1: Precompute Traces

See [Section 2](#2-precompute-traces-commandprecompute_traces) above.

#### Step 2: Define Cost Model

Derive mixer costs by fitting a linear model to microbenchmark data.

**1. Run microbenchmarks:**
- Sample many mixer allocations (e.g., 10 FA + 20 SWA + 18 GDN)
- Placement order doesn't affect latency, only counts matter
- Measure tokens/s for each allocation
- Use sufficiently long context (e.g., 16k tokens) to capture realistic attention costs
- Can use an untrained model (latency depends on architecture, not weights)

**2. Fit linear model to inverse throughput:**

The cost model assumes per-token latency is additive across layers:
```
latency ∝ Σ (count_i × cost_i)
```
Fit to 1/throughput (not throughput directly) since latency is additive but throughput is not. Normalize so FA = 1.0.

Example fitted values for Apriel 15B:

| Mixer | Cost | Approx. Speedup |
|-------|------|-----------------|
| FA | 1.0 | 1.0× (baseline) |
| SWA | 0.156 | ~6× |
| GDN | 0.09 | ~11× |
| KDA | 0.11 | ~9× |

**3. Set cost budget:**

Budget = baseline_cost / target_speedup = num_layers / target_speedup (when FA=1.0)
- Example: 48 layers, 7.5× target → budget = 48 / 7.5 = 6.4

**4. Create problem config** (`conf/problem/cost_mymodel.yaml`):
```yaml
# @package problem
problem_type: cost_max_performance

cost_model:
  cost_model_type: linear
  mixer_costs: [1.0, 0.156, 0.09, 0.11]  # [FA, SWA, GDN, KDA]

cost_budget: 6.4  # 48 layers / 7.5x target
```

See `conf/problem/cost_apriel_15b.yaml` for a complete example.

#### Step 3: Optimize

Run cost-constrained optimization using IS evaluation:

```bash
python -m place_layers \
    command=optimize \
    evaluator=importance_weighted \
    problem=cost_mymodel \
    search_strategy=cost_constrained_hierarchical \
    result_handler=json
```

The optimizer uses a two-level hierarchical search:

1. **Outer loop**: Iterates over **allocations** (mixer counts) in cost order
2. **Inner loop**: For each allocation, finds the best **layer assignment** using beam search
3. Returns the best placement found within the cost budget

**Key insight**: Cost depends only on allocation (how many layers use each mixer), but performance depends on assignment (which specific layers use which mixer). This decomposition makes the search tractable.

<details>
<summary><strong>How the algorithm works (click to expand)</strong></summary>

#### Algorithm Overview

```
OUTER LOOP: for each allocation (expensive → cheap)              Budget ≤ 4.0
│
├─► Allocation (2,2,2) cost=3.6
│   │
│   │   INNER LOOP: hierarchical beam search
│   ├──► Phase 0: select 2 layers for mixer 0  →  {3,1}
│   │         ↓ lock in
│   ├──► Phase 1: select 2 layers for mixer 1  →  {0,4}
│   │         ↓ lock in
│   └──► Phase 2: assign remaining to mixer 2  →  {2,5}
│             ↓
│         [1,0,2,0,1,2]  score=0.85                             ← best
│
├─► Allocation (1,3,2) cost=3.4  →  beam search  →  score=0.81
├─► Allocation (1,2,3) cost=3.1  →  beam search  →  score=0.78
│
└─► ... stop after max_allocations

Return: best scoring assignment found
```

The **outer loop** enumerates allocations from expensive to cheap, skipping any above the cost budget. The **inner loop** (hierarchical beam search) finds the best layer assignment for each allocation — one phase per mixer, with decisions cascading down.

---

#### Worked Example

Tracing the inner loop for allocation `(2,2,2)` with `beam_width=2`.

**Notation:**
- **Selection** `{3,1}` — layers chosen for the current mixer
- **Assignment** `[1,0,2,0,1,2]` — complete layer→mixer mapping (position=layer, value=mixer)

To evaluate a selection, undecided layers get a **fallback mixer**, creating a complete assignment we can score.

---

#### Phase 0: Find Best 2 Layers for Mixer 0

Fallback mixer = 1 (undecided layers temporarily assigned here).

**Step 1: Score each layer individually**

| Selection | Assignment | Allocation | Matches? | Score |
|-----------|------------|------------|----------|-------|
| {0} | `[0,1,1,1,1,1]` | (1, 5, 0) | ❌ | 0.68 |
| {1} | `[1,0,1,1,1,1]` | (1, 5, 0) | ❌ | 0.71 |
| {2} | `[1,1,0,1,1,1]` | (1, 5, 0) | ❌ | 0.65 |
| {3} | `[1,1,1,0,1,1]` | (1, 5, 0) | ❌ | **0.73** ← best |
| {4} | `[1,1,1,1,0,1]` | (1, 5, 0) | ❌ | 0.70 |
| {5} | `[1,1,1,1,1,0]` | (1, 5, 0) | ❌ | 0.66 |

Top 2 selections initialize the beam: `{3}` and `{1}`.

**Step 2: Grow beam to size 2**

Expand each beam entry by adding one more layer:

| Selection | Assignment | Allocation | Matches? | Score |
|-----------|------------|------------|----------|-------|
| {3, 1} | `[1,0,1,0,1,1]` | (2, 4, 0) | ❌ | **0.79** ← best |
| {3, 4} | `[1,1,1,0,0,1]` | (2, 4, 0) | ❌ | 0.77 |
| {3, 0} | `[0,1,1,0,1,1]` | (2, 4, 0) | ❌ | 0.75 |
| {1, 4} | `[1,0,1,1,0,1]` | (2, 4, 0) | ❌ | 0.76 |
| ... | | | | |

Still no match to `(2, 2, 2)` — allocation is `(2, 4, 0)`. But we've found the best pair.

**Phase 0 result:** layers **{3, 1}** → mixer 0 (locked in)

---

#### Phase 1: Find Best 2 Layers for Mixer 1

**Fixed from Phase 0:** layers 1 and 3 → mixer 0

Remaining layers: {0, 2, 4, 5}. Fallback shifts to mixer 2.

**Step 1: Score each remaining layer**

| Selection | Assignment | Allocation | Matches? | Score |
|-----------|------------|------------|----------|-------|
| {0} | `[1,0,2,0,2,2]` | (2, 1, 3) | ❌ | 0.81 |
| {2} | `[2,0,1,0,2,2]` | (2, 1, 3) | ❌ | 0.78 |
| {4} | `[2,0,2,0,1,2]` | (2, 1, 3) | ❌ | **0.83** ← best |
| {5} | `[2,0,2,0,2,1]` | (2, 1, 3) | ❌ | 0.80 |

Allocation `(2, 1, 3)` — still doesn't match target `(2, 2, 2)`.

**Step 2: Grow beam to size 2**

| Selection | Assignment | Allocation | Matches? | Score |
|-----------|------------|------------|----------|-------|
| {0, 4} | `[1,0,2,0,1,2]` | **(2, 2, 2)** | ✅ | **0.86** |
| {4, 5} | `[2,0,2,0,1,1]` | (2, 2, 2) | ✅ | 0.85 |
| {2, 4} | `[2,0,1,0,1,2]` | (2, 2, 2) | ✅ | 0.84 |
| ... | | | | |

**Now** the allocations match! When the selection reaches target size, we get valid assignments.

**Phase 1 result:** layers **{0, 4}** → mixer 1 (locked in)

---

#### Phase 2: Remainder

**Fixed from Phase 0:** layers 1 and 3 → mixer 0
**Fixed from Phase 1:** layers 0 and 4 → mixer 1

Remaining layers {2, 5} are assigned to the last mixer (no search needed).

| Selection | Assignment | Allocation | Matches? | Score |
|-----------|------------|------------|----------|-------|
| {2, 5} | `[1,0,2,0,1,2]` | **(2, 2, 2)** | ✅ | **0.86** |

**Phase 2 result:** layers **{2, 5}** → mixer 2

---

#### Final Result

```
Layer:      0    1    2    3    4    5
Mixer:      1    0    2    0    1    2
            ─────────────────────────────
Assignment: [1,   0,   2,   0,   1,   2]
Allocation: (2, 2, 2) ✅
```

---

#### Why This Design?

| Property | Benefit |
|----------|---------|
| **Outer loop by cost** | Efficiently navigate cost-performance frontier |
| **Phase-by-phase** | Reduces search space exponentially |
| **Proxy scores** | Enable evaluation before full assignment is known |
| **Beam search** | Explores multiple promising paths, not purely greedy |
| **Deterministic** | Reproducible results, no random seed sensitivity |

| Tradeoff | |
|----------|---------|
| **Proxy scores** | Intermediate evaluations don't match target allocation — may mislead in some cases |
| **Greedy phases** | Cannot reconsider earlier phase decisions |

</details>

**Tuning parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `search_strategy.beam_width` | 5 | Wider beam explores more candidates per phase |
| `search_strategy.max_allocations` | 50 | More allocations = better cost-performance coverage |
| `search_strategy.early_stop_threshold` | 0.001 | Stop allocation early if best score improves by less than this; ~70% eval reduction |
| `search_strategy.cross_phase_beam_width` | 1 | Candidate paths across phases; 1=greedy (default), k=explore phase interactions at ~c× cost |

**Evaluation count:**

For N layers, M mixers, beam_width k, A allocations, with n = N/M layers per mixer:

```
Raw candidates (before deduplication):

  Evals_raw = A × { Σᵢ [Rᵢ × (1 + k(n-1)) - k×n(n-1)/2] + 1 }

  where Rᵢ = N - i×n (remaining layers at phase i), summed over i = 0 to M-2

Deduplication savings:

  Fixed (step 1, always):    D_fixed = A × (M-1) × k(k-1)/2
  Variable (steps 2+):       D_var ∈ [0, A × (M-1) × (n-2) × k(k-1)/2]

Actual evals:

  Evals ∈ [Evals_raw - D_fixed - D_var_max, Evals_raw - D_fixed]
```

**Why bounds, not exact?** Deduplication occurs when beam entries share s-1 layers. At step 1, beam entries are singletons, so `{Li}+Lj` always collides with `{Lj}+Li` → **step-1 dedup is deterministic**. At steps 2+, whether entries share layers depends on which candidates scored highest → **varies with evaluator scores**.

The bounds are tight (~6% apart). Early stopping typically reduces evals by ~70%, but this is empirical (depends on score landscape, not computable a priori).

Example (N=48, M=4, n=12):

| Setting | Per Alloc | Total | With Early Stop | Use Case |
|---------|-----------|-------|-----------------|----------|
| A=5, k=1 | 1,099 | 5,495 | ~1,600 | Quick test |
| A=5, k=5 | [4,729 – 5,029] | [23k – 25k] | ~7,500 | Quick, wide beam |
| A=5, k=8 | [7,105 – 7,945] | [35k – 39k] | ~12k | Quick, widest beam |
| A=50, k=5 | [4,729 – 5,029] | [236k – 251k] | ~75k | Production |
| A=50, k=8 | [7,105 – 7,945] | [355k – 397k] | ~120k | Production, wide beam |
| A=100, k=8 | [7,105 – 7,945] | [710k – 795k] | ~240k | Thorough |

Note: k=1 has no deduplication (single beam path). For k>1, range reflects [max dedup, min dedup].

**Cross-phase beam (c = `cross_phase_beam_width`):** Above formulas assume c=1 (default). With c>1, phases 1 through M-2 run c× (once per candidate path), so multiply those phases by c. Phase 0 is unchanged (always starts with 1 path).

#### Step 4: Validate

Evaluate the winning placements with generative evaluation:

```bash
python -m place_layers \
    command=evaluate \
    evaluator=generative \
    placement='[attention, gdn, attention, ...]'  # winning placement
```

## Multi-GPU Parallelization

For large-scale evaluation or trace generation, you can distribute work across multiple GPUs using the REST server/client architecture.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Client (evaluate, precompute_traces)           │
│              Uses: backend=rest                             │
│              Round-robins requests across servers           │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  GPU 0:8080  │      │  GPU 1:8081  │      │  GPU 2:8082  │
│  serve cmd   │      │  serve cmd   │      │  serve cmd   │
│  Transformers│      │  Transformers│      │  Transformers│
└──────────────┘      └──────────────┘      └──────────────┘
```

### Step 1: Start Servers

Start one server per GPU, each on a different port:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m place_layers \
    +command=serve +backend=transformers +model=apriel2_0_5b_dev port=8080

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m place_layers \
    +command=serve +backend=transformers +model=apriel2_0_5b_dev port=8081

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python -m place_layers \
    +command=serve +backend=transformers +model=apriel2_0_5b_dev port=8082
```

Each server loads the model and exposes it via REST API. Servers are stateless for placement - the client includes the desired placement in each request.

### Step 2: Run Commands with REST Backend

Use `backend=rest` with any command that supports backends:

**Generative Evaluation (distributed):**
```bash
python -m place_layers \
    +command=evaluate \
    +evaluator=generative \
    +backend=rest \
    'backend.servers=["http://localhost:8080","http://localhost:8081","http://localhost:8082"]' \
    +model=apriel2_0_5b_dev \
    +task=gsm8k \
    +generation=greedy \
    +placement=all_attention
```

**Trace Precomputation (distributed):**
```bash
python -m place_layers \
    +command=precompute_traces \
    +backend=rest \
    'backend.servers=["http://localhost:8080","http://localhost:8081","http://localhost:8082"]' \
    +model=apriel2_0_5b_dev \
    +task=gsm8k \
    +generation=sampling
```

### Configuration

**REST Backend (`backend=rest`):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `servers` | `["http://localhost:8080"]` | List of server URLs |
| `max_concurrent_per_server` | 8 | Max parallel requests per server |
| `timeout_seconds` | 300 | Request timeout |

**Serve Command (`command=serve`):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `0.0.0.0` | Bind address |
| `port` | `8080` | Listen port |
| `backend` | (required) | Must be `transformers` |

### Performance

| GPUs | Approximate Speedup |
|------|---------------------|
| 1 | 1× (baseline) |
| 4 | ~4× |
| 8 | ~8× |

The REST backend distributes requests using round-robin. Each server processes requests independently, so speedup scales linearly with GPU count.

### Applicable Commands

The REST backend works with commands that use the `backend` config group:

| Command | Evaluator | Use Case |
|---------|-----------|----------|
| `evaluate` | `generative` | Distribute generation across GPUs |
| `evaluate` | `importance_weighted` | Distribute IS log-prob computation across GPUs |
| `precompute_traces` | - | Distribute trace generation across GPUs |

> **Note:** The `serve` command only supports `backend=transformers`. You cannot serve a vLLM backend (vLLM is already a server).

### Tips

1. **Same model everywhere:** All servers must load the same model checkpoint.

2. **Placement handling:** The client stores placement locally and includes it in every request. Servers set placement before processing (no-op if unchanged). This enables load balancing without coordination.

3. **Environment variables:** Use `REST_SERVER_0`, `REST_SERVER_1`, etc. for server URLs:
   ```bash
   REST_SERVER_0=http://node1:8080 REST_SERVER_1=http://node2:8080 \
       python -m place_layers +command=evaluate +backend=rest ...
   ```

4. **Health checks:** The REST backend verifies server connectivity on startup.

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration. Key config groups:

| Group | Options | Description |
|-------|---------|-------------|
| `command` | evaluate, optimize, precompute_traces, serve | What to run |
| `evaluator` | generative, likelihood, importance_weighted, mock_* | How to score placements |
| `backend` | transformers, vllm, rest | Model inference backend |
| `model` | apriel2_0_5b_dev, qwen2_0_5b | Model loading config |
| `task` | gsm8k, gsm8k_quick | Evaluation task |
| `generation` | greedy, greedy_short, sampling | Generation parameters |
| `placement` | all_attention, every_Nth_gdn/kda | Layer placement |
| `search_strategy` | hierarchical_beam, cost_constrained_* | Optimization algorithm |
| `problem` | layer_placement, cost_* | Optimization problem definition |

See `CLAUDE.md` for detailed configuration patterns and architecture documentation.

## Output

Results are saved to `outputs/<config-name>/`:
- `evaluation_results.json` - scores and placement details
- `validation.jsonl` - individual completions (for generative eval)
- `.hydra/config.yaml` - resolved configuration
