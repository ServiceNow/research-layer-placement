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
python -m place_layers +experiments=eval_10000_gsm8k_attention
```

## Commands

### 1. Evaluate (`command=evaluate`)

#### Generative Evaluation

Generate completions and check correctness. Accurate but slow.

**Single placement:**
```bash
python -m place_layers \
    command=evaluate \
    evaluator=generative \
    backend=transformers \
    model=10000 \
    task=gsm8k \
    generation=greedy \
    placement=all_attention
```

**Sweep over placements:**
```bash
python -m place_layers \
    command=evaluate \
    evaluator=generative \
    backend=transformers \
    model=10000 \
    task=gsm8k \
    generation=greedy \
    placement=all_attention,every_2nd_gdn,every_4th_gdn \
    --multirun
```

**Pre-configured experiments:**
```bash
# 10000 checkpoint with all-attention
python -m place_layers +experiments=eval_10000_gsm8k_attention

# 10000 checkpoint with GDN at every 4th layer
python -m place_layers +experiments=eval_10000_gsm8k_every4th_gdn

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
TRACE_PATH=/path/to/traces python -m place_layers +experiments=eval_10000_gsm8k_is

# Compare placements quickly
TRACE_PATH=/path/to/traces python -m place_layers +experiments=eval_10000_gsm8k_is \
    placement=all_attention,every_4th_gdn --multirun
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

### 3. Placement Optimization

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

The optimizer:
1. Enumerates **allocations** (mixer counts) at or below `cost_budget`
2. For each allocation, searches for the best **layer assignment**
3. Uses fast IS scoring (no generation needed)
4. Returns the best placement within the budget

Key insight: cost depends only on allocation (counts), but performance depends on which specific layers use which mixers.

**Tuning:** The optimizer uses hierarchical beam search (deterministic, reproducible). Key parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `search_strategy.beam_width` | 5 | Wider beam = better quality, more evals |
| `search_strategy.max_allocations` | 50 | More allocations = better coverage |

Typical evaluation counts:

| Setting | Evals | Use Case |
|---------|-------|----------|
| beam=1, alloc=5 | ~3k | Quick validation |
| beam=5, alloc=50 | ~70k | Production |
| beam=8, alloc=54 | ~155k | Final optimization |

#### Step 4: Validate

Evaluate the winning placements with generative evaluation:

```bash
python -m place_layers \
    command=evaluate \
    evaluator=generative \
    placement='[attention, gdn, attention, ...]'  # winning placement
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration. Key config groups:

| Group | Options | Description |
|-------|---------|-------------|
| `command` | evaluate, optimize, precompute_traces | What to run |
| `evaluator` | generative, importance_weighted, mock_supernet | How to score placements |
| `backend` | transformers, vllm | Model inference backend |
| `model` | 6000, 10000, qwen2_0_5b | Model checkpoint |
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
