# Layer Placement

A toolkit for evaluating and optimizing layer placements in supernet models (models with multiple mixer types per layer, e.g., attention, GDN, KDA).

## Installation

> **Environment setup is a work in progress.** Team setup for Superpod is in progress.

**For developers with Docker access:** Use the devcontainer (`.devcontainer/`) - this is the reference setup. All dependencies are pre-installed.

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

**1. Serve the model** (one GPU):
```bash
python -m place_layers +experiments=serve_vllm +model=apriel2_1p6_v2_annealed_0204
```

**2. Evaluate** (separate terminal):
```bash
python -m place_layers +experiments=eval_gsm8k_rest \
    +model=apriel2_1p6_v2_annealed_0204 +placement=all-attention
```

## Deployment

### Backend Support

The key differentiator between backends is **placement switching** - the ability to dynamically change which mixer each layer uses. This is required for supernet evaluation and optimization.

| Backend | Placement Switching | Best For |
|---------|---------------------|----------|
| **Transformers** | Yes | Single-GPU supernet evaluation/optimization |
| **Local vLLM** | Yes | Fast single-GPU supernet evaluation (Apriel2 models) |
| **Remote vLLM** | No | Fast trace generation from standard (non-supernet) teachers |
| **REST** | Yes | Multi-GPU distributed supernet evaluation/optimization |

**Local vs. Remote vLLM:**

- **Local vLLM** (`+backend=local_vllm`) runs an in-process vLLM engine and uses `collective_rpc` for placement control. It supports supernet evaluation and optimization with Apriel2 models. It can also be served (`+experiments=serve_vllm`) for multi-GPU setups.
- **Remote vLLM** (`+backend=vllm`) is a standard HTTP client to an external vLLM server. It cannot reach engine internals for placement switching, so it is limited to trace generation from non-supernet teachers and evaluating standard models.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│           Client (evaluate, explore_placements, ...)     │
│           Uses: backend=rest                             │
│           Round-robins requests across servers           │
└──────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  GPU 0:8080  │      │  GPU 1:8081  │      │  GPU 2:8082  │
│  serve_vllm  │      │  serve_vllm  │      │  serve_vllm  │
│  local_vllm  │      │  local_vllm  │      │  local_vllm  │
└──────────────┘      └──────────────┘      └──────────────┘
```

### Starting Servers

Start one server per GPU using the `serve_vllm` experiment:

```bash
for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU python -m place_layers \
      +experiments=serve_vllm +model=apriel2_1p6_v2_annealed_0204 \
      port=$((8080+GPU)) &
done
```

Then point any client command at the servers:

```bash
python -m place_layers +experiments=eval_gsm8k_rest \
    +model=apriel2_1p6_v2_annealed_0204 +placement=all-attention \
    'backend.servers=["http://localhost:8080","http://localhost:8081","http://localhost:8082","http://localhost:8083"]'
```

### REST Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend.servers` | `["http://localhost:8080"]` | List of server URLs |
| `backend.max_concurrent_per_server` | 8 | Max parallel requests per server |
| `backend.timeout_seconds` | 300 | Request timeout |

For Kubernetes deployments with predictable DNS naming, use `+backend=rest_dns` with `DNS_PREFIX` and `NUM_SERVERS` environment variables.

## Landscape Pipeline

The primary workflow for mapping the cost-vs-quality Pareto frontier. Three steps: **explore** (build surrogate dataset) → **validate** (confirm surrogate accuracy) → **visualize** (render landscape plot).

### Step 1: Explore Placements

Adaptively sample placements using a surrogate model. Uses `uniform_apriel_15b` cost model for unconstrained exploration (all allocations sampled equally). Terminates when the surrogate's cross-validated Spearman rho plateaus.

```bash
python -m place_layers +experiments=explore_gsm8k_rest \
    +model=apriel2_1p6_v2_annealed_0204
```

Key parameters (override on CLI):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_samples` | 10000 | Hard cap on total evaluations |
| `target_rank_correlation` | 0.95 | Stop when Ridge CV Spearman rho reaches this |
| `plateau_patience` | 3 | Batches without improvement before stopping |

Output: `exploration_progress.jsonl` (incremental), `exploration_results.json` (final).

### Step 2: Validate Landscape

Run per-band real evaluation to confirm surrogate accuracy. Uses `linear_apriel_15b` cost model (baked into the experiment config) since validation needs real cost bands.

```bash
python -m place_layers +experiments=validate_gsm8k_rest \
    +model=apriel2_1p6_v2_annealed_0204 \
    landscape.exploration_dataset_path=exploration_results.json
```

Output: `validation_results.json` with per-band surrogate vs. real scores and Spearman rho summary.

### Step 3: Visualize Landscape

Fit a Ridge surrogate, solve per-band DP-optimal placements via MRF Viterbi, and render a density plot with layermap.

```bash
# Surrogate-only preview
python -m place_layers +experiments=placement_landscape \
    landscape.exploration_dataset_path=exploration_results.json

# With validation overlay (dashed surrogate line + solid validated line)
python -m place_layers +experiments=placement_landscape \
    landscape.exploration_dataset_path=exploration_results.json \
    placement_landscape.validation_results_path=validation_results.json
```

Output: `placement_landscape.pdf`, `placement_landscape.png`, `landscape_data.json`.

## Cost-Constrained Optimization

When you need a single optimal placement for a specific cost budget, the optimizer builds on exploration data from the landscape pipeline. You've already mapped the space with `explore_placements` — now point the optimizer at a target budget and let the Viterbi surrogate search find the best placement.

### Running Optimization

```bash
# Step 1: Explore (reuse exploration_results.json from landscape pipeline, or run fresh)
python -m place_layers +experiments=explore_gsm8k_rest \
    +model=apriel2_1p6_v2_annealed_0204

# Step 2: Find optimal placement at a specific cost budget
python -m place_layers +command=optimize \
    +evaluator=generative +backend=rest +task=gsm8k +generation=greedy \
    +model=apriel2_1p6_v2_annealed_0204 \
    +cost_model=linear_apriel_15b +problem=cost_max_perf_apriel_15b_v2 \
    +search_strategy=viterbi_surrogate +result_handler=json \
    search_strategy.exploration_dataset_path=exploration_results.json
```

### Cost Model

The cost model assumes per-token latency is additive across layers. Normalize so FA = 1.0.

Fitted values for Apriel 15B:

| Mixer | Cost | Approx. Speedup |
|-------|------|-----------------|
| FA | 1.0 | 1.0x (baseline) |
| SWA | 0.156 | ~6x |
| KDA | 0.11 | ~9x |
| GDN | 0.09 | ~11x |

Budget = num_layers / target_speedup. Example: 48 layers, 7.5x target -> budget = 6.4.

See `conf/problem/cost_max_perf_apriel_15b_v2.yaml` for a complete problem config.

<details>
<summary><strong>Algorithm details (click to expand)</strong></summary>

The optimizer fits a Ridge surrogate to the exploration dataset, then uses the Viterbi algorithm to find globally optimal placements — no greedy phases, no beam width, no approximation.

1. **Surrogate as MRF**: The Ridge surrogate learned during exploration decomposes into chain-structured Markov random field (MRF) potentials — unary (how good is mixer X at layer Y), pairwise (how well do adjacent mixer choices interact), and optionally triplet terms. Each Ridge coefficient maps directly to a potential.

2. **Exact DP**: Because layers form a chain, the Viterbi algorithm finds the globally optimal placement in O(N·K²) time. For each feasible allocation (mixer count that satisfies the cost budget), DP finds THE best layer assignment.

3. **Bootstrap ensemble**: The surrogate maintains 50 bootstrap Ridge models. Each has slightly different coefficients, yielding slightly different MRF potentials and slightly different Viterbi optima. Pooling candidates from all bootstrap models gives automatic exploration/exploitation: when the surrogate is uncertain, bootstrap models disagree and propose diverse candidates; when confident, they converge.

4. **Expected Improvement**: From the pooled candidates, EI selects the most promising for real evaluation — balancing predicted quality (exploit) with surrogate uncertainty (explore).

5. **Iterative refinement**: Each round evaluates ~20 candidates with the real model, refits the surrogate, and repeats. The surrogate sharpens exactly where it matters (near the optimum). Typically converges in 2-4 rounds.

```
exploration_results.json (from explore_placements)
        │
        ▼
  ┌─────────────┐
  │  Fit Ridge  │  50 bootstrap resamples
  │  surrogate  │  → 50 sets of MRF potentials
  └──────┬──────┘
         │
    ┌────┴────┐
    ▼         ▼         ...
  Viterbi   Viterbi          (one per bootstrap model)
  DP #1     DP #2            each finds exact top-N placements
    │         │               within feasible allocations
    └────┬────┘
         ▼
  Pool & deduplicate candidates
         │
         ▼
  Score with Expected Improvement (μ + uncertainty)
         │
         ▼
  Evaluate top-K with real model     ← only ~20 real evals per round
         │
         ▼
  Refit surrogate with new data
         │
         ▼
  Repeat (2-4 rounds)
```

#### Why This Design?

| Property | Benefit |
|----------|---------|
| **Exact DP** | Globally optimal placement per allocation — no greedy approximation |
| **Layer interactions** | Pairwise/triplet MRF potentials capture how mixer choices at adjacent layers affect each other |
| **Bootstrap diversity** | Automatic exploration/exploitation — no tuning required |
| **Expected Improvement** | Directs real evaluations where the surrogate is most uncertain |
| **Data reuse** | Builds on exploration dataset — no wasted evaluations |

</details>

**Tuning parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `search_strategy.num_rounds` | 4 | Refinement iterations (2-4 usually sufficient) |
| `search_strategy.evals_per_round` | 500 | Real evaluations per round |
| `search_strategy.candidates_per_model` | 200 | Top-N placements from each bootstrap Viterbi solve |
| `search_strategy.ei_xi` | 0.01 | EI exploration parameter (higher = more exploration) |

### Validate Winning Placement

Evaluate the winning placement with generative evaluation:

```bash
python -m place_layers +experiments=eval_gsm8k_rest \
    +model=apriel2_1p6_v2_annealed_0204 \
    placement='[attention, gdn, attention, ...]'
```

## Evaluation Commands

### Generative Evaluation

Generate completions and check correctness. Accurate but slow.

```bash
# Single placement
python -m place_layers +experiments=eval_gsm8k_rest \
    +model=apriel2_1p6_v2_annealed_0204 +placement=all-attention

# Sweep over placements (multirun)
python -m place_layers +experiments=eval_gsm8k_rest \
    +model=apriel2_1p6_v2_annealed_0204 \
    +placement=all-attention,every2nd-gdn,every4th-gdn \
    --multirun
```

### Likelihood Evaluation

Multiple-choice scoring (MMLU). Faster than generative - scores each answer choice without generation.

```bash
python -m place_layers +experiments=eval_mmlu_rest \
    +model=apriel2_1p6_v2_annealed_0204 +placement=all-attention
```

### IS Evaluation

Fast evaluation using importance sampling with precomputed traces. See [Precompute Traces](#precompute-traces) for trace generation. Use `+experiments=dev/eval_gsm8k_is` for a pre-configured setup.

### Available Placements

Pattern-based configs (repeat cyclically across all layers):

- `all-attention` - baseline, all attention layers
- `every2nd-gdn` - alternating attention/GDN (24 GDN layers for 48-layer model)
- `every4th-gdn` - GDN at every 4th layer (12 GDN layers)
- `every2nd-kda`, `every4th-kda` - KDA variants
- `every2nd-swa`, `every4th-swa` - SWA (sliding window) variants
- `all-gdn`, `all-kda`, `all-swa` - all one mixer type

Structural configs (48-layer model):

- `bookend-attn-gdn` - first 2 + last 2 layers attention, rest GDN
- `bookend-attn-kda`, `bookend-attn-swa` - same pattern, other mixers

## Precompute Traces

Generate teacher completions with log-probabilities. Required for IS evaluation and placement optimization.

```bash
# Start teacher model server, then:
python -m place_layers +command=precompute_traces \
    +backend=rest +task=gsm8k +generation=sampling
```

Output: `validation.jsonl` with traces containing `teacher_log_prob` for each completion. See `conf/experiments/dev/precompute_gsm8k_vllm.yaml` for an example setup.

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration. Key config groups:

| Group | Options | Description |
|-------|---------|-------------|
| `command` | evaluate, optimize, explore_placements, validate_landscape, placement_landscape, precompute_traces, serve | What to run |
| `evaluator` | generative, likelihood, importance_weighted, extraction | How to score placements |
| `backend` | **local_vllm**, **rest**, transformers, vllm, rest_dns | Model inference backend |
| `model` | **apriel2_1p6_v2_annealed_0204**, apriel2_1p6_v2_0202, apriel2_0_5b_dev | Model loading config |
| `task` | gsm8k, gsm8k_quick, mmlu, mmlu_optimize | Evaluation task |
| `generation` | greedy, greedy_short, sampling | Generation parameters |
| `placement` | all-attention, every2nd-gdn, every4th-gdn, ... | Layer placement |
| `cost_model` | linear_apriel_15b, uniform_apriel_15b | Per-mixer cost coefficients |
| `landscape` | apriel_15b | Shared landscape settings |
| `search_strategy` | viterbi_surrogate, cost_constrained_hierarchical | Optimization algorithm |
| `problem` | cost_max_perf_apriel_15b_v2, cost_apriel_0_5b | Optimization problem definition |

Development configs (`conf/experiments/dev/`, `mock_*` evaluators) exist for local testing with the smaller Apriel2-0.5B-dev model. Analysis tools (`calibrate_sa`, `noise_analysis` commands) are available for SA tuning and evaluator noise estimation.

See `CLAUDE.md` for detailed configuration patterns and architecture documentation.

## Output

Results are saved to `outputs/<config-name>/`:
- `evaluation_results_{objective}.json` - per-objective scores and placement details
- `validation.jsonl` - individual completions (for generative eval)
- `exploration_progress.jsonl` - incremental surrogate exploration data
- `exploration_results.json` - final exploration dataset
- `validation_results.json` - per-band surrogate vs. real scores
- `placement_landscape.pdf` / `placement_landscape.png` - landscape plot
- `landscape_data.json` - landscape plot data
- `.hydra/config.yaml` - resolved configuration
