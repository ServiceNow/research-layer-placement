# Multi-Evaluator Guide

Evaluators produce raw benchmark scores; objectives map those scores to what search optimizes.

## Concepts: Evaluators vs Objectives

### Evaluators

The `configs:` dict defines one or more named evaluators. Each runs a benchmark and produces a scalar score. Keys are meaningful names (e.g. `gsm8k`, `mmlu`, `mock`).

```yaml
evaluators:
  configs:
    gsm8k:
      evaluator_type: generative
      # ...
    mmlu:
      evaluator_type: likelihood
      # ...
```

### Objectives

The `objective:` block maps N evaluator scores to M objective scores. Four types: `weighted_average`, `min`, `max`, `independent`.

All commands go through `evaluate_objectives()` — the objective layer is not optional, even with a single evaluator.

## Configuration

The top-level structure is `EvaluatorsConfig`:

```yaml
evaluators:
  configs:
    <name>: {evaluator_type: ..., ...}
  objective:
    objective_type: ...
```

### Single evaluator (most common)

Compose evaluators directly — no wrapper needed. Command configs provide a default objective (`weighted_average`) via `/objective@evaluators.objective: weighted_average` in their defaults list. You just add your evaluator:

```yaml
defaults:
  - /command: evaluate
  - /evaluator@evaluators.configs.gsm8k: generative
  - /backend: transformers
  - /model: apriel2_0_5b_dev
  - /task: gsm8k
  - /generation: greedy
```

The key name (`gsm8k`) is meaningful — use the task or evaluator name. The `weighted_average` objective auto-fills its weight to 1.0 for a single evaluator.

### Multiple evaluators inline

Define evaluators directly in a config file:

```yaml
# @package evaluators
configs:
  gsm8k:
    evaluator_type: generative
    tasks: ["gsm8k"]
    generation: ${generation}
  mmlu:
    evaluator_type: likelihood
    tasks: ["mmlu"]

objective:
  objective_type: weighted_average
  include_individual: false
  weights:
    gsm8k: 0.6
    mmlu: 0.4
```

### Multiple evaluators via defaults

Load existing `evaluator/*.yaml` configs into the dict using Hydra's package override:

```yaml
# @package evaluators
defaults:
  - /evaluator@configs.range1: mock_mrf_range1
  - /evaluator@configs.triplet: mock_mrf_triplet
  - /evaluator@configs.range2_triplet: mock_mrf_range2_triplet
  - /evaluator@configs.range3: mock_mrf_range3
  - _self_

objective:
  objective_type: independent
```

## Objective Types

### `weighted_average`

`score = sum(w_i * s_i)`. Weights must sum to 1.0. Empty weights dict = uniform (1/N for each).

```yaml
objective:
  objective_type: weighted_average
  include_individual: false
  weights:
    gsm8k: 0.6
    mmlu: 0.4
```

### `min`

Worst-case score across all evaluators.

```yaml
objective:
  objective_type: min
  include_individual: false
```

### `max`

Best-case score across all evaluators.

```yaml
objective:
  objective_type: max
  include_individual: false
```

### `independent`

Each evaluator is its own objective (N evaluators -> N objectives). Used for per-benchmark exploration.

```yaml
objective:
  objective_type: independent
```

### `include_individual` flag

The three aggregating types (`weighted_average`, `min`, `max`) support `include_individual: true`. This exposes per-evaluator scores as additional objectives alongside the aggregate.

```yaml
objective:
  objective_type: weighted_average
  include_individual: true
  weights:
    gsm8k: 0.6
    mmlu: 0.4
```

This produces three objectives: `weighted_average`, `gsm8k`, and `mmlu`.

## Evaluator Types

| Type | Purpose |
|------|---------|
| `generative` | Accuracy via generation (reasoning, extraction) |
| `likelihood` | Multiple-choice scoring (MMLU, ARC) |
| `importance_weighted` | Fast eval from pre-computed traces |
| `trace_likelihood` | Log-likelihood of pre-computed traces (with optional normalization) |
| `trace_precompute` | Generate traces for importance sampling |
| `mock_supernet` | Synthetic evaluator for testing |

### Trace normalization

`trace_likelihood` supports optional normalization to map raw log-likelihoods to [0, 1]:

```yaml
configs:
  aime24:
    evaluator_type: trace_likelihood
    trace_sources: ???
    normalization:
      type: linear
      low: -5505.18    # Maps to 0.0 (worst expected)
      high: -1137.85   # Maps to 1.0 (best expected)
```

This is useful when combining trace_likelihood evaluators with accuracy-based evaluators in a weighted_average objective, since raw log-likelihoods are on a very different scale.

## Preset Configs

Available in `conf/evaluators/`:

| File | Evaluators | Objective |
|------|-----------|-----------|
| `single_generative.yaml` | 1 generative (gsm8k) | `weighted_average` |
| `multi_gsm8k_mmlu.yaml` | generative + likelihood | `weighted_average` (60/40) |
| `multi_uniform_weights.yaml` | 3 evals (gsm8k, mmlu, arc_easy) | `weighted_average` (uniform) |
| `multi_with_importance.yaml` | 3 mixed types (generative, likelihood, importance) | `weighted_average` (50/30/20) |
| `apriel2_mix.yaml` | 6 evals (mmlu, fda, swde, aime24, aime25, math500) | `weighted_average` (uniform) |
| `mock_mrf_variants.yaml` | 4 mock MRF evaluators | `independent` |

## Output Organization

Each evaluator writes traces to its own subdirectory. Results are saved as per-objective files:

```
output_dir/
  evals/
    gsm8k/
      <placement>/
        validation.jsonl
    mmlu/
      <placement>/
        validation.jsonl
  evaluation_results_gsm8k.json
  evaluation_results_mmlu.json
```

Per-objective result JSON:

```json
{
  "timestamp": "2026-02-19T...",
  "placement": ["attention", "gdn", "attention", "..."],
  "placement_indices": [0, 1, 0, "..."],
  "objective_name": "gsm8k",
  "score": 0.8123
}
```

With `weighted_average` and two evaluators, you get three files: `evaluation_results_weighted_average.json`, `evaluation_results_gsm8k.json`, `evaluation_results_mmlu.json` (the last two only if `include_individual: true`).

## CLI Examples

Single evaluator (direct composition — no preset needed):

```bash
python -m place_layers +command=evaluate \
    +evaluator@evaluators.configs.gsm8k=generative \
    +backend=transformers +model=apriel2_0_5b_dev \
    +task=gsm8k +generation=greedy \
    placement=null
```

Override objective type on CLI:

```bash
python -m place_layers +command=evaluate \
    +evaluator@evaluators.configs.gsm8k=generative \
    +backend=transformers +model=apriel2_0_5b_dev \
    +task=gsm8k +generation=greedy \
    +objective@evaluators.objective=independent
```

Multi-evaluator preset:

```bash
python -m place_layers +command=evaluate \
    +evaluators=multi_gsm8k_mmlu \
    +backend=transformers +model=apriel2_0_5b_dev \
    +generation=greedy \
    placement=null
```

Override weights on CLI:

```bash
python -m place_layers +command=evaluate \
    +evaluators=multi_gsm8k_mmlu \
    +backend=transformers +model=apriel2_0_5b_dev \
    +generation=greedy \
    evaluators.objective.weights.gsm8k=0.7 \
    evaluators.objective.weights.mmlu=0.3 \
    placement=null
```

## Landscape Pipeline

The three landscape commands — `explore_placements`, `validate_landscape`, `placement_landscape` — form a pipeline that iterates over `evaluators.objective_names`. Each objective gets its own independent dataset, surrogate model, validation run, and plot. File naming uses `objective_result_path()`: `{prefix}_{objective_name}.json`.

### Step 1: explore_placements

Loops over `objective_names`, runs `_explore_objective()` for each. Each objective gets its own `ExplorationDataset` with scalar scores.

Saves per objective:
- `exploration_results_{obj_name}.json` — final dataset (assignments, scores, allocations)
- `exploration_progress_{obj_name}.jsonl` — incremental write for resumption

With `independent` objective and 4 evaluators: 4 completely separate exploration loops, each with its own adaptive sampling, surrogate quality tracking, and stopping criteria.

### Step 2: validate_landscape

Loops over `objective_names`, loads each `exploration_results_{obj_name}.json` from the explore output directory. Fits a per-objective Ridge surrogate, runs per-band Viterbi search with real evaluations.

Saves per objective:
- `validation_results_{obj_name}.json` — all band results with surrogate vs. validated scores
- `bands_{obj_name}/band_NNN/result.json` — per-band search result (for incremental resumption)
- `bands_{obj_name}/band_NNN/surrogate_dp.json` — surrogate DP solution for comparison

Resumable: completed bands are detected and skipped on restart.

### Step 3: placement_landscape

Loops over `objective_names`, loads exploration data per objective, optionally loads validation data. Generates per-objective density plots with optional validation overlay.

Saves per objective:
- `landscape_data_{obj_name}.json` — band edges, DP-optimal placements, metadata
- `placement_landscape_{obj_name}.pdf` / `.png` — two-panel figure (density + layermap)

### Data flow

The `{prefix}_{obj_name}.json` naming convention stitches the three steps together. Each step reads the previous step's output by constructing the same path via `objective_result_path()`:

```
explore_placements                validate_landscape              placement_landscape
─────────────────                ────────────────────            ─────────────────────
exploration_results_{obj}.json ──> load + fit surrogate ──────> load + fit surrogate
exploration_progress_{obj}.jsonl   validation_results_{obj}.json  landscape_data_{obj}.json
                                   bands_{obj}/band_NNN/       ──> load validation overlay
                                                                  placement_landscape_{obj}.pdf
```

The link between steps is `landscape.explore_output_dir` (points step 2/3 at step 1's output) and `placement_landscape.validation_output_dir` (points step 3 at step 2's output).

### How objective type affects the pipeline

**`weighted_average` (single aggregate)**:
- `objective_names = ("weighted_average",)` — one dataset, one surrogate, one plot
- Each sample evaluates ALL evaluators, combines via weights
- Output files: `exploration_results_weighted_average.json`, `validation_results_weighted_average.json`, etc.

**`independent` (per-benchmark)**:
- `objective_names = evaluator names` — N datasets, N surrogates, N plots
- Each sample evaluates only the needed evaluator
- Output files: `exploration_results_range1.json`, `exploration_results_triplet.json`, etc.

**`include_individual: true`**:
- `objective_names = (aggregate_name,) + evaluator_names` — (1+N) of everything
- The aggregate objective evaluates all evaluators; each individual objective evaluates only one

### Output directory structure

Full tree for `independent` with 4 mock MRF objectives across all 3 steps:

```
# Step 1: explore_placements output
explore_output/
  exploration_results_range1.json
  exploration_results_triplet.json
  exploration_results_range2_triplet.json
  exploration_results_range3.json
  exploration_progress_range1.jsonl
  exploration_progress_triplet.jsonl
  exploration_progress_range2_triplet.jsonl
  exploration_progress_range3.jsonl

# Step 2: validate_landscape output
validate_output/
  validation_results_range1.json
  validation_results_triplet.json
  validation_results_range2_triplet.json
  validation_results_range3.json
  bands_range1/
    band_000/result.json, surrogate_dp.json
    band_001/...
    ...
  bands_triplet/
    band_000/...
    ...
  bands_range2_triplet/...
  bands_range3/...

# Step 3: placement_landscape output
landscape_output/
  landscape_data_range1.json
  landscape_data_triplet.json
  landscape_data_range2_triplet.json
  landscape_data_range3.json
  placement_landscape_range1.pdf
  placement_landscape_range1.png
  placement_landscape_triplet.pdf
  placement_landscape_triplet.png
  placement_landscape_range2_triplet.pdf
  placement_landscape_range2_triplet.png
  placement_landscape_range3.pdf
  placement_landscape_range3.png
```

### CLI examples

**Multi-objective pipeline (4 independent mock evaluators):**

All three steps must use the same `evaluators` config so `objective_names` matches across the pipeline. The canonical mock experiments each use `/evaluators: mock_mrf_variants`:

```bash
# Step 1: explore (produces 4 datasets)
python -m place_layers +experiments=explore_mock

# Step 2: validate (reads 4 datasets, produces 4 validation results)
python -m place_layers +experiments=validate_mock \
    landscape.explore_output_dir=/path/to/explore/output

# Step 3: plot (reads 4 datasets + 4 validation results, produces 4 figures)
python -m place_layers +experiments=landscape_mock \
    landscape.explore_output_dir=/path/to/explore/output \
    validation_output_dir=/path/to/validate/output
```

**Single-objective pipeline (generative evaluator via REST):**

With direct composition + `weighted_average` (the default), all three steps iterate a single objective:

```bash
# Step 1: explore (produces 1 dataset: exploration_results_weighted_average.json)
python -m place_layers +experiments=explore_gsm8k_rest \
    +model=apriel2_1p6_v2_annealed_0204

# Step 2: validate
python -m place_layers +experiments=validate_gsm8k_rest \
    +model=apriel2_1p6_v2_annealed_0204 \
    landscape.explore_output_dir=/path/to/explore/output

# Step 3: plot
python -m place_layers +experiments=landscape_mock \
    +evaluator@evaluators.configs.gsm8k=generative \
    +backend=rest +task=gsm8k +generation=greedy \
    landscape.explore_output_dir=/path/to/explore/output \
    validation_output_dir=/path/to/validate/output
```

**Plot without validation (skip step 2):**

```bash
python -m place_layers +experiments=landscape_mock \
    landscape.explore_output_dir=/path/to/explore/output
```

When `validation_output_dir` is omitted (defaults to `null`), the density plot shows only the surrogate DP-optimal line without a validated overlay.
