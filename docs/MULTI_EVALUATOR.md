# Multi-Evaluator Guide

This guide explains how to use multiple evaluators simultaneously for evaluation and optimization.

## Quick Start

### Single Evaluator (simplest)

Create an inline evaluators config in your experiment:

```yaml
# experiments/my_eval.yaml
defaults:
  - /command: evaluate
  - /backend: transformers
  - /model: apriel2_0_5b_dev
  - /task: gsm8k
  - /generation: greedy

evaluators:
  configs:
    main:
      evaluator_type: generative
      backend: ${backend}
      tasks: ${task.tasks}
      generation: ${generation}
  objective:
    objective_type: weighted_average
    include_individual: false

placement: null
```

### Multiple Evaluators with Weights

Evaluate on multiple benchmarks and combine scores:

```yaml
# experiments/multi_gsm8k_mmlu.yaml
defaults:
  - /command: evaluate
  - /backend: transformers
  - /model: apriel2_0_5b_dev
  - /generation: greedy

evaluators:
  configs:
    gsm8k:
      evaluator_type: generative
      backend: ${backend}
      tasks: ["gsm8k"]
      generation: ${generation}
      max_problems_per_task: 100

    mmlu:
      evaluator_type: likelihood
      backend: ${backend}
      tasks: ["mmlu"]
      max_problems_per_task: 100

  objective:
    objective_type: weighted_average
    include_individual: false
    weights:
      gsm8k: 0.6
      mmlu: 0.4

placement: null
```

## Using Existing Evaluator Configs

To reuse existing `conf/evaluator/*.yaml` configs, use Hydra's package override syntax:

```yaml
defaults:
  - /command: evaluate
  - /evaluator@evaluators.configs.main: generative  # Load generative.yaml into dict
  - /backend: transformers
  - /model: apriel2_0_5b_dev
  - /task: gsm8k
  - /generation: greedy
```

For multiple evaluators, load each one into the dict:

```yaml
defaults:
  - /command: evaluate
  - /evaluator@evaluators.configs.gsm8k: generative
  - /evaluator@evaluators.configs.mmlu: likelihood
  - /backend: transformers
  - /model: apriel2_0_5b_dev
  - /generation: greedy

# Override specific evaluator configs
evaluators:
  configs:
    gsm8k:
      tasks: ["gsm8k"]
      max_problems_per_task: 100
    mmlu:
      tasks: ["mmlu"]
      max_problems_per_task: 100
  objective:
    objective_type: weighted_average
    include_individual: false
    weights:
      gsm8k: 0.6
      mmlu: 0.4
```

## Objective Strategies

Specify how to combine scores from multiple evaluators via the `objective:` block:

### `weighted_average` (default)
Combines scores using weights: `score = sum(weight_i * score_i)`

```yaml
evaluators:
  configs:
    gsm8k: {...}
    mmlu: {...}
  objective:
    objective_type: weighted_average
    include_individual: false
    weights:
      gsm8k: 0.6
      mmlu: 0.4
```

Weights must sum to 1.0. If no weights provided, uses uniform weights (1/N for each).

### `min`
Takes minimum score across all evaluators (optimize worst-case):

```yaml
evaluators:
  configs:
    gsm8k: {...}
    mmlu: {...}
  objective:
    objective_type: min
    include_individual: false
```

### `max`
Takes maximum score across all evaluators (optimize best-case):

```yaml
evaluators:
  configs:
    gsm8k: {...}
    mmlu: {...}
  objective:
    objective_type: max
    include_individual: false
```

### `independent`
Each evaluator is its own objective. Search solves N problems in parallel:

```yaml
evaluators:
  configs:
    gsm8k: {...}
    mmlu: {...}
  objective:
    objective_type: independent
```

### `include_individual` (aggregating objectives only)

The three aggregating objectives (`weighted_average`, `min`, `max`) support an
`include_individual` flag. When `true`, individual evaluator scores are exposed
as additional objectives alongside the aggregate:

```yaml
evaluators:
  configs:
    gsm8k: {...}
    mmlu: {...}
  objective:
    objective_type: weighted_average
    include_individual: true
    weights:
      gsm8k: 0.6
      mmlu: 0.4
```

This produces three objectives: `weighted_average`, `gsm8k`, and `mmlu` — the
aggregate plus per-evaluator breakdowns without needing `independent`.

## Output Organization

Each evaluator gets its own subdirectory with isolated outputs:

```
outputs/
└── +experiments=my_multi_eval/
    ├── evals/
    │   ├── gsm8k/
    │   │   ├── validation.jsonl       # GSM8K generations
    │   │   └── summary.jsonl          # GSM8K metrics
    │   ├── mmlu/
    │   │   ├── validation.jsonl       # MMLU scores
    │   │   └── summary.jsonl          # MMLU metrics
    │   └── importance/
    │       └── ...
    ├── evaluation_results.json         # Combined results
    └── app.log
```

The `evaluation_results.json` contains:
```json
{
  "timestamp": "2026-02-10T...",
  "evaluator_type": "multi[weighted_average]",
  "scores": {                   // Per-evaluator scores
    "gsm8k": 0.8123,
    "mmlu": 0.5893
  },
  "placement": [...],
  "placement_indices": [...]
}
```

## CLI Usage

### Evaluate Command

```bash
# Single evaluator
python -m place_layers +experiments=my_eval

# Multi-evaluator
python -m place_layers +experiments=multi_gsm8k_mmlu

# Override on CLI
python -m place_layers +experiments=multi_gsm8k_mmlu \
  evaluators.objective.weights.gsm8k=0.7 \
  evaluators.objective.weights.mmlu=0.3
```

### Optimize Command

Multi-evaluator works seamlessly with optimization:

```bash
python -m place_layers +command=optimize \
  +evaluators=multi_gsm8k_mmlu \
  +backend=transformers \
  +model=apriel2_0_5b_dev \
  +generation=greedy \
  +problem=layer_placement \
  +search_strategy=hierarchical_beam \
  problem.budgets=[5,5]
```

The search optimizes for the aggregated score. At the end, individual evaluator scores are logged for the best placement.

## Preset Configs

Several preset evaluator configs are available in `conf/evaluators/`:

- **`single_generative.yaml`**: Single generative evaluator
- **`multi_gsm8k_mmlu.yaml`**: GSM8K (60%) + MMLU (40%), weighted average
- **`multi_uniform_weights.yaml`**: Multiple evaluators with equal weights
- **`multi_with_importance.yaml`**: Generative + Likelihood + Importance-weighted

Use them with:
```bash
python -m place_layers +command=evaluate \
  +evaluators=multi_gsm8k_mmlu \
  +backend=transformers \
  +model=apriel2_0_5b_dev \
  +generation=greedy \
  placement=null
```

## Common Patterns

### Multi-dataset Evaluation
Evaluate performance across multiple benchmarks:

```yaml
evaluators:
  configs:
    gsm8k: {evaluator_type: generative, tasks: ["gsm8k"], ...}
    math: {evaluator_type: generative, tasks: ["math"], ...}
    mmlu: {evaluator_type: likelihood, tasks: ["mmlu"], ...}
  objective:
    objective_type: weighted_average
    include_individual: false
    weights: {gsm8k: 0.4, math: 0.3, mmlu: 0.3}
```

### Multi-method Evaluation
Compare different evaluation methods on same data:

```yaml
evaluators:
  configs:
    direct: {evaluator_type: generative, tasks: ["gsm8k"], ...}
    importance: {evaluator_type: importance_weighted, trace_dir: "/traces/gsm8k", ...}
  objective:
    objective_type: weighted_average
    include_individual: true
    weights: {direct: 0.8, importance: 0.2}
```

### Mixed Task Types
Combine reasoning, extraction, and multiple-choice:

```yaml
evaluators:
  configs:
    gsm8k: {evaluator_type: generative, tasks: ["gsm8k"], ...}          # reasoning
    fda: {evaluator_type: generative, tasks: ["fda"], ...}              # extraction
    mmlu: {evaluator_type: likelihood, tasks: ["mmlu"], ...}            # multiple-choice
  objective:
    objective_type: weighted_average
    include_individual: false
    weights: {gsm8k: 0.5, fda: 0.3, mmlu: 0.2}
```

## Migration from Old Configs

Old experiment configs used:
```yaml
defaults:
  - /evaluator: generative
```

Update to:
```yaml
defaults:
  - /evaluator@evaluators.configs.main: generative
```

This loads the existing `evaluator/generative.yaml` config into the evaluators dict.
