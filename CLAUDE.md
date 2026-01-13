# Place Layers Project Guide

This codebase implements inference benchmarking for supernet layer placement optimization using Hydra + Pydantic configuration patterns.

> **Note**: For generic Hydra+Pydantic patterns (discriminated unions, smart constructors, adding new components), see the `hydra-pydantic` skill which will be auto-triggered when relevant.

---

## Table of Contents

1. [Evaluator Class Hierarchy](#evaluator-class-hierarchy)
2. [Hydra Config Directory Organization](#hydra-config-directory-organization)

---

## Evaluator Class Hierarchy

The evaluator subsystem uses a class hierarchy where each level adds specific functionality. Not all evaluators need all capabilities, so concrete evaluators branch off at different levels.

### Inheritance Tree

```
BaseEvaluator                         # Abstract interface only
├── MockSupernetEvaluator             # Mock evaluator (no backend needed)
└── BackendEvaluator                  # + Backend lifecycle, placement switching
    ├── ImportanceWeightedEvaluator   # Uses pre-computed traces, not lm-eval tasks
    └── TaskBasedEvaluator            # + lm-eval task loading, _prepare_task()
        ├── LikelihoodEvaluator       # Multiple-choice tasks (MMLU, etc.)
        └── BaseGenerativeEvaluator   # + Generation loop, sampling
            ├── GenerativeEvaluator   # Direct accuracy measurement
            └── TracePrecomputeEvaluator  # Generate traces for IS
```

### What Each Level Adds

| Class | Module | Adds |
|-------|--------|------|
| `BaseEvaluator` | `base_evaluator.py` | Abstract interface: `evaluate()`, `num_layers`, `num_mixers`, `mixer_names` |
| `BackendEvaluator` | `backend_evaluator.py` | Backend lifecycle (`__aenter__`/`__aexit__`), `_set_placement()`, `TraceWriter` |
| `TaskBasedEvaluator` | `task_based_evaluator.py` | `_prepare_task()` for lm-eval task loading, resumability via `get_completed_problems()` |
| `BaseGenerativeEvaluator` | `base_generative_evaluator.py` | Generation loop (`_run_tasks_sync/async`), progress bars, memory logging, abstract `_make_records()` |

### When to Use Each Base Class

- **Inherit from `BaseEvaluator`**: Mock/synthetic evaluators that don't need a real model
- **Inherit from `BackendEvaluator`**: Evaluators that use a model but not lm-eval tasks (e.g., importance-weighted uses pre-computed traces)
- **Inherit from `TaskBasedEvaluator`**: Evaluators that iterate over lm-eval tasks but have custom evaluation logic (e.g., likelihood-based scoring)
- **Inherit from `BaseGenerativeEvaluator`**: Evaluators that generate completions and need the full generation loop

---

## Hydra Config Directory Organization

The key principle is **minimal root config with compositional groups**.

### Directory Structure

```
conf/
├── config.yaml              # Minimal root - only mandatory settings
├── hydra/                   # Hydra-specific settings
│   ├── base.yaml           # Standard hydra config (logging, output dirs)
│   ├── job_logging/        # Job logging variants
│   └── hydra_logging/      # Hydra logging variants
│
├── command/                 # Command configs (@package _global_)
│   ├── evaluate.yaml       # command: evaluate + command-specific options
│   ├── optimize.yaml       # command: optimize + command-specific options
│   └── precompute_traces.yaml
│
├── evaluator/              # Evaluator configs (@package evaluator)
│   ├── generative.yaml     # Uses interpolation: ${backend}, ${task.tasks}
│   ├── importance_weighted.yaml
│   └── mock_supernet.yaml
│
├── backend/                # Backend configs (@package backend)
│   ├── transformers.yaml   # Local HuggingFace model
│   └── vllm.yaml          # Remote vLLM server
│
├── model/                  # Model configs (@package model)
│   ├── apriel2_0_5b_dev.yaml  # Apriel2-0.5B-dev supernet
│   └── qwen2_0_5b.yaml    # Qwen2-0.5B
│
├── task/                   # Task configs (@package task)
│   ├── gsm8k.yaml         # Full GSM8K evaluation
│   └── gsm8k_quick.yaml   # Quick 100-problem subset
│
├── generation/             # Generation configs (@package generation)
│   ├── greedy.yaml        # Deterministic decoding (temperature=0)
│   ├── greedy_short.yaml  # Short context (512 tokens)
│   └── sampling.yaml      # Stochastic sampling
│
├── placement/              # Placement configs (@package _global_)
│   ├── all_attention.yaml # Baseline: all attention layers
│   ├── every_2nd_gdn.yaml # GDN at every 2nd layer
│   ├── every_3rd_gdn.yaml # GDN at every 3rd layer
│   └── every_4th_gdn.yaml # GDN at every 4th layer
│
├── experiments/            # Standalone root configs (@package _global_)
│   ├── eval_gsm8k_attention.yaml
│   └── precompute_gsm8k_vllm.yaml
│
└── [problem/, search_strategy/, result_handler/]  # Other config groups
```

### Key Principles

#### 1. Minimal Root Config

```yaml
# config.yaml
defaults:
  - command: ???  # Must specify - forces user to choose a command
  - hydra: base   # Pull in standard hydra settings
  - _self_
```

#### 2. Command Configs are Global Scope

```yaml
# command/evaluate.yaml
# @package _global_

command: evaluate
placement: null
```

#### 3. Component Configs are Scoped

```yaml
# backend/transformers.yaml
# @package backend

backend_type: transformers
model_path: ${model.model_path}
torch_dtype: bfloat16
```

#### 4. Evaluators Use Interpolation

```yaml
# evaluator/generative.yaml
# @package evaluator

evaluator_type: generative
backend: ${backend}
tasks: ${task.tasks}
# Model structure (num_layers, mixer_names) discovered at runtime from backend
```

#### 5. Experiments Use Global Scope with Absolute Paths

```yaml
# experiments/eval_gsm8k_attention.yaml
# @package _global_

defaults:
  - /command: evaluate
  - /evaluator: generative
  - /backend: transformers
  - /model: apriel2_0_5b_dev
  - /task: gsm8k
  - /generation: greedy
  - /placement: all_attention
  - _self_
```

### Usage Patterns

```bash
# Run a pre-configured experiment
python -m place_layers +experiments=eval_gsm8k_attention

# Compose on CLI
python -m place_layers \
    +command=evaluate \
    +evaluator=generative \
    +backend=transformers \
    +model=apriel2_0_5b_dev \
    +task=gsm8k \
    +generation=greedy

# Override specific values
python -m place_layers \
    +experiments=eval_gsm8k_attention \
    task.max_problems_per_task=50

# Sweep over placements (multirun)
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

### Common Gotchas

| Issue | Fix |
|-------|-----|
| Missing `+` prefix | Use `+command=evaluate` not `command=evaluate` |
| Relative paths in experiments | Use `/command: evaluate` not `command: evaluate` |
| Hydra sub-defaults conflict | Use `override job_logging: rich_files` |
