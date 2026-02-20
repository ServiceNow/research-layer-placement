# Place Layers Project Guide

This codebase implements inference benchmarking for supernet layer placement optimization using Hydra + Pydantic configuration patterns.

> **Note**: For generic Hydra+Pydantic patterns (discriminated unions, smart constructors, adding new components), see the `hydra-pydantic` skill which will be auto-triggered when relevant.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Entry Point and Commands](#entry-point-and-commands)
4. [Backend Implementations](#backend-implementations)
5. [Evaluator Class Hierarchy](#evaluator-class-hierarchy)
6. [Search Strategies](#search-strategies)
7. [Trace System](#trace-system)
8. [Server Implementation](#server-implementation)
9. [Hydra Config Directory Organization](#hydra-config-directory-organization)
10. [Testing](#testing)
11. [Key Dependencies](#key-dependencies)
12. [Key File Locations](#key-file-locations)
13. [Apriel2 vLLM Model Architecture](#apriel2-vllm-model-architecture)
14. [Analysis Documents](#analysis-documents)

---

## Project Overview

The framework evaluates and optimizes **layer placements** in supernet models - models with multiple mixer types per layer (e.g., attention, GDN, KDA). A placement assigns each layer to a specific mixer type.

**Core concepts**:
- **Supernet**: A model with multiple mixer implementations per layer
- **LayerPlacement**: `tuple[int, ...]` mapping layer index to mixer index
- **Evaluator**: Measures accuracy for a given placement
- **Search Strategy**: Finds optimal placements via various algorithms

---

## Architecture

```
User CLI
   ↓
Hydra (config_path=conf, config_name=config)
   ↓
CommandAdapter.validate_python() → discriminated union
   ↓
Command.run() [async]
   ├─→ EvaluateCommand
   ├─→ OptimizeCommand
   │   ├─→ Problem.evaluate() → Evaluator
   │   ├─→ SearchStrategy.search()
   │   └─→ ResultHandler.save()
   └─→ Other commands...

Backend Hierarchy:
  AsyncBackend (vLLM, REST)
  └─ Concurrent streaming results

  SyncBackend (Transformers)
  └─ Internal batching
```

---

## Entry Point and Commands

**Main entry point**: `src/place_layers/__main__.py`

The entry point:
1. Registers OmegaConf resolvers: `path_sanitize_fragment`, `path_hash`, `path_truncate`
2. Loads Hydra config from `conf/config.yaml`
3. Validates config via `command_adapter.validate_python()` (discriminated union)
4. Runs the selected command async via `asyncio.run(command.run())`

**Available commands** (defined in `src/place_layers/command/__init__.py`):

| Command | Config | Purpose |
|---------|--------|---------|
| `evaluate` | `EvaluateCommandConfig` | Evaluate a single placement |
| `optimize` | `OptimizeCommandConfig` | Run placement optimization |
| `precompute_traces` | `PrecomputeTracesCommandConfig` | Generate teacher traces |
| `serve` | `ServeCommandConfig` | Start HTTP backend server |
| `landscape_analysis` | `LandscapeAnalysisCommandConfig` | Analyze search landscape |
| `convergence_analysis` | `ConvergenceAnalysisCommandConfig` | Study convergence |
| `sensitivity_analysis` | `SensitivityAnalysisCommandConfig` | Sensitivity studies |
| `hyperparameter_analysis` | `HyperparameterAnalysisCommandConfig` | Hyperparameter sweeps |
| `calibrate_sa` | `CalibrateSACommandConfig` | Simulated annealing calibration |

---

## Backend Implementations

Backends handle model inference. Located in `src/place_layers/backend/`.

### Backend Types

**Abstract bases** (`base.py`):

| Class | Use Case | Key Difference |
|-------|----------|----------------|
| `AsyncBackend` | Remote servers (vLLM, REST) | Concurrent requests, yields as complete |
| `SyncBackend` | Local models (Transformers) | Internal batching, processes in chunks |

**Common interface**:
- `generate_chat()` / `generate_text()` - Text generation
- `score_tokens()` / `score_text()` / `score_chat()` - Log-probability scoring
- `set_placement()` - Switch supernet placement
- `model_structure` - Model info (num_layers, mixer_names)

### Concrete Implementations

| Backend | File | Type | Supports Placement |
|---------|------|------|-------------------|
| `TransformersBackend` | `transformers.py` | Sync | Yes |
| `VLLMBackend` | `vllm.py` | Async | No |
| `RESTBackend` | `rest.py` | Async | Yes (remote) |
| `MockBackend` | `mock.py` | Sync | Yes (synthetic) |

### Batching Helpers (`base.py`)

- `batched_map[T, R]()` - Simple batch processing
- `batched_expand_map[]()` - Fan-out/batch/fan-in (for multiple-choice)
- `concurrent_expand_map[]()` - Async concurrent alternative

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

## Search Strategies

Located in `src/place_layers/search_strategy/`.

### Base Class

```python
class BaseSearchStrategy[ParametersT]:
    async def search(self, problem: BaseProblem[ParametersT]) -> SearchResult[ParametersT]:
        """Find optimal parameters."""
```

### Implementations

| Strategy | File | Description |
|----------|------|-------------|
| `ExhaustiveSearch` | `exhaustive_search.py` | Enumerate all valid placements (ground truth) |
| `HierarchicalBeamSearch` | `hierarchical_beam_search.py` | Layer-by-layer greedy beam search |
| `CostConstrainedBeamSearch` | `cost_constrained_beam_search.py` | Two-level: allocations in cost order, beam within |

### Cost-Constrained Search Modes

`CostConstrainedBeamSearch` supports two optimization modes:
- **MAX_PERFORMANCE**: Maximize accuracy subject to cost ≤ budget
- **MIN_COST**: Minimize cost subject to accuracy ≥ target

### Problem Interface

Problems are defined in `src/place_layers/problem/`:

```python
class BaseProblem[ParametersT]:
    evaluator: BaseEvaluator[ParametersT]
    async def evaluate(self, parameters: ParametersT) -> float
```

Concrete problems: `LayerPlacementProblem`, `CostConstrainedProblem`

---

## Trace System

Located in `src/place_layers/evaluator/traces/`.

**Key principle**: JSONL trace files are the single source of truth. State is reconstructed by scanning files.

### Directory Structure

```
output_dir/
├── metadata.json       # Config hash, model, placement
├── validation.jsonl    # Traces (Generation, Likelihood, etc.)
└── skipped.jsonl       # Skipped problems (errors)
```

### Trace Record Types (`types.py`)

| Type | Purpose |
|------|---------|
| `Generation` | Direct generation results |
| `ChatLikelihoodRecord` | Chat-format likelihood scoring |
| `TextLikelihoodRecord` | Text-format likelihood scoring |
| `ISProblemRecord` | Importance sampling records |
| `SkippedProblem` | Error/skip records |

### TraceWriter (`writer.py`)

- Appends traces to JSONL files
- Tracks metadata (model, config hash, placement)
- Supports resumability via `get_completed_problems()`
- Validates config hash on resume to prevent mixing runs

---

## Server Implementation

Located in `src/place_layers/server/`.

Wraps any backend in a FastAPI REST API for remote access.

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/info` | GET | Model structure and placement info |
| `/health` | GET | Health check |
| `/generate/chat` | POST | Chat generation |
| `/generate/text` | POST | Text generation |
| `/score/tokens` | POST | Token scoring |
| `/score/text` | POST | Text continuation scoring |
| `/score/chat` | POST | Chat continuation scoring |
| `/placement` | GET/PUT | Get/set dynamic placement |

### Server Adapter (`adapter.py`)

Handles backend differences:
- **AsyncBackend**: Thin wrapper, forwards directly
- **SyncBackend**: Queue-based batching with configurable batch size

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
| Output logs contain multiple runs | Hydra appends to existing output directories, so `app.log` files can contain logs from multiple consecutive runs with the same config. Check timestamps to identify run boundaries. |

---

## Testing

Tests are in `tests/` using pytest.

### Running Tests

```bash
# Run all tests (excluding slow)
pytest

# Run slow tests too
pytest -m ""

# Run specific test file
pytest tests/test_cost_constrained_search.py
```

### Test Configuration (`pyproject.toml`)

- `asyncio_mode = "auto"` - Async tests auto-detected
- `markers = ["slow"]` - Mark slow tests with `@pytest.mark.slow`
- Coverage enabled by default

### Key Test Files

| File | Tests |
|------|-------|
| `test_batching_helpers.py` | `batched_map`, `concurrent_expand_map` |
| `test_cost_constrained_search.py` | Cost-constrained algorithms |
| `test_rest_backend.py` | REST backend (with aioresponses mocks) |
| `test_importance_weighted.py` | Importance weighting |
| `test_lm_eval_utils_filters.py` | Filter handling |

---

## Key Dependencies

**Core**:
- `pydantic >= 2.0.0` - Data validation/serialization
- `hydra-core >= 1.3.0` - Configuration management
- `omegaconf >= 2.3.0` - Config objects

**Async/HTTP**:
- `aiohttp >= 3.9.0` - Async HTTP client
- `fastapi >= 0.109.0` - Web framework
- `uvicorn >= 0.27.0` - ASGI server

**ML**:
- `torch >= 2.0.0` (optional) - Deep learning
- `transformers >= 4.35.0` (optional) - HF models
- `lm-eval >= 0.4.0, < 0.4.10` - Benchmark evaluation

**Utilities**:
- `rich >= 13.0.0` - Pretty logging
- `tenacity >= 8.2.0` - Retry logic

---

## Key File Locations

| File | Purpose |
|------|---------|
| `src/place_layers/__main__.py` | Hydra entry point |
| `src/place_layers/command/__init__.py` | Command discriminated union |
| `src/place_layers/backend/base.py` | Abstract backend interfaces, batching |
| `src/place_layers/backend/transformers.py` | Local Transformers backend |
| `src/place_layers/backend/rest.py` | Remote REST backend |
| `src/place_layers/evaluator/base_evaluator.py` | Abstract evaluator interface |
| `src/place_layers/evaluator/backend_evaluator.py` | Backend lifecycle |
| `src/place_layers/evaluator/generative.py` | Direct generation evaluation |
| `src/place_layers/evaluator/likelihood.py` | Multiple-choice scoring |
| `src/place_layers/search_strategy/cost_constrained_beam_search.py` | Cost-aware search |
| `src/place_layers/evaluator/traces/writer.py` | Trace recording |
| `src/place_layers/server/app.py` | FastAPI server |
| `src/place_layers/core/types.py` | Core type definitions |

---

## Apriel2 vLLM Model Architecture

The primary model this project evaluates is **Apriel2**, a hybrid supernet with heterogeneous mixer types. Its vLLM implementation lives in a separate repo:

**File**: `/home/toolkit/dev/Fast-LLM/fast_llm_external_models/apriel2/vllm/modeling_apriel2.py`

### Mixer Types

| Mixer | Class | Custom Op | State Type |
|-------|-------|-----------|------------|
| Attention | `Apriel2Attention` | `vllm::unified_attention` (via vLLM `Attention`) | KV cache (FullAttentionSpec) |
| GDN (Gated DeltaNet) | `Apriel2GatedDeltaNet` | `vllm::apriel2_gdn_attention_core` | Recurrent state (MambaSpec) |
| KDA | `Apriel2KDAMixer` | `vllm::kda_attention` | Recurrent state (MambaSpec) |
| Mamba | `Apriel2MambaMixer` | `vllm::mamba_mixer` (via vLLM `MambaMixer`) | Recurrent state (MambaSpec) |
| Stochastic | `Apriel2StochasticMixer` | `vllm::stochastic_mixer_dispatch` | Delegates to active sub-mixer |

### Stochastic Mixer (Supernet Layer)

`Apriel2StochasticMixer` contains multiple sub-mixers in an `nn.ModuleDict`. At runtime, it routes to the **active** mixer via a custom op that causes a graph break in `torch.compile`.

- Each sub-mixer gets a **virtual layer index** (`layer_idx + (mixer_index+1) * num_layers`) for KV cache slot isolation
- All sub-mixer weights are loaded; switching is just changing `active_mixer_name`
- Placement switching: `model.set_layer_placements(["attention", "gdn", ...])` via `collective_rpc`
- After switching, KV cache is zeroed to prevent stale data from incompatible mixer formats

### `is_hybrid = False` Design Decision

`Apriel2ForCausalLM` deliberately sets `is_hybrid = False` (unlike `AprielHForCausalLM`). This avoids triggering `HybridAttentionMambaModelConfig.verify_and_update_config()`, which assumes **homogeneous** mamba layer shapes — it calls `model_cls.get_mamba_state_shape_from_config()` returning a **single** shape tuple. Apriel2 has heterogeneous blocks (GDN and KDA with different state dimensions), so the single-shape assumption breaks.

Instead, each layer implements its own `get_kv_cache_spec()` and calls `get_unified_page_size_for_config()` to compute the convex hull across all mixer types, ensuring compatible `page_size_bytes` for the unified KV cache allocator.

### CUDA Graph Modes

| Config | CUDA Graph Mode | Behavior |
|--------|----------------|----------|
| Non-stochastic (fixed placement) | FULL / FULL_AND_PIECEWISE | Standard vLLM capture |
| Stochastic (supernet) | **PIECEWISE** (forced) | Graph breaks at `stochastic_mixer_dispatch` so Python dispatch selects the active mixer at runtime |
| Stochastic + `APRIEL2_MIXER_CUDA_GRAPHS=1` | PIECEWISE + per-mixer graphs | Each sub-mixer's forward is captured as a separate CUDA graph during `capture_model()`, replayed on decode |

### Page Size Unification

For hybrid models, all KV cache pages must be the same size. `get_unified_page_size_for_config()` (module-level, cached) computes:
1. `attn_page_per_token` — bytes per token for attention KV cache
2. `max_mamba_page` — largest recurrent state across all mixer types
3. `block_size` — minimum attention block size such that `block_size * attn_page_per_token >= max_mamba_page`
4. Aligns to kernel requirements (16-token alignment) and hash block size (256)

### Key vLLM Integration Points

| Concept | Location | Notes |
|---------|----------|-------|
| `static_forward_context` | `compilation_config.static_forward_context[prefix]` | Registry for layers that need runtime lookup via custom ops |
| `no_compile_layers` | `ForwardContext.no_compile_layers` | Populated from `static_forward_context` each forward; used by custom ops to find layer instances |
| `_attention_ops` | `CompilationConfig._attention_ops` | List of custom ops that cause graph breaks in PIECEWISE mode |
| `cudagraph_capturing_enabled` | `vllm.compilation.monitor` | Global flag, True only during `capture_model()` |
| `GDNAttentionMetadata.num_prefills` | `vllm.v1.attention.backends.gdn_attn` | Detects mixed prefill+decode batches (CUDA graphs can't handle prefill) |

### Related Repos

| Repo | Path | Contains |
|------|------|----------|
| Fast-LLM | `/home/toolkit/dev/Fast-LLM/` | Apriel2 vLLM model (`fast_llm_external_models/apriel2/vllm/modeling_apriel2.py`) |
| apriel-internal | `/home/toolkit/dev/apriel-internal/` | Older AprielH model + vLLM plugin with `register_plugin.py` monkey-patching |
| vllm-SLAM | `/home/toolkit/dev/vllm-SLAM/` | Older SLAM fork with `AprielSSMHybridForCausalLM` |

---

## Analysis Documents

The `ai/` directory contains analysis and design documents:

| File | Purpose |
|------|---------|
| `ai/VLLM.md` | Comprehensive guide to vLLM KV cache memory management, paged attention, block pools, and hybrid model support |
| `ai/cuda-graph-crash-analysis.md` | Root cause analysis of CUDA graph crashes with pure-mamba placements; `is_hybrid=False` implications, `mamba_block_size` fix |
| `ai/gdn-throughput-analysis.md` | GDN throughput analysis |
| `ai/per-mixer-cuda-graph-plan.md` | Design + implementation for per-mixer CUDA graph caching in stochastic mixers (env var `APRIEL2_MIXER_CUDA_GRAPHS`) |
