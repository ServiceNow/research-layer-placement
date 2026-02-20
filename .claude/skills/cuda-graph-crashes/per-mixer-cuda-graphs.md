# Per-Mixer CUDA Graph Caching for Apriel2 Stochastic Mixer

## vLLM Startup Phases and CUDA Graph Lifecycle

vLLM's GPU model runner goes through distinct phases during startup. Understanding these is critical for any code that captures or replays CUDA graphs.

### Phase 1: `profile_run()` — Memory Profiling

- Runs the model forward **once** with the maximum possible batch size (fills all GPU memory)
- Purpose: measure peak GPU memory usage so vLLM knows how many KV cache blocks to allocate
- **`cudagraph_runtime_mode = CUDAGraphMode.NONE`** — no graph capture/replay, purely eager
- **`cudagraph_capturing_enabled = True`** (the default in `vllm/compilation/monitor.py` — this is a pitfall!)
- After this run: `free_memory = total_gpu - peak_usage` → allocates KV cache blocks
- **cuBLAS is lazily initialized** during this phase (first matmul call triggers it)

### Phase 2: `capture_model()` — CUDA Graph Capture

- Runs the model forward **once per batch size** (e.g., 1, 2, 4, 8, ..., up to `max_num_seqs`)
- Purpose: pre-record GPU kernel sequences as CUDA graphs for fast replay during inference
- **`cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE`** — vLLM's piecewise backend captures graph pieces between split points
- **`cudagraph_capturing_enabled = True`** (explicitly set by vLLM around this phase)
- cuBLAS is already initialized (from profile_run), NCCL is in graph-safe mode (for TP>1)
- This is where per-mixer graphs should be captured

### Phase 3: Normal Inference

- **`cudagraph_capturing_enabled = False`**
- **`cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE`**
- Graphs are replayed, not captured

### Startup Sequence

```
load weights
  → profile_run()          # eager, measures memory, inits cuBLAS
  → allocate KV cache      # uses memory measurement
  → capture_model()        # captures CUDA graphs per batch size
  → ready for inference     # replays graphs
```

### Key Variables

| Variable | Location | profile_run | capture_model | Inference |
|----------|----------|------------|---------------|-----------|
| `cudagraph_capturing_enabled` | `vllm/compilation/monitor.py` | `True` (default!) | `True` (set) | `False` |
| `forward_context.cudagraph_runtime_mode` | `vllm/forward_context.py` | `NONE` | `PIECEWISE` | `PIECEWISE` |

**Critical pitfall**: `cudagraph_capturing_enabled` defaults to `True`, so checking only this flag is NOT sufficient to determine if we're in the actual capture phase. Always also check `runtime_mode != NONE`.

---

## PIECEWISE CUDA Graph Mode

The `Apriel2StochasticMixer` forces PIECEWISE mode via `_force_piecewise_cudagraph_for_stochastic_mixers()`. In this mode:

1. `torch.compile` traces the model and splits the FX graph at registered "splitting ops" (custom ops like attention, stochastic_mixer_dispatch)
2. Each compute piece between splits (norms, MLPs, etc.) is captured as a separate CUDA graph
3. Split-point ops run **eagerly** (Python code, not captured)

```
[norm graph piece] → [eager: stochastic_mixer_dispatch] → [MLP graph piece]
```

The `stochastic_mixer_dispatch` custom op runs eagerly at every decode step, selecting the active mixer and calling its forward. Without per-mixer CUDA graph caching, this means the sub-mixer forward (attention/GDN/KDA) runs fully eagerly — adding kernel launch overhead.

---

## Per-Mixer CUDA Graph Caching Architecture

### Goal

Cache a CUDA graph per mixer variant per batch size, so that instead of running the sub-mixer eagerly, we replay a pre-captured graph.

```
Before: [norm graph] → [eager dispatch → eager mixer.forward()] → [MLP graph]
After:  [norm graph] → [eager dispatch → replay mixer CUDA graph] → [MLP graph]
```

### Components

**`MixerGraphEntry`** (dataclass): Holds `torch.cuda.CUDAGraph` + captured pointer addresses (for debug assertions).

**`MixerGraphCache`**: Dict-based cache keyed by `(mixer_name: str, num_tokens: int)`.
- Uses `current_platform.get_global_graph_pool()` (same pool as vLLM's piecewise graphs)
- `capture()`: wraps `torch.cuda.graph()`, stores the entry
- `replay()`: replays the cached graph, asserts pointer stability in debug mode
- `has()`: check existence

**`_capture_all_mixers_for_num_tokens()`**: Called during capture_model() phase. For each mixer:
1. Eager warmup (triggers Triton autotuning outside graph capture)
2. `synchronize()` + `zero_()` (clean slate)
3. `cache.capture()` (graph capture)
4. After all captures, restore active mixer output eagerly

**`stochastic_mixer_dispatch()`**: Three-way dispatch:
1. Cache disabled → eager
2. Capture phase (`cudagraph_capturing_enabled` AND `runtime_mode != NONE`) → capture all mixers
3. Decode-only batch with cached graph → replay
4. Else → eager fallback

### Env Var

`APRIEL2_MIXER_CUDA_GRAPHS` — `"1"` to enable, `"0"` to disable. Default: `"0"`.

---

## Pitfalls Discovered During Implementation

### Pitfall 1: `cudagraph_capturing_enabled` Defaults to `True`

The module-level variable in `vllm/compilation/monitor.py` starts as `True`. This means during `profile_run()`, any code checking only `cudagraph_capturing_enabled` will think it's in capture mode.

**Symptom**: `CUBLAS_STATUS_NOT_INITIALIZED when calling cublasCreate(handle)` — cuBLAS lazy init can't happen inside `torch.cuda.graph()`.

**Fix**: Always check BOTH conditions:
```python
from vllm.config.compilation import CUDAGraphMode
runtime_mode = forward_context.cudagraph_runtime_mode
if (
    cudagraph_capturing_enabled
    and runtime_mode is not None
    and runtime_mode != CUDAGraphMode.NONE
):
    # Actually in capture phase
```

### Pitfall 2: Triton Autotuning During Graph Capture

Triton kernels decorated with `@triton.autotune` run benchmarks (including `cuda.synchronize()`) the first time they see a new input shape. This is illegal during CUDA stream capture.

**Symptom**: `CUDA error: operation not permitted when stream is capturing` from `triton/testing.py:do_bench → synchronize()`.

**Fix**: Run each mixer eagerly once before capturing:
```python
# Eager warmup — triggers Triton autotuning outside capture
mixer(hidden_states, output, positions=positions)
torch.cuda.synchronize()
output.zero_()
# Now safe to capture
cache.capture(mixer_name, num_tokens, mixer, hidden_states, output, positions)
```

### Pitfall 3: Capture Must Happen During `capture_model()`, Not Lazily

After `capture_model()`, NCCL leaves graph-safe mode. Lazy capture during inference would fail for TP>1. All mixer variants must be captured upfront during the capture phase.

### Pitfall 4: All Mixers Must Capture With Same Output Buffer

All graphs write to the address captured. If all graphs are captured with the same `output` buffer address, any mixer can be replayed into the current `output` buffer without needing scratch buffers or `cudaMemcpy`.

Input/output tensors from the enclosing piecewise graph pieces are at **stable addresses** for a given batch size, satisfying CUDA graph replay requirements.

---

## Memory Cost

`num_layers × num_mixers × num_batch_sizes` graphs. For 24 layers × 3 mixers × ~35 batch sizes = ~2520 small graphs. Each contains one mixer layer's kernels (small memory footprint). Shared graph pool reuses memory across graphs.

---

## What Does NOT Change

- PIECEWISE mode for the outer model (stochastic_dispatch remains a graph break)
- The `stochastic_mixer_dispatch` custom op registration and fake impl
- The `_force_piecewise_cudagraph_for_stochastic_mixers()` function
- Sub-mixer classes (Attention, GDN, KDA) — no modifications needed
- Placement switching via `set_layer_placements` — KV cache addresses are stable
- Weight loading, kv_cache_spec, or any other mixer functionality

---

## Verification

1. **Correctness**: `APRIEL2_MIXER_CUDA_GRAPHS=1` vs `=0`, compare outputs — should be identical
2. **Mixer switching**: Capture with attention active, switch to GDN, verify GDN graph replays
3. **Prefill fallback**: Mixed batches take eager path (check `_batch_has_prefill`)
4. **Performance**: Measure decode latency per token with graphs enabled vs disabled
5. **TP>1**: Verify NCCL operations in captured graphs work correctly
6. **Disable flag**: `APRIEL2_MIXER_CUDA_GRAPHS=0` produces identical behavior to pre-change code
