# Per-Mixer CUDA Graph Caching for Apriel2 Stochastic Mixer

## Context

The `Apriel2StochasticMixer` currently forces PIECEWISE CUDA graph mode, which means the sub-mixer forward (attention/GDN/KDA) runs **fully eagerly** at every decode step — no CUDA graph capture or replay. This adds kernel launch overhead (~5-10us per kernel × ~10 kernels per mixer × 24 layers = ~1-2ms per decode step). By caching a CUDA graph per mixer variant per batch size, we can replay pre-captured graphs during decode, reducing the per-step overhead to a single graph replay per layer.

**File**: `/home/toolkit/dev/Fast-LLM/fast_llm_external_models/apriel2/vllm/modeling_apriel2.py` (all changes in this single file)

## Architecture

```
Current:  [norm graph] → [eager dispatch → eager mixer.forward()] → [MLP graph]
Proposed: [norm graph] → [eager dispatch → replay mixer CUDA graph] → [MLP graph]
```

The `stochastic_mixer_dispatch` custom op remains a graph-splitting op (runs eagerly for Python mixer selection). The change is that *within* the dispatch, instead of calling the mixer eagerly, we replay a pre-captured CUDA graph.

Input/output tensors from the enclosing piecewise graph pieces are at **stable addresses** for a given batch size, satisfying CUDA graph replay requirements.

### Why this works

1. **Input buffer stability**: `hidden_states` and `output` arrive from the previous PIECEWISE graph piece's output — stable addresses per batch size.
2. **KV cache stability**: Pre-allocated at fixed addresses, updated in-place before each forward.
3. **No graph nesting**: `stochastic_mixer_dispatch` is a graph-splitting op, so it runs OUTSIDE any outer `torch.cuda.graph()` context. We can freely start our own captures here.
4. **ForwardContext access**: GDN/KDA core ops call `get_forward_context()` for `attn_metadata` and `kv_cache`. Python code runs during capture (not replay), but metadata buffers are at fixed addresses and updated in-place before replay — same pattern vLLM uses for attention in FULL mode.

## Implementation Steps

### Step 1: Add env var flag and new imports (~line 86)

```python
import os
APRIEL2_MIXER_CUDA_GRAPHS: bool = os.environ.get("APRIEL2_MIXER_CUDA_GRAPHS", "1") == "1"
```

New imports needed:
- `from vllm.compilation.monitor import cudagraph_capturing_enabled, validate_cudagraph_capturing_enabled`
- `from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id`
- `from vllm.utils.torch_utils import current_stream`

(`current_platform` and `dataclass` already imported)

### Step 2: Add `MixerGraphEntry` and `MixerGraphCache` (~line 120)

**`MixerGraphEntry`**: Simple dataclass holding `torch.cuda.CUDAGraph` + captured pointer addresses (for debug assertions).

**`MixerGraphCache`**: Dict-based cache keyed by `(mixer_name: str, num_tokens: int)`.
- `capture(mixer_name, num_tokens, mixer_fn, hidden_states, output, positions)` — wraps `torch.cuda.graph()` with the global graph pool, calls `validate_cudagraph_capturing_enabled`, stores the entry
- `replay(mixer_name, num_tokens, hidden_states, output)` — replays the cached graph, asserts pointer stability in debug mode
- `has(mixer_name, num_tokens)` — check existence
- Uses `current_platform.get_global_graph_pool()` (same pool as vLLM's piecewise graphs)

### Step 3: Add `_batch_has_prefill()` helper (~line 865)

CUDA graphs captured during decode cannot handle prefill. Detect mixed batches by:
1. If `cudagraph_runtime_mode == CUDAGraphMode.NONE` → True (profile/warmup run)
2. For GDN/KDA: check `GDNAttentionMetadata.num_prefills > 0` via `forward_context.attn_metadata[mixer.prefix]`
3. Default: False (assume decode-only, safe for attention mixers)

### Step 4: Add `_capture_all_mixers_for_num_tokens()` helper (~line 870)

Called during vLLM's `capture_model()` phase. For each mixer variant:
1. `torch.cuda.synchronize()` + `output.zero_()` between captures
2. `cache.capture(mixer_name, ...)` using the **same `output` buffer** for all mixers (ensures all captured graphs write to the same stable address)
3. After all captures, run active mixer eagerly once to restore correct `output` for downstream layers

**Why same output buffer**: On replay, each graph writes to the address captured. All graphs captured with the same `output` address means any mixer can be replayed into the current `output` buffer. No scratch buffers needed.

### Step 5: Replace `stochastic_mixer_dispatch` body (~line 870)

Three-way dispatch:

```
1. if cache is None → eager (feature disabled)
2. if cudagraph_capturing_enabled → _capture_all_mixers_for_num_tokens() (capture phase)
3. if not has_prefill and cache.has(active, num_tokens) → cache.replay() (fast path)
4. else → eager fallback (uncached batch size or mixed batch)
```

### Step 6: Wire cache into `Apriel2StochasticMixer.__init__` (~line 2452)

After `_force_piecewise_cudagraph_for_stochastic_mixers()`:
```python
self._mixer_graph_cache = MixerGraphCache() if APRIEL2_MIXER_CUDA_GRAPHS else None
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Manual `torch.cuda.graph()` vs vLLM's `CUDAGraphWrapper` | `CUDAGraphWrapper` manages output memory (weak refs) and is coupled to `PiecewiseBackend`. Our output is externally owned by the piecewise graph piece. Manual capture is simpler and avoids fighting the wrapper's assumptions. |
| Capture ALL mixers during `capture_model()` | After `capture_model()`, `cudagraph_capturing_enabled=False` and NCCL leaves graph-safe mode. Lazy capture during inference would fail for TP>1. Capture everything upfront. |
| Same `output` buffer for all mixer captures | All graphs write to the same stable address → any graph can replay into the current `output`. No scratch buffers, no `cudaMemcpy`. |
| `torch.cuda.synchronize()` between captures | Prevents kernel overlap between separate graph captures. One-time cost during startup only. |
| Env var `APRIEL2_MIXER_CUDA_GRAPHS` | Follows vLLM convention (`VLLM_*` env vars). Easy to disable without code changes. Default: enabled. |

## Memory Cost

`num_layers × num_mixers × num_batch_sizes` graphs. For 24 layers × 3 mixers × ~30 batch sizes = ~2160 small graphs. Each contains one mixer layer's kernels (small memory footprint). Shared graph pool reuses memory across graphs.

## What Does NOT Change

- PIECEWISE mode for the outer model (stochastic_dispatch remains a graph break)
- The `stochastic_mixer_dispatch` custom op registration and fake impl
- The `_force_piecewise_cudagraph_for_stochastic_mixers()` function
- Sub-mixer classes (Attention, GDN, KDA) — no modifications needed
- Placement switching via `set_layer_placements` — KV cache addresses are stable, so cached graphs remain valid after switch
- Weight loading, kv_cache_spec, or any other mixer functionality

## Key vLLM Integration Points

| Component | Import Path | Usage |
|-----------|-------------|-------|
| `cudagraph_capturing_enabled` | `vllm.compilation.monitor` | Detect capture phase (True only during `capture_model()`) |
| `validate_cudagraph_capturing_enabled` | `vllm.compilation.monitor` | Assert we're in valid capture context |
| `set_graph_pool_id` | `vllm.distributed.device_communicators.pynccl_allocator` | Share graph memory pool with vLLM's piecewise graphs |
| `current_stream` | `vllm.utils.torch_utils` | Capture on correct CUDA stream |
| `current_platform.get_global_graph_pool()` | `vllm.platforms` (already imported) | Get the shared graph pool handle |
| `ForwardContext.cudagraph_runtime_mode` | `vllm.forward_context` (already imported) | Detect eager vs graph mode |
| `GDNAttentionMetadata.num_prefills` | `vllm.v1.attention.backends.gdn_attn` (already imported) | Detect mixed prefill+decode batches |

## Edge Cases

| Case | Behavior |
|------|----------|
| Mixed batch (prefill+decode) | `_batch_has_prefill()` → True → eager fallback |
| Profile/warmup run | `cudagraph_runtime_mode == NONE` → eager |
| Uncached batch size | `cache.has()` → False → eager fallback |
| Placement switch | KV cache at same addresses → cached graphs remain valid |
| TP>1 | NCCL in graph-safe mode during `capture_model()` → safe capture |
| Feature disabled | `APRIEL2_MIXER_CUDA_GRAPHS=0` → `cache is None` → all eager |

## Verification

1. **Correctness**: Run model with `APRIEL2_MIXER_CUDA_GRAPHS=1` vs `=0`, compare outputs for same prompt/placement — should be identical
2. **Mixer switching**: Capture with attention active, switch to GDN, verify GDN graph replays (check `cache.has("gdn", batch_size)` is True)
3. **Prefill fallback**: Verify mixed batches take eager path (log message or debug flag)
4. **Performance**: Measure decode latency per token with graphs enabled vs disabled
5. **TP>1**: Run with tensor parallelism, verify NCCL operations in captured graphs work correctly
6. **Disable flag**: `APRIEL2_MIXER_CUDA_GRAPHS=0` should produce identical behavior to current code

---

## Implementation Summary

All changes are in a single file: `fast_llm_external_models/apriel2/vllm/modeling_apriel2.py`.

### What was added

**New imports** (lines 11, 15, 65-66):
- `os`, `Callable`
- `cudagraph_capturing_enabled`, `validate_cudagraph_capturing_enabled` from `vllm.compilation.monitor`
- `set_graph_pool_id` from `vllm.distributed.device_communicators.pynccl_allocator`
- `current_stream` from `vllm.utils.torch_utils`

**Feature flag** (line 110):
```python
APRIEL2_MIXER_CUDA_GRAPHS: bool = os.environ.get("APRIEL2_MIXER_CUDA_GRAPHS", "0") == "1"
```
Default: **disabled**. Set `APRIEL2_MIXER_CUDA_GRAPHS=1` to enable.

**`MixerGraphEntry` dataclass** (line 119): Holds `torch.cuda.CUDAGraph` + captured `input_ptr`/`output_ptr` for debug pointer-stability assertions.

**`MixerGraphCache` class** (line 127): Dict-based cache keyed by `(mixer_name, num_tokens)`.
- `capture()` — validates we're in capture phase, records graph via `torch.cuda.graph()` using global graph pool
- `replay()` — replays cached graph with debug pointer assertions
- `has()` — existence check

**`_batch_has_prefill()` helper** (line 966): Detects mixed prefill+decode batches by checking `GDNAttentionMetadata.num_prefills` or `cudagraph_runtime_mode == NONE`. Returns True → forces eager fallback.

**`_capture_all_mixers_for_num_tokens()` helper** (line 987): During `capture_model()`, iterates all mixer variants, captures each into the **same `output` buffer** (so all graphs write to the same stable address). Runs `torch.cuda.synchronize()` + `output.zero_()` between captures. Restores correct output by running active mixer eagerly at the end.

### What was modified

**`stochastic_mixer_dispatch()`** (line 1026): Three-way dispatch replacing the previous direct eager call:
1. **Capture phase** (`cudagraph_capturing_enabled` is True) → calls `_capture_all_mixers_for_num_tokens()` to capture all mixer variants
2. **Replay** (no prefill, cached graph exists) → `cache.replay()` for the active mixer
3. **Eager fallback** (feature disabled, uncached batch size, or mixed batch) → direct `active_mixer()` call as before

**`Apriel2StochasticMixer.__init__`** (line 2626): Added `self._mixer_graph_cache` initialization, conditional on the env var flag.
