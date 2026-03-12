# Per-Mixer CUDA Graph Caching — Results & Findings

## Goal

Bridge the throughput gap between PIECEWISE mode (supernet, ~290 tok/s) and FULL CUDA graph mode (fixed layout, ~900 tok/s) by caching a separate CUDA graph per mixer variant at each stochastic dispatch split point.

See `per-mixer-cuda-graph-plan.md` for the original design.

## Implementation Summary

**File**: `/home/toolkit/dev/Fast-LLM/fast_llm_external_models/apriel2/vllm/modeling_apriel2.py`

**Key components added**:

- `APRIEL2_MIXER_CUDA_GRAPHS` env var (default `"0"`, set `"1"` to enable)
- `MixerGraphEntry` dataclass — holds captured `torch.cuda.CUDAGraph` + pointer addresses
- `MixerGraphCache` class — per-mixer, per-batch-size cache keyed by `(mixer_name, num_tokens)`
- `_capture_all_mixers_for_num_tokens()` — captures graphs for all sub-mixers during `capture_model()`
- Modified `stochastic_mixer_dispatch` — capture/replay/eager dispatch logic
- `_batch_has_prefill()` — detects mixed prefill-decode batches (must fall back to eager)
- Cache wired into `Apriel2StochasticMixer.__init__`

## Bugs Encountered & Fixed

### Bug 1: CUBLAS_STATUS_NOT_INITIALIZED during profile_run

**Symptom**: `RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED` on first server startup.

**Root cause**: `cudagraph_capturing_enabled` in `vllm.compilation.monitor` defaults to `True`. During `profile_run()` (before `capture_model()`), our dispatch saw this flag and tried to capture CUDA graphs. But cuBLAS hasn't been lazily initialized yet (profile_run is the first model forward), and lazy init can't happen inside graph capture.

**Fix**: Added `forward_context.cudagraph_runtime_mode != CUDAGraphMode.NONE` check. During profile_run, runtime_mode is `NONE`. During actual capture_model(), it's `PIECEWISE`.

### Bug 2: Triton autotuning during CUDA graph capture

**Symptom**: `torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing` from KDA mixer's `fused_kda_gate` Triton kernel.

**Root cause**: First call to a Triton autotuned kernel at a new batch size triggers benchmarking, which calls `cuda.synchronize()` — illegal during CUDA graph stream capture.

**Fix**: Added an eager warmup call for each mixer before capture. The warmup triggers Triton autotuning outside graph capture. The subsequent capture call uses the cached autotuned config.

### Bug 3 (critical): Graph pool memory causing 2.2x throughput regression

**Symptom**: With `APRIEL2_MIXER_CUDA_GRAPHS=1`, warmup throughput drops from ~290 tok/s to ~126 tok/s even when graph replay is working correctly.

**Root cause**: Capturing ~5,040 CUDA graphs (48 layers × 3 mixers × ~35 batch sizes) consumes too much GPU memory, causing inference performance degradation.

**Investigation sequence**:

| Test | Warmup tok/s | Conclusion |
| ---- | ------------ | ---------- |
| `CUDA_GRAPHS=0` (baseline) | 290 | Normal |
| `CUDA_GRAPHS=1`, capture + replay (shared pool) | 125 | Regression |
| `CUDA_GRAPHS=1`, capture only, replay disabled (shared pool) | 132 | Replay is not the issue |
| `CUDA_GRAPHS=1`, no capture, no replay (cache object exists but empty) | 290 | Python overhead is negligible |
| `CUDA_GRAPHS=1`, capture + replay (private pool) | 126 | Pool isolation doesn't help |

**Key findings**:

1. The **graph replay path itself** adds minimal overhead (132 → 125, ~5%)
2. The **Python overhead** of the dispatch checks is negligible (290 tok/s with empty cache)
3. The regression comes entirely from **GPU memory consumed by 5,040 captured CUDA graphs**
4. Using a **private `torch.cuda.graph_pool_handle()`** instead of the global pool does NOT help — the issue is total GPU memory pressure, not fragmentation of vLLM's pool

## vLLM Startup Phases (Reference)

```
startup
  → load_weights
  → profile_run()     # cudagraph_capturing_enabled=True, runtime_mode=NONE
                       # Measures peak GPU memory, cuBLAS not yet initialized
  → allocate KV cache # Uses memory measurement from profile_run
  → capture_model()   # cudagraph_capturing_enabled=True, runtime_mode=PIECEWISE
                       # Captures CUDA graphs per batch size
  → inference          # cudagraph_capturing_enabled=False, runtime_mode=PIECEWISE
```

**Key pitfall**: Always check BOTH `cudagraph_capturing_enabled` AND `runtime_mode != NONE` to detect the real capture phase. `cudagraph_capturing_enabled` defaults to `True`, so it's also `True` during profile_run.

## NONE/2048 Batch Pattern (Unrelated)

During 10-request benchmark measurement, vLLM shows alternating `NONE/2048` and `PIECEWISE/8` batches. This is a **pre-existing vLLM scheduling behavior** (occurs with both `CUDA_GRAPHS=0` and `=1`), unrelated to our changes. `2048` is `DEFAULT_MAX_NUM_BATCHED_TOKENS`. The pattern does NOT appear during 5-request warmup.

## Next Steps

The per-mixer approach is fundamentally sound (fixed-layout FULL graphs achieve 900 tok/s vs 290 tok/s PIECEWISE), but the current implementation captures too many graphs.

**Proposed approach**: Instead of pre-capturing all mixers for all batch sizes during `capture_model()`, capture lazily per-placement:

1. When a placement is set, capture graphs only for the active mixer at each layer
2. When placement changes (`set_placement()`), invalidate old graphs (free memory) and re-capture for the new placement
3. This keeps graph count to 48 layers × 1 mixer × N batch sizes (where N can be limited to common decode sizes)

**Open questions**:

- Can we safely capture CUDA graphs during inference (outside `capture_model()`)? At TP=1 this should work. At TP>1, NCCL graph-safe mode is only active during `capture_model()`.
- How many batch sizes do we actually need? If we only capture for the 2-3 most common decode sizes, that's 48-144 graphs total instead of 1,680+.
- Should we zero/free KV cache on placement switch before re-capture? The stale KV cache from a different mixer type would be invalid anyway.
