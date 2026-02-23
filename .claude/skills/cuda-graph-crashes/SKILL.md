---
name: cuda-graph-crashes
description: Use when debugging CUDA graph capture crashes with pure GDN/KDA models, understanding the is_hybrid=False classification gap, fixing MambaModelConfig integration, investigating why vLLM misclassifies Apriel2 as a standard transformer, or understanding vLLM CUDA graph capture phases and per-mixer CUDA graph caching.
---

# CUDA Graph Crash Analysis for Apriel2

Root cause analysis and fix for CUDA graph capture crashes, plus documentation of vLLM's CUDA graph capture lifecycle and the per-mixer CUDA graph caching system.

## Quick Reference

### Problem 1: Pure GDN/KDA Crash (FULL capture)

Pure GDN/KDA models crash during FULL CUDA graph capture:
```
causal_conv1d.py:1160  assert num_cache_lines >= batch
```
Capture batch sizes exceed allocated mamba state cache slots.

**Root Cause Chain:**
1. `Apriel2ForCausalLM` sets `is_hybrid = False` (to avoid homogeneous shape assumptions)
2. vLLM classifies it as standard transformer (`is_transformer = True`)
3. No `MambaModelConfig` runs → no mamba-aware scheduling
4. CUDA graph capture sizes based on transformer assumptions → exceed `num_gpu_blocks`

### Problem 2: Per-Mixer CUDA Graph Capture Crashes

When implementing per-mixer CUDA graph caching, two capture crashes were encountered and fixed:

**Crash A: CUBLAS_STATUS_NOT_INITIALIZED during profile_run**
- `cudagraph_capturing_enabled` in `vllm.compilation.monitor` defaults to `True`
- During `profile_run()` (before `capture_model()`), cuBLAS hasn't been lazily initialized
- Lazy init can't happen inside `torch.cuda.graph()` capture
- **Fix**: Also check `forward_context.cudagraph_runtime_mode != CUDAGraphMode.NONE` (profile_run uses NONE, capture_model uses PIECEWISE)

**Crash B: Triton autotuning calls synchronize() during graph capture**
- KDA's `fused_kda_gate` uses `@triton.autotune`, first call at new batch size triggers benchmarking
- Triton's `do_bench()` calls `cuda.synchronize()` which is illegal during stream capture
- **Fix**: Run each mixer eagerly once (warmup) before `torch.cuda.graph()` capture to trigger autotuning outside capture

### The Classification Gap

| Flag | Value | Consequence |
|------|-------|-------------|
| `is_hybrid` | `False` | No `HybridAttentionMambaModelConfig` |
| `is_transformer` | `True` (wrong for pure GDN/KDA) | Transformer scheduling assumptions |
| `has_inner_state` | `True` | Not checked for config verification |
| `attn_type` | `"decoder"` (should be `"hybrid"`) | No mamba-aware scheduling |

### Proposed Fix (Classification Gap)

Dynamically call the right config verifier based on actual layer composition:

| Composition | Config Verifier |
|---|---|
| All attention/sliding_window | None (standard transformer) |
| All GDN/KDA/mamba | `MambaModelConfig` |
| Mixed (stochastic or both) | `HybridAttentionMambaModelConfig` |

## When to Use This Skill

- CUDA graph capture assertion failures with mamba-like layers
- Understanding why `is_hybrid = False` was chosen and its consequences
- Fixing model classification for non-standard architectures in vLLM
- Debugging `mamba_block_size`, `max_num_seqs`, or cache slot count issues
- Understanding vLLM's startup phases (profile_run, capture_model) and their CUDA graph implications
- Debugging CUDA graph capture crashes (CUBLAS, Triton autotuning, stream capture violations)
- Understanding or modifying the per-mixer CUDA graph caching system

## Detailed Documentation

- [Full Crash Analysis](analysis.md) - Complete root cause for the is_hybrid classification gap
- [Per-Mixer CUDA Graph Guide](per-mixer-cuda-graphs.md) - vLLM startup phases, capture lifecycle, and per-mixer graph caching architecture
- [Weight Offloading Guide](weight-offloading.md) - Inactive mixer weight offloading to CPU, pitfalls (shared buffers, torch.compile guards), and 2.58× throughput gain

## Key Code Locations

| Concept | Location |
|---------|----------|
| `is_hybrid = False` | `modeling_apriel2.py:2805-2808` |
| `is_hybrid` flow | `vllm/config/model.py:1429` → `vllm/config/vllm.py:1292` |
| `MambaModelConfig` | `vllm/model_executor/models/config.py:319-352` |
| `HybridAttentionMambaModelConfig` | `vllm/model_executor/models/config.py:355+` |
| GDN attention backend | `vllm/v1/attention/backends/gdn_attn.py:93-94` |
| `MixerGraphCache` | `modeling_apriel2.py:~118-199` |
| `_capture_all_mixers_for_num_tokens()` | `modeling_apriel2.py:~988-1031` |
| `stochastic_mixer_dispatch` (with graph logic) | `modeling_apriel2.py:~1034-1066` |
| `_force_piecewise_cudagraph_for_stochastic_mixers()` | `modeling_apriel2.py:~922-957` |
| `APRIEL2_MIXER_CUDA_GRAPHS` env var | `modeling_apriel2.py:~110` |
| `cudagraph_capturing_enabled` | `vllm/compilation/monitor.py:34` |
| `ForwardContext.cudagraph_runtime_mode` | `vllm/forward_context.py` |
