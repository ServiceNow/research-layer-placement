# Inactive Mixer Weight Offloading

## Overview

Env var: `APRIEL2_OFFLOAD_INACTIVE_MIXERS=1` (default when `APRIEL2_FULL_CUDA_GRAPHS=1`).

The Apriel2 supernet loads all 4 mixer weights per layer (~45.92 GiB), leaving only
~24.48 GiB for KV cache. A fixed-layout model uses ~26.91 GiB for weights and gets
~43.50 GiB of KV cache (1.78× more), yielding 583 tok/s vs ~200 tok/s.

Weight offloading moves inactive mixer parameters to CPU after startup, freeing
~19 GiB for KV cache. Result: **516 tok/s** (88% of fixed-layout baseline).

---

## How It Works

### Startup Sequence

```
load_weights()         # all weights on GPU (45.92 GiB)
  → profile_run()      # torch.compile traces forward, all weights on GPU
  → offload_inactive() # move 3/4 mixer params to CPU, free ~19 GiB
  → allocate KV cache  # vLLM uses adjusted budget (~43.5 GiB)
  → warmup & capture   # compiled model runs with only active mixer on GPU
  → ready for inference
```

The offload is triggered inside a monkey-patched `Worker.determine_available_memory()`,
which runs after `profile_run()` and adjusts `available_kv_cache_memory_bytes` upward
by the freed amount.

### Layout Switch

When `set_layer_placements()` is called with a new layout:

1. KV cache is cleared (prevent stale data from incompatible mixer formats)
2. Weights are swapped layer by layer:
   - Old active → CPU, removed from `nn.ModuleDict`, stored in `_offloaded_mixers`
   - New active → GPU, restored from `_offloaded_mixers` to `nn.ModuleDict`
3. CUDA graphs are invalidated and re-captured (~5-15 s)

Layer-by-layer swapping avoids transient OOM (old freed before new loaded).

---

## Pitfalls Encountered

### Pitfall 1: Cannot Offload in `load_weights()`

`torch.compile` captures ALL model parameters as compiled graph inputs.
If inactive mixer params are moved to CPU during `load_weights()`, the
subsequent `profile_run()` crashes — the compiled forward tries to pass
CPU tensors to GPU operations.

**Symptom**: `EngineCore failed to start` during `profile_run()` → compiled
forward → RuntimeError about CPU tensors.

**Fix**: Offload AFTER `profile_run()` completes (inside the patched
`determine_available_memory`).

### Pitfall 2: Shared Buffers (RotaryEmbedding)

`RotaryEmbedding` instances are shared across mixer instances via vLLM's
`get_rope()` LRU cache (`_ROPE_DICT`). The `cos_sin_cache` buffer lives
on the shared instance.

If `_move_module_device()` moves ALL parameters AND buffers, the shared
`cos_sin_cache` gets moved to CPU — breaking the active mixer's rotary
embedding.

**Symptom**: `RuntimeError: Expected all tensors to be on the same device,
but got index is on cuda:0, different from other tensors on cpu` in
`rotary_embedding/base.py:forward_static → cos_sin_cache.index_select()`.

**Fix**: Only move `module.parameters()`, NOT `module.buffers()`. Weight
parameters are unique per mixer instance; buffers may be shared.

```python
def _move_module_device(module, device):
    for param in module.parameters():
        param.data = param.data.to(device, non_blocking=True)
    # Do NOT move buffers — they may be shared (e.g. cos_sin_cache)
```

### Pitfall 3: torch.compile Guard Invalidation

After offloading, if modules stay in `nn.ModuleDict` with CPU-device
parameters, `torch.compile`'s dynamo guards detect the device change on
the next forward call. This triggers re-tracing which can fail.

**Fix**: Remove offloaded mixers from `nn.ModuleDict` and store them in a
plain Python dict (`self._offloaded_mixers`). This hides them from dynamo's
module tree walk entirely.

```python
def offload_inactive_mixers(self):
    self._offloaded_mixers = {}
    for name in inactive_names:
        mixer = self.mixers[name]
        _move_module_device(mixer, torch.device("cpu"))
        self._offloaded_mixers[name] = mixer
        del self.mixers[name]  # remove from ModuleDict
```

### Pitfall 4: set_active_mixer Validation

After offloading, `self.mixers` only contains the active mixer. The
`set_active_mixer()` validation must check `self._mixer_names` (cached
list from init) instead of `self.mixers` (ModuleDict).

---

## Memory Budget

| State | Weights | KV cache | Free |
|-------|---------|----------|------|
| After load_weights | 45.92 GiB | 0 | ~32 GiB |
| After profile_run + offload | 26.91 GiB | 0 | ~51 GiB |
| After KV cache allocation | 26.91 GiB | 43.50 GiB | ~8 GiB |

The math: `available_kv_cache = requested_memory - non_kv_cache_memory + freed_by_offload`

---

## What Changes

- `_move_module_device()` helper (parameters only, not buffers)
- `Apriel2StochasticMixer.offload_inactive_mixers()` — moves to CPU + removes from ModuleDict
- `Apriel2ForCausalLM.offload_inactive_mixers()` — iterates all stochastic layers
- `_swap_mixer_weights()` — layer-by-layer swap on layout change
- Monkey-patched `Worker.determine_available_memory` — offloads after profile_run
- `set_active_mixer()` validates against `_mixer_names` instead of `self.mixers`

## What Does NOT Change

- KV cache allocation logic (vLLM's sharing mechanism already efficient)
- Virtual layer indices and `static_forward_context` registration
- `stochastic_mixer_dispatch` custom op behavior
- Sub-mixer classes (Attention, GDN, KDA)
- Weight loading (`load_weights()`)

---

## Performance Results

H100 80 GB, all-attention layout, 10 concurrent requests, 16k output:

| Mode | Offload | Weights | KV cache | tok/s |
|------|---------|---------|----------|-------|
| 0 (fixed-layout) | - | 26.9 Gi | 43.5 Gi | 583 |
| 1 (supernet, default) | on | 26.9 Gi | 43.5 Gi | 516 |
| 1b (supernet) | off | 45.9 Gi | 24.5 Gi | ~200 |

Weight offloading provides a **2.58× speedup** over the non-offload supernet mode.
