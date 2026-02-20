# CUDA Graph Capture Crash for Pure GDN/KDA Models

## Problem

After fixing `create_checkpoints.sh` to produce non-stochastic configs (Option A from the throughput analysis), pure GDN and pure KDA models crash during FULL CUDA graph capture:

```text
Capturing CUDA graphs (decode, FULL):   0%|          | 0/51
...
causal_conv1d.py:1160  assert num_cache_lines >= batch
AssertionError
```

The PIECEWISE capture (mixed prefill-decode) succeeds. The FULL capture (decode-only) fails on the very first (largest) batch.

Pure attention and pure sliding_window work fine.

## Root Cause

CUDA graph capture tries batch sizes larger than the number of available mamba state cache slots.

### The assertion

```python
# causal_conv1d.py:1160
num_cache_lines, _, state_len = conv_state.size()
assert num_cache_lines >= batch
```

- `num_cache_lines` = `conv_state.size(0)` = number of allocated mamba state slots = `num_gpu_blocks`
- `batch` = `num_actual_tokens` during decode = number of requests in the captured batch

### Why it's too large

vLLM generates CUDA graph capture sizes up to `min(max_num_seqs * 2, 512)`:

```python
# Default capture sizes:
[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))
```

For FULL decode graphs, capture sizes are filtered to `<= max_num_seqs`. But if `max_num_seqs` exceeds `num_gpu_blocks` (the number of mamba state slots), the largest capture batch won't fit in the conv state tensor.

---

## The `is_hybrid` Problem: Why vLLM Misclassifies the Model

This is the central issue. The `is_hybrid` flag controls how vLLM classifies a model, which in turn determines cache configuration, scheduler behavior, and CUDA graph setup.

### The two models compared

| Property | Old model (`apriel_hybrid.py`) | New model (`modeling_apriel2.py`) |
|----------|-------------------------------|----------------------------------|
| Model class | `AprielHForCausalLM` | `Apriel2ForCausalLM` |
| Interfaces | `HasInnerState, IsHybrid, SupportsPP` | `has_inner_state = True, is_hybrid = False` |
| Layer types | `LlamaDecoderLayer` + `AprielGDNDecoderLayer` | `Apriel2AttentionDecoderLayer`, `Apriel2GDNDecoderLayer`, `Apriel2KDADecoderLayer`, `Apriel2StochasticDecoderLayer` |
| GDN impl | vLLM built-in `Qwen3NextGatedDeltaNet` | Custom `Apriel2GatedDeltaNet` |
| Config verifier | `HybridAttentionMambaModelConfig` (via `IsHybrid`) | None (not registered, `is_hybrid=False`) |

### How `is_hybrid` flows through vLLM

```text
Model class declares is_hybrid
    │
    ▼
vllm/config/model.py:1429 ─── ModelConfig.is_hybrid property
    │
    ├──► vllm/config/vllm.py:1292 ─── triggers HybridAttentionMambaModelConfig
    │       │
    │       ├── calls MambaModelConfig.verify_and_update_config()
    │       │     └── sets mamba_block_size = max_model_len
    │       │
    │       └── unifies page sizes between attention and mamba blocks
    │
    ├──► vllm/config/model.py:1557 ─── attn_type returns "hybrid" (vs "decoder")
    │
    └──► vllm/config/model.py:1192 ─── is_transformer = False
            │
            └── affects layer counting, scheduling, etc.
```

### What happens with `is_hybrid = False` (current new model)

1. **No config verifier runs**: `HybridAttentionMambaModelConfig` is never called. `MambaModelConfig` is only called for models in `MODELS_CONFIG_MAP` (which doesn't include `Apriel2ForCausalLM`). Result: `mamba_block_size` stays `None` (handled by fallback in `get_kv_cache_spec`), but other setup is skipped.

2. **Model classified as standard transformer**:

    ```python
    # vllm/config/model.py:1192-1194
    is_transformer = (
        not self.is_hybrid and not self.has_noops and not self.is_attention_free
    )
    # With is_hybrid=False: is_transformer = True, even for pure GDN/KDA
    ```

3. **`attn_type` returns `"decoder"`** instead of `"hybrid"`:

    ```python
    # vllm/config/model.py:1557-1558
    elif self.is_hybrid:
        return "hybrid"  # <-- never reached
    ```

4. **No mamba-aware scheduling**: vLLM doesn't know it's dealing with mamba-like state. Scheduler decisions about `max_num_seqs`, memory budgets, and CUDA graph sizes are based on standard transformer assumptions.

### What happened with `is_hybrid = True` (old model)

1. **`HybridAttentionMambaModelConfig` ran**, which:
   - Called `MambaModelConfig.verify_and_update_config()` → set `mamba_block_size = max_model_len`
   - Computed unified page sizes between attention and mamba layers
   - The old model had BOTH attention and mamba layers, so attention KV cache inflated page sizes → fewer total blocks → capture sizes naturally stayed in bounds

2. **Model classified as hybrid**: `is_transformer = False`, `attn_type = "hybrid"`. vLLM used mamba-aware scheduling.

### Why `is_hybrid = False` was chosen for the new model

The comment in `modeling_apriel2.py:2805-2808`:

```python
# Don't use is_hybrid=True - it triggers HybridAttentionMambaModelConfig
# which assumes all mamba-like layers have the same shape.
# Apriel2 has heterogeneous blocks, each with its own get_kv_cache_spec().
is_hybrid = False
```

The problem: `HybridAttentionMambaModelConfig` (triggered via the global `is_hybrid` path) computes a single unified page size assuming homogeneous mamba shapes. Apriel2 has heterogeneous blocks — GDN and KDA have different state shapes (different conv dimensions, different recurrent state sizes). Each layer already handles its own unified page size via `get_unified_page_size_for_config()`, which correctly computes the convex hull across ALL mixer types from the config.

Setting `is_hybrid = True` would cause `HybridAttentionMambaModelConfig` to run its own page unification logic that conflicts with our per-layer approach.

### The consequence: a classification gap

The model falls through all of vLLM's model classification:

- Not in `MODELS_CONFIG_MAP` → no architecture-specific config
- `is_hybrid = False` → no `HybridAttentionMambaModelConfig`
- `has_inner_state = True` but this flag isn't checked for config verification
- `is_attention_free = False` → not treated as attention-free

Result: vLLM treats it as a vanilla transformer. For pure attention configs this is correct. For pure GDN/KDA configs, it's wrong — the model has mamba-like state that needs proper cache configuration.

---

## Other Contributing Code Paths

### Config convertor returns 0 for non-attention models

```python
# config_convertor.py:26-53
def _get_first_attention_block(self):
    # For type="gdn", falls through all branches → returns {}
    # → heads=0, head_groups=0, head_size=0
```

This is actually **correct** — vLLM's own `MambaModelArchConfigConvertor` also returns 0 for head params. Pure mamba/GDN/KDA models have no attention heads. But it means vLLM doesn't see any attention memory cost, which can affect `max_num_seqs` estimation.

### The GDN attention backend's `decode_cudagraph_max_bs`

```python
# vllm/v1/attention/backends/gdn_attn.py:93-94
self.decode_cudagraph_max_bs = (
    self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
)
```

This is based on `max_num_seqs`, not on the actual number of allocated state slots. At init time, the cache hasn't been allocated yet, so `num_gpu_blocks` is unknown.

---

## Why It Worked Before (Summary)

| Scenario | Why no crash |
|----------|-------------|
| **Stochastic wrapper** (old benchmarks) | Forced PIECEWISE mode → no FULL graph capture at all. Also, all 4 sub-mixer weights consumed ~45.9 GiB → fewer cache slots → smaller batches. |
| **Old `apriel_hybrid.py`** | `IsHybrid` triggered `HybridAttentionMambaModelConfig`. Had both attention + mamba layers → attention KV inflated page sizes → fewer blocks → capture sizes in bounds. |
| **Pure attention** (new config) | No mamba state → no conv state assertion. Standard KV cache works fine. |
| **Pure GDN/KDA** (new config, **crashes**) | `is_hybrid=False` → no mamba config → vLLM treats as transformer. Many small mamba state slots (397) but capture sizes go up to 512. |

---

## Proposed Fix: Proper `MambaModelConfig` Integration

The `Apriel2ForCausalLM` model class should dynamically invoke the appropriate vLLM config verifier based on the actual layer composition.

### Detection logic

| Layer composition | Config verifier to call |
|---|---|
| All attention / sliding_window | None (standard transformer) |
| All GDN / KDA / mamba | `MambaModelConfig` |
| Mixed (stochastic or pattern with both attention and mamba) | `HybridAttentionMambaModelConfig` |

### Implementation approach

In `Apriel2ForCausalLM.__init__`, inspect `config.layers_block_type` to determine the model's composition:

```python
from vllm.model_executor.models.config import (
    MambaModelConfig,
    HybridAttentionMambaModelConfig,
)

block_types = set(config.layers_block_type)
mamba_types = {"gdn", "kda", "mamba", "stochastic"}
attn_types = {"attention"}

has_mamba = bool(block_types & mamba_types)
has_attention = bool(block_types & attn_types)

if has_mamba and has_attention:
    HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)
elif has_mamba:
    MambaModelConfig.verify_and_update_config(vllm_config)
# else: pure attention, no special config needed
```

### Why this works

- `is_hybrid = False` remains, so the global `is_hybrid` path in `vllm_config` doesn't run `HybridAttentionMambaModelConfig` with its homogeneity assumptions
- We call `verify_and_update_config()` manually, controlling exactly when and how it runs
- `MambaModelConfig` sets `mamba_block_size = max_model_len`, which our `get_kv_cache_spec` already falls back to but should be set properly
- For stochastic configs (which have both attention and mamba sub-mixers), `HybridAttentionMambaModelConfig` runs page unification — but our per-layer `get_unified_page_size_for_config()` already handles this, so we need to verify there's no conflict

### What `MambaModelConfig` does

```python
# vllm/model_executor/models/config.py:319-352
class MambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config):
        # 1. Handle prefix caching for mamba models
        # 2. Set mamba_block_size = max_model_len if not set
        if cache_config.mamba_block_size is None:
            cache_config.mamba_block_size = model_config.max_model_len
```

### What `HybridAttentionMambaModelConfig` does

```python
# vllm/model_executor/models/config.py:355-...
class HybridAttentionMambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config):
        MambaModelConfig.verify_and_update_config(vllm_config)  # sets mamba_block_size
        # Then: computes unified page sizes between attention and mamba layers
        # Adjusts block_size so attention page >= max mamba page
        # Pads mamba pages to match attention page
```

---

## Separate Concern: Runtime Placement Switching in Supernet Mode

When running the supernet (stochastic config) and changing the active placement at runtime, the KV cache properties change per layer. This section documents the design and potential issues.

### How the stochastic mixer pre-allocates cache

Each sub-mixer gets a unique virtual layer index for cache registration:

```python
# modeling_apriel2.py:2409
virtual_layer_idx = layer_idx + (mixer_index + 1) * num_layers
virtual_prefix = f"{layers_base}.{virtual_layer_idx}.stochastic_{name}"
```

Since all sub-mixers are in `nn.ModuleDict`, vLLM discovers them when walking the module tree and allocates cache for ALL of them. The unified page size (`get_unified_page_size_for_config`) computes the "convex hull" across all mixer types, ensuring compatible page sizes.

### What `get_kv_cache_spec` returns for stochastic mixers

```python
# modeling_apriel2.py:2504-2510
def get_kv_cache_spec(self, vllm_config):
    active_mixer = self.mixers[self.active_mixer_name]
    return active_mixer.get_kv_cache_spec(vllm_config)
```

This returns only the **active** mixer's spec. vLLM uses this during init to determine:

- Cache group assignment (which attention backend processes this layer)
- Block table structure
- KV cache spec type (FullAttentionSpec vs MambaSpec)

### The concern

When switching from attention -> GDN at runtime:

1. The cache **memory** is pre-allocated and compatible (unified page sizes)
2. But the **attention metadata builder** for that layer was set up for the initial active mixer type
3. The cache **group** (attention vs mamba backend) is frozen from init

The PIECEWISE CUDA graph mode + `stochastic_mixer_dispatch` custom op handle the dispatch correctly at the Python level (breaking out of compiled code at each layer). But the underlying assumption in vLLM's cache management is that a layer's spec type doesn't change.

### Current mitigations

- Virtual layer indices ensure each sub-mixer has its own cache allocation
- Unified page sizes ensure memory compatibility across all mixer types
- PIECEWISE mode allows Python-level dispatch at each layer boundary
- The `stochastic_mixer_dispatch` op uses `forward_context.no_compile_layers` for runtime routing

### Open questions

1. When vLLM collects cache specs by walking the module tree, does it find ALL sub-mixers (via `nn.ModuleDict`) or only the stochastic mixer's `get_kv_cache_spec` (active mixer)?
2. If cache groups are assigned based on spec type at init, do layers that switch between attention and mamba get the wrong metadata builder?
3. Does the block table management (slot allocation, eviction) work correctly when the cache spec type changes per layer at runtime?

These questions are orthogonal to the throughput benchmark (which uses pure non-stochastic configs) but are important for the runtime placement switching feature.
