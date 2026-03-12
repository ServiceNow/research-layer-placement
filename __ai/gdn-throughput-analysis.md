# Analysis: Why GDN Throughput Is Lower With New Modeling File

## Context

Throughput benchmarks show pure GDN at **248 tok/s** vs pure attention at **430 tok/s** using the Fast-LLM modeling file (`modeling_apriel2.py`). In older benchmarks using the apriel-internal modeling file (`apriel_hybrid.py`), GDN was **~11x faster** than attention (~4750 vs ~427 tok/s). We need to identify what changed.

## Throughput Results (Current)

| Variant | tok/s (1000 prompts) |
|---------|---------------------|
| Pure Sliding Window | 481.95 |
| Pure Attention | 430.35 |
| Pure GDN | 248.64 |
| Pure KDA | ~139.90 (10 prompts only) |

## Root Causes

### 1. Stochastic Wrapper Loads ALL Sub-Mixer Weights (~4x memory)

**This is likely the dominant factor.**

The pure placement checkpoints still have `"type": "stochastic"` in config.json — `create_checkpoints.sh` only changes `main_mixer_name`. So every layer is instantiated as `Apriel2StochasticDecoderLayer` containing an `Apriel2StochasticMixer` with ALL 4 sub-mixers (attention, GDN, KDA, sliding_window) in `nn.ModuleDict`.

- **New code** (`modeling_apriel2.py:2397`): Each layer loads weights for attention + GDN + KDA + sliding_window
- **Old code** (`apriel_hybrid.py:242`): Each GDN layer loads only `Qwen3NextGatedDeltaNet`

Impact: The supernet model uses **45.9 GiB** (all 4 mixer weights x 48 layers). A pure-GDN model with only GDN weights would use ~1/4 of that for mixer parameters. With `gpu_memory_utilization=0.9` on 80 GiB, the stochastic model leaves far less memory for KV cache -> smaller batch sizes -> lower throughput.

**Note**: `--load-format dummy` still allocates the same GPU memory for all parameters — it just fills them with random values instead of loading from disk.

**Files**:
- `modeling_apriel2.py:2394-2429` — `Apriel2StochasticMixer.__init__` creates all sub-mixers
- `create_checkpoints.sh:26` — only changes `main_mixer_name`, keeps `type: stochastic`

### 2. PIECEWISE CUDA Graphs (vs FULL)

The stochastic mixer forces PIECEWISE cudagraph mode (`modeling_apriel2.py:922-957`). This is necessary for runtime mixer switching but harmful for pure placements:

- **New code**: PIECEWISE splits the graph at every `stochastic_mixer_dispatch` op (48 splits for 48 layers). Each split adds kernel launch overhead.
- **Old code**: Dedicated `AprielGDNDecoderLayer` can use FULL CUDA graphs — entire forward captured as one graph.

The comment at line 937-939 even notes: *"This is only needed for STOCHASTIC mixers where runtime switching between mixers is required. Standalone GDN/KDA/Mamba models (non-stochastic) can use FULL cudagraph mode without issues."*

**Files**:
- `modeling_apriel2.py:922-957` — `_force_piecewise_cudagraph_for_stochastic_mixers()`
- `modeling_apriel2.py:914-917` — adds `stochastic_mixer_dispatch` to splitting ops

### 3. Custom Op Dispatch Overhead Per Layer

Every forward pass goes through `torch.ops.vllm.stochastic_mixer_dispatch()` (`modeling_apriel2.py:870-884`) which:
- Escapes `torch.compile` (registered as custom op with fake impl)
- Does a Python dict lookup (`forward_context.no_compile_layers[layer_name]`) per layer
- Does another dict lookup (`stochastic_mixer.mixers[active_mixer_name]`)

The old code calls `self.mixer(hidden_states, output=attn_output)` directly — no dispatch, no compile escape.

### 4. Different GDN Implementation

- **New**: Custom `Apriel2GatedDeltaNet` (~600 lines, custom Triton kernels, debug infrastructure)
- **Old**: vLLM's built-in `Qwen3NextGatedDeltaNet`

These may differ in kernel implementations, memory patterns, or have debug overhead. The new implementation has debug flags (`DEBUG_GDN_LAYER`, `DEBUG_GDN_STATE`, etc.) that are `False` but could still affect compilation.

### 5. Virtual Layer Index KV Cache Allocation

Each sub-mixer in the stochastic mixer gets a virtual layer index: `layer_idx + (mixer_index + 1) * num_layers` (`modeling_apriel2.py:2409`). While `get_kv_cache_spec` only delegates to the active mixer (`modeling_apriel2.py:2504-2510`), vLLM may discover ALL sub-mixers when walking the module tree, potentially allocating KV cache for dormant mixers too.

---

## Proposed Fix

To get a fair throughput comparison, the benchmark should use **non-stochastic checkpoints** where each layer is a direct GDN/KDA/attention/sliding_window layer (no stochastic wrapper).

### Option A: Modify `create_checkpoints.sh` to rewrite config as non-stochastic (Recommended)

Change the config from:
```json
{"decoder": {"block": {"mixer": {"type": "stochastic", "main_mixer_name": "gdn", "mixers": {...}}}}}
```
to:
```json
{"decoder": {"block": {"mixer": {"type": "gdn", ...gdn_params...}}}}
```

This way `get_block_config_for_layer()` returns `"gdn"` and vLLM instantiates `Apriel2GDNDecoderLayer` directly — no stochastic wrapper, no extra weights, FULL CUDA graphs.

### Option B: Use the old `apriel_hybrid.py` modeling file with `--trust-remote-code`

Place the old modeling file in the checkpoint directory and let vLLM load it via `auto_map`. This bypasses the stochastic machinery entirely.

### Option C: Add a `--load-format dummy` optimization

When `--load-format dummy` is used, skip instantiating dormant sub-mixers. But this requires changes to the modeling code and doesn't help real serving.

**Recommendation**: Option A is cleanest — it tests the actual layer implementations without stochastic overhead, giving an apples-to-apples comparison.

---

## Verification

After applying the fix:
1. Check that pure_gdn loads as `Apriel2GDNDecoderLayer` (not stochastic) in vLLM logs
2. Check CUDA graph mode is not forced to PIECEWISE
3. Compare model memory usage (should be ~1/4 of current)
4. Re-run all 4 benchmarks and compare throughput
