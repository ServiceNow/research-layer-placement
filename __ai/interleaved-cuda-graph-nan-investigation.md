# Interleaved Heterogeneous Model — CUDA Graph NaN Investigation

## Problem Statement

Predefined heterogeneous hybrid Apriel2 models with **interleaved** layer layouts produce NaN outputs when served with vLLM's default **FULL** CUDA graph mode (`FULL_AND_PIECEWISE`). The NaN does not appear immediately — it accumulates over ~3 minutes of continuous decode steps, then corrupts all subsequent outputs.

**Affected model**: `10000_surogate_search_lieklihood_6.4cost_placement_interleaved`
- 48 layers: 30 attention + 13 GDN + 5 KDA, interleaved (e.g., attn→GDN→attn→KDA→attn→...)

**Likely also affected**: Grouped layouts (may produce subtly wrong outputs without NaN — not yet verified with quality checks).

**Not affected**: Pure-attention, pure-GDN, pure-KDA models.

---

## Timeline of Investigation

### Phase 1: Enabling Hybrid Classification (Resolved)

**Goal**: Enable `is_hybrid = True` for pattern models so vLLM correctly classifies layers and enables `FULL_AND_PIECEWISE` CUDA graph capture.

**Problem**: `Apriel2ForCausalLM` had `is_hybrid = False` to avoid `HybridAttentionMambaModelConfig.verify_and_update_config()`, which assumes homogeneous mamba shapes. Apriel2 has heterogeneous blocks (GDN and KDA with different state dimensions).

**Approach tried**: Set `is_hybrid = True` unconditionally + implement `get_mamba_state_shape_from_config()` returning the envelope (largest) shape.

**Blocker**: Pure-mamba models (e.g., pure GDN, pure KDA) crashed with `ZeroDivisionError` because `HybridAttentionMambaModelConfig` divides by `num_kv_heads` which is 0 when no attention layers exist.

**Solution implemented**: Reverted to `is_hybrid = False` and instead registered `Apriel2ModelConfig` in vLLM's `MODELS_CONFIG_MAP` (in `config_convertor.py`). This handler inspects `layers_block_type` at startup and routes to the correct config handler:

```python
class Apriel2ModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        layer_types = getattr(hf_config, "layers_block_type", None)
        has_attention = any(t == "attention" for t in layer_types)
        has_mamba = any(t == "mamba" for t in layer_types)

        if has_attention and has_mamba:
            HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)
        elif has_mamba:
            MambaModelConfig.verify_and_update_config(vllm_config)
        # Pure attention: no special config needed.
```

Also added `layers_block_type` property to the HF config class that normalizes all mixer types to canonical `"attention"` or `"mamba"` strings for vLLM compatibility.

**Status**: All homogeneous models (pure attention, pure GDN, pure KDA) and grouped heterogeneous models now start and serve correctly.

### Phase 2: Systematic Model Testing (Resolved)

Tested all exported model configurations:

This is with max_concurrent_requests = 4

| Model | Layout | FULL Graphs | PIECEWISE | Result |
|-------|--------|-------------|-----------|--------|
| `pure_att` | 48× attention | Yes | Yes | Works |
| `pure_gdn` | 48× GDN | Yes | Yes | Works (after Phase 1 fix) |
| `pure_kda` | 48× KDA | Yes | Yes | Works |
| `att36_gdn12` | 36 attn + 12 GDN (grouped) | Yes | Yes | Works |
| `att36_kda12` | 36 attn + 12 KDA (grouped) | Yes | Yes | Works |
| `att_low_6_14_14_14` | 6 attn + 14 GDN + 14 KDA + 14 other (grouped) | Yes | Yes | Works |
| `no_att_0_16_16_16` | Pure mamba mix (grouped) | Yes | Yes | Works |
| `6.4cost_placement_grouped` | 30 attn + 13 GDN + 5 KDA (grouped) | Yes | Yes | Works |
| **`6.4cost_placement`** | 30 attn + 13 GDN + 5 KDA | **NaN** | Yes | 
| **`6.4cost_placement_interleaved`** | 30 attn + 13 GDN + 5 KDA (interleaved) | **NaN** | Yes | **BUG** |

**Key finding**: The interleaved layout is the most visibly broken (NaN/gibberish). Grouped layouts were initially marked as "works" because they didn't produce NaN, but they may also have subtly wrong outputs with NCR>1 — this was not verified with quality comparisons against enforce-eager baseline.

I suspect that it has to do with cache contamination. 

Another data point:
- running with max_concurrent_requests = 1 works fine in any setting
- I can now also run with max_concurrent_requests=12 and get no nans. However, only the first output is sensical, the subseqcuent samples the generations are nonsensical (see generations in /home/toolkit/dev/re
  search-layer-placement/outputs/evals_vllm_serve_test_vllm16/[12 prallel requests]+backend=vllm,+command=evaluate,+evaluator@evaluators.configs.aime=generat  
  ive,+generation=greedy_very_long,+task=aime25___0bf3dff4e8dd/evals/aime/validation.jsonl)

### Phase 3: NaN Root Cause Investigation (Ongoing)

#### Experiment 1: Compiler vs CUDA Graphs

Tested which component causes NaN:

| Mode | torch.compile | CUDA Graphs | Interleaved Model | Result |
|------|---------------|-------------|-------------------|--------|
| `--enforce-eager` | OFF | OFF | Works | Baseline |
| `cudagraph_mode: NONE` | ON | OFF | Works | Compile is fine |
| `cudagraph_mode: PIECEWISE` | ON | Subgraphs | Works, ~323 tok/s | Graph breaks at custom ops |
| `cudagraph_mode: FULL_AND_PIECEWISE` (default) | ON | Full forward | **NaN after ~3 min** | Bug |

**Conclusion**: NaN is strictly a **FULL CUDA graph** issue — torch.compile alone is safe, PIECEWISE subgraphs are safe.

Scaling max_concurrent_requests > 1 either causes NaNs or nonsensical generations.

#### Experiment 2: Address Mismatch Check (FULL_AND_PIECEWISE + DEBUG)

Ran with `VLLM_LOGGING_LEVEL=DEBUG` which enables `CUDAGraphWrapper`'s input tensor address consistency check on every graph replay.

**Findings**:
- NaN reproduced (same ~3 minute delay)
- **NO address mismatch assertions fired** — all input tensor addresses were stable across replays
- First FULL decode batch at 21:35:13, NaN detected at 21:38:31
- Batch sizes were consistently `num_tokens=4, num_reqs=4` (no padding slots)
- Error: `ValueError: Out of range float values are not JSON compliant: nan`

#### Experiment 3: Inductor Kernel Fusion

Tested disabling aggressive kernel fusion (ComboKernels warnings in compile logs suggested inductor was fusing reduction nodes):

| Configuration | Result |
|---------------|--------|
| `combo_kernels: false, benchmark_combo_kernel: false` | Still NaN |
| `allow_buffer_reuse: false` | Still NaN |
| Cleared `~/.cache/vllm/torch_aot_compile/` | Still NaN |

**Conclusion**: Not an inductor fusion issue.

#### Experiment 4: PIECEWISE Mode Throughput

Ran with `--compilation-config '{"cudagraph_mode": "PIECEWISE"}'`:

- **Throughput**: ~323 tok/s steady-state (4 concurrent requests)
- **NaN**: None — ran for extended period without issues
- **Graph capture**: 51 piecewise graphs captured in ~3s

#### Experiment 5: PIECEWISE Output Quality (NCR=1 vs NCR=12)

Tested whether PIECEWISE produces correct outputs and whether concurrency affects output quality.

**Key finding: PIECEWISE is NOT numerically equivalent to FULL mode**, even with NCR=1:

| Run | Config Hash | Accuracy | Prob 0 Output |
|-----|------------|----------|---------------|
| FULL NCR=1 | `948d9c38fd57` | 4/30 | Correct, concise (2,378 chars) |
| PIECEWISE NCR=1 | `948d9c38fd57` | 0/3 | Wrong, rambling (59,326 chars, truncated) |

Outputs diverge as early as character 30–143. FULL NCR=1 produces concise correct answers; PIECEWISE NCR=1 produces verbose incorrect generations. This means **PIECEWISE is not a viable workaround** — it changes model behavior significantly, likely because graph breaks at custom ops force intermediate materialization at bfloat16 precision boundaries, and these differences compound through the recurrent state.

Within PIECEWISE mode, NCR=1 vs NCR=12 produces different (but both coherent) outputs. This is expected floating-point non-determinism from different batch compositions in attention layers, amplified through recurrent state.

#### Experiment 6: Enforce-Eager with NCR>1

**`--enforce-eager` with NCR>1 works correctly.** This confirms:
- The model itself handles concurrent requests correctly
- No state contamination at the model/kernel level
- The bug is strictly in the CUDA graph infrastructure

#### Experiment 7: State Index Collision Check (PIECEWISE + `APRIEL2_DEBUG_STATE_INDICES=1`)

Added logging to print `non_spec_state_indices_tensor` (block IDs) at every GDN and KDA forward call, checking for duplicate block IDs across concurrent requests.

**Results** (PIECEWISE, NCR=12):
- **60,000 state index checks**, **0 duplicates**
- All concurrent requests always assigned unique block IDs
- Mixed prefill+decode batches (e.g., `decodes=2 prefills=10`) also had unique indices
- Batch compositions: mostly `decodes=12 prefills=0` (31,684) and `decodes=1 prefills=0` (23,850)

**Conclusion**: Block allocation is correct — no state slot collision between concurrent requests. The recurrent state indexing (`ssm_state_indices`, `conv_state_indices`) correctly isolates per-request state.

#### Experiment 8: FULL Mode State Index Logging (NCR=12)

Added state index logging to the **GDN metadata builder** (`gdn_attn.py`) instead of the model code, since in FULL CUDA graph mode the Python model forward doesn't execute during replay. The builder's `build()` method runs in Python before every graph replay.

```python
# Added to gdn_attn.py, inside the FULL CUDA graph branch of build()
if _DEBUG_STATE_INDICES:
    indices = non_spec_state_indices_tensor[:num_decodes].tolist()
    ...
    print(f"[GDN-META FULL-CG] decodes={num_decodes} indices={indices}{tag}")
```

**Note on initial attempt**: First tried logging inside `_forward_core` in `modeling_apriel2.py`, but saw zero output during FULL mode decode — because in FULL graph replay, the Python model forward does NOT execute. Only the metadata builder (which runs before `graph.replay()`) can log during FULL mode.

**Note on capture crash**: The initial logging didn't guard against the CUDA graph capture phase — calling `.tolist()` triggers CUDA synchronization, which is illegal during graph capture (`torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing`). Fixed by adding `and not cudagraph_capturing_enabled()` guard to the model-side logging, and placing the builder-side logging BEFORE the `copy_()` into pre-allocated buffers.

**Results (FULL mode, NCR=12)**:
- **During capture**: All-zeros indices (`[0, 0, 0, ...]`) flagged as DUPLICATE — this is **expected**, capture uses synthetic dummy data with all sequences pointing to state slot 0
- **During decode**: **0 duplicates** among real indices. Example: `decodes=16 indices=[126, 20, 226, 40, 50, 60, 70, 80, 90, 100, 116, 284, -1, -1, -1, -1]`
- The `-1` entries are `PAD_SLOT_ID` — CUDA graph batch padding (12 real decodes padded to graph size 16)

**PAD_SLOT_ID (-1) handling verified as safe**:
- `fused_recurrent_gated_delta_rule` kernel (line 114): `if state_idx < 0: return` — early exit for padding slots
- `causal_conv1d_update` kernel (line 81): `if idx_seq == pad_slot_id:` — skips padding slots
- No out-of-bounds memory access from padding

**Conclusion**: State indices are correct in FULL mode. No state slot collision. Padding is handled safely. The corruption is NOT caused by state index routing.

#### Experiment 9: FULL Mode with NCR=2 (Zero Padding Test)

Tested with exactly NCR=2 to eliminate padding as a variable. With 2 concurrent decodes, the batch_size=2 graph is used with zero padding slots.

**Results**:
- Record 0: 2,112 chars, **correct** (correct=True) — likely processed alone before record 1 entered decode
- Records 1–4: 17K–67K chars, **gibberish** — e.g., "We Copengeong to6666's's6 an anansen-u alone..."

**Key insight**: Even with zero padding (batch_size=2, 2 real sequences), the second and subsequent samples produce gibberish. The corruption begins as soon as the batch_size=2 graph is used for decode.

**Conclusion**: **Padding is NOT the cause.** The bug is fundamental to having >1 sequence in a FULL CUDA graph batch for hybrid models.

### Phase 3 Summary

| Mode | NCR=1 | NCR>1 |
|------|-------|-------|
| `--enforce-eager` | Correct (baseline) | **Correct** |
| `cudagraph_mode: NONE` | Correct | (not tested) |
| `cudagraph_mode: PIECEWISE` | Wrong outputs (not equivalent to FULL) | Wrong outputs + float non-determinism |
| `cudagraph_mode: FULL_AND_PIECEWISE` | Correct | **Gibberish/NaN** |

The bug is: **FULL CUDA graphs + NCR>1 + any hybrid model with GDN/KDA layers → corrupted outputs**. Enforce-eager is the only mode that produces correct results at all concurrency levels.

---

## Deep Dive: What Differs Between FULL and PIECEWISE

### PIECEWISE Mode
Each custom op (`vllm::gdn_attention_core`, `vllm::kda_attention`, `vllm::unified_attention`) acts as a **graph break**. Between breaks, Python dispatch runs. This means:
- Metadata tensors are rebuilt in Python at each step
- Triton kernels for GDN/KDA execute as Python-dispatched ops between subgraph segments
- Memory allocations for intermediates happen from the regular CUDA allocator
- For 48 layers: ~48 graph breaks per forward pass (~0.5ms overhead)
- **Graph breaks force intermediate materialization** — if intermediates are kept in float32 within a fused graph but drop to bfloat16 at break boundaries, precision is lost

### FULL Mode
The entire 48-layer forward is captured as **one monolithic CUDA graph** — no Python dispatch between layers. This means:
- ALL intermediate tensor allocations come from the CUDA graph's private memory pool
- The graph pool reuses freed memory aggressively within the single graph
- Kernel launch parameters (grid size, shared memory, etc.) are baked into the graph
- Pre-allocated metadata tensors must be updated in-place before each graph replay

### Why Interleaving Matters

In a **grouped** layout (all attention first, then all GDN, then all KDA), the memory allocation pattern within the CUDA graph pool is:
```
[attn intermediates][attn intermediates]...[GDN intermediates][GDN intermediates]...[KDA intermediates]...
```

In an **interleaved** layout (attn→GDN→attn→KDA→attn→...):
```
[attn intermediates][GDN intermediates][attn intermediates][KDA intermediates][attn intermediates]...
```

The interleaved pattern causes more frequent memory reuse across different layer types within the graph pool. Freed attention intermediates may be reused for GDN/KDA recurrent state computations and vice versa.

---

## Ruled-Out Causes

### State Slot Collision (Ruled Out — Experiment 7)
Block IDs are always unique per concurrent request. The `MambaManager` block allocator correctly isolates state. 60,000 checks across GDN and KDA layers with NCR=12 found zero duplicates.

### Recurrent Kernel Correctness (Ruled Out)
The `fused_recurrent_gated_delta_rule` Triton kernel correctly indexes state per-sequence:
- Reads initial state from `ssm_state[state_idx]` via `ssm_state_indices[i_n]`
- Writes final state to the same slot with `INPLACE_FINAL_STATE=True`
- Each thread block handles one (sequence, head) pair independently
- `causal_conv1d_update` similarly uses `conv_state_indices` for per-request isolation

### PAD_SLOT_ID Handling (Ruled Out — Experiments 8 & 9)
Both recurrent kernels correctly handle `-1` padding:
- `fused_recurrent`: `if state_idx < 0: return` (line 114) AND `T == 0` early exit
- `causal_conv1d_update`: `if idx_seq == pad_slot_id:` skip (line 81)
- `if final_state_idx >= 0:` guard on state write-back (line 158)

Furthermore, **Experiment 9 proved padding is irrelevant**: NCR=2 with zero padding still produces gibberish.

### Batch Padding as Corruption Source (Ruled Out — Experiment 9)
Tested with exactly NCR=2 (batch_size=2 graph, 2 real sequences, 0 padding slots). Still produced gibberish. The corruption is fundamental to having >1 sequence in a FULL CUDA graph, not related to padding.

### Inductor Kernel Fusion (Ruled Out — Experiment 3)
Disabling combo kernels, buffer reuse, and clearing compile cache all still produced NaN.

### Dtype Flow (Ruled Out)
- `fused_gdn_gating` kernel: `g` is explicitly float32, `beta` is bfloat16
- Recurrent kernel: internal computation in float32, stores back via `.to(p_ht.dtype.element_ty)` → bfloat16
- No dtype mismatches found

### Exp Function (Ruled Out)
- Uses `tl.exp` (standard precision), not fast/approximate math

---

## Hypotheses

### H1: CUDA Graph Pool Memory Aliasing (Leading)

**Theory**: In FULL mode, the CUDA graph allocator reuses freed intermediate memory within the monolithic graph. A GDN/KDA layer's intermediate tensor may alias memory that an attention layer freed. When batch_size>1, the aliased memory regions are larger, and different sequences' intermediate computations may interact through this shared memory in ways that don't occur with batch_size=1.

**Evidence for**:
- Only NCR>1 triggers corruption (batch_size=1 graph works fine)
- PIECEWISE works (each subgraph has its own memory scope)
- Enforce-eager works (no graph memory pool at all)
- Corruption is immediate on first batch_size>2 decode step (not gradual) — consistent with an aliasing issue that depends on batch size

**Evidence against**:
- vLLM's CUDAGraphWrapper address check passed (but this checks input addresses, not intermediate allocations)
- CUDA graph aliasing should respect lifetimes — all operations are on a single stream and execute sequentially
- Grouped layouts were initially thought to work, undermining the "interleaving causes different aliasing" theory. However, grouped layouts may also be broken (not yet quality-verified)

### H2: Pre-allocated Metadata Tensor Mutation Race (Weakened)

**Theory**: Pre-allocated metadata tensors updated via `copy_(non_blocking=True)` might not be ready before graph replay.

**Evidence against** (newly weakened):
- State index logging in FULL mode metadata builder (Experiment 8) showed correct values being written
- `copy_()` on the default stream is ordered before `graph.replay()` on the same stream
- NCR=2 with zero padding still fails — not a metadata race since there's nothing to race with

### H3: Triton Kernel Autotuning State Captured in Graph (Unlikely)

**Theory**: Triton autotuning selects kernel configurations baked into the graph. Different batch sizes might select different configs.

**Status**: Unlikely — enforce-eager with same batch sizes works correctly, so the Triton kernels themselves are fine.

### H4: `torch.compile` Inductor Code Generation Difference (New)

**Theory**: In FULL mode, `torch.compile` compiles the model in pieces (between custom op graph breaks). In interleaved layouts, the pattern of compiled blocks between attention and GDN custom ops is different from grouped layouts. The inductor might generate different fused kernels or make different optimization decisions for these patterns, potentially introducing issues that only manifest during CUDA graph replay with batch_size>1.

**Evidence for**:
- `torch.compile` is involved in both PIECEWISE and FULL modes
- PIECEWISE also produces wrong outputs (though coherent, not gibberish) — suggesting compile itself introduces errors
- FULL adds CUDA graph capture on top of compile, compounding any issues

### H5: Graph Input Address Mismatch for Non-obvious Tensors (New)

**Theory**: While the main input tensors (`input_ids`, `positions`, metadata) have stable addresses (verified), there may be less obvious tensors that the graph reads from that DON'T have stable addresses between capture and replay. For example, internal buffers created by `torch.compile`'s inductor, or intermediate results cached by flash attention.

**Testable**: `--compilation-config '{"cudagraph_copy_inputs": true}'` forces explicit copy of all graph inputs before replay.

---

## Diagnostic Instrumentation Applied

### NaN Clamping (temporary)
Added NaN clamping in the decode paths of both GDN and KDA mixers.

### State Index Logging (`APRIEL2_DEBUG_STATE_INDICES=1`)
Logs `non_spec_state_indices_tensor` at every GDN and KDA forward call. Flags duplicate block IDs with `** DUPLICATE **`.

---

## Code Changes Summary

### `config_convertor.py` (new code)
- `Apriel2ModelConfig`: Dynamic config handler that routes to correct vLLM config based on layer composition
- Registered in `MODELS_CONFIG_MAP` for both `Apriel2ForCausalLM` and `Apriel2ForConditionalGeneration`

### `modeling_apriel2.py` (modifications)
- `is_hybrid = False` (kept) — avoids `HybridAttentionMambaModelConfig` crash for pure-mamba
- `layers_block_type` property on HF config — normalizes mixer types to `"attention"` / `"mamba"`
- `get_mamba_state_shape_from_config()` — returns envelope shape for page size alignment
- `get_mamba_state_dtype_from_config()` — returns dtypes matching envelope shape
- NaN clamping diagnostic in GDN and KDA decode paths (temporary)
- State index logging controlled by `APRIEL2_DEBUG_STATE_INDICES` env var

---

## Next Steps

### Immediate
- **Use `--enforce-eager`** for correct results with NCR>1 (only mode that produces correct outputs at all concurrency levels)

### Short-term (Root Cause FULL Mode)
1. **Test with `cudagraph_copy_inputs: true`** (READY — see `__scripts/vllm_serve.sh` commented experiment block)
   - Added `compilation_config` field to `LocalVLLMClientConfig` → passes dict through to `AsyncEngineArgs`
   - Serve with: `'+backend.client.compilation_config={cudagraph_copy_inputs: true}'`
   - **If fix**: The bug is an input tensor address mismatch between capture and replay (H5 confirmed)
   - **If no fix**: Rules out H5, strengthens H1 (graph pool aliasing) or H4 (inductor codegen)
2. ~~Run state index logging in FULL mode~~ **DONE** (Experiment 8) — metadata correct, no duplicates during real inference
3. **Compare graph pool allocations** — use `CUDA_LAUNCH_BLOCKING=1` + `torch.cuda.memory._record_memory_history()` to compare allocation patterns between grouped and interleaved layouts
4. ~~Test FULL mode with NCR=2~~ **DONE** (Experiment 9) — zero padding still produces gibberish

### Medium-term (Fix)
5. **Isolate which mixer causes corruption** — instrument only GDN layers, only KDA layers, then both
6. **Test with `torch.cuda.graphs(pool=...)` shared pool** — the graph pool isolation strategy may matter
7. **File vLLM issue** if root cause is in the graph capture infrastructure

---

## Environment

- **vLLM**: 0.16.0
- **PyTorch**: 2.x with CUDA graphs support
- **GPU**: NVIDIA (single GPU, `gpu-memory-utilization=0.9`)
- **Model**: Apriel2 1.6B (distilled from 15B), 48 layers
- **Checkpoint**: `/mnt/checkpoints/fast_llm_exp/.../10000_surogate_search_lieklihood_6.4cost_placement_interleaved`


# Alex final comment:
- Long story short: currently it looks like we can only reliably eval with full supernet loaded and piecewise + rest api of with predefined layout, FULL graph and NCR = 1
- vllm serve with PIECEVISE rpdiuces some qualitatively different results with both NCR=1 and >1.
- FULL graph with NCR > 1 either Nans or qualitatively suboptimal results