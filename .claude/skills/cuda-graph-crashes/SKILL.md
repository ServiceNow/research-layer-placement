---
name: cuda-graph-crashes
description: Use when debugging CUDA graph capture crashes with pure GDN/KDA models, understanding the is_hybrid=False classification gap, fixing MambaModelConfig integration, or investigating why vLLM misclassifies Apriel2 as a standard transformer.
---

# CUDA Graph Crash Analysis for Apriel2

Root cause analysis and fix for CUDA graph capture crashes when running pure GDN/KDA placements through vLLM.

## Quick Reference

### The Problem

Pure GDN/KDA models crash during FULL CUDA graph capture:
```
causal_conv1d.py:1160  assert num_cache_lines >= batch
```
Capture batch sizes exceed allocated mamba state cache slots.

### Root Cause Chain

1. `Apriel2ForCausalLM` sets `is_hybrid = False` (to avoid homogeneous shape assumptions)
2. vLLM classifies it as standard transformer (`is_transformer = True`)
3. No `MambaModelConfig` runs → no mamba-aware scheduling
4. CUDA graph capture sizes based on transformer assumptions → exceed `num_gpu_blocks`

### The Classification Gap

| Flag | Value | Consequence |
|------|-------|-------------|
| `is_hybrid` | `False` | No `HybridAttentionMambaModelConfig` |
| `is_transformer` | `True` (wrong for pure GDN/KDA) | Transformer scheduling assumptions |
| `has_inner_state` | `True` | Not checked for config verification |
| `attn_type` | `"decoder"` (should be `"hybrid"`) | No mamba-aware scheduling |

### Proposed Fix

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

## Detailed Documentation

- [Full Crash Analysis](analysis.md) - Complete root cause, code paths, and implementation details

## Key Code Locations

| Concept | Location |
|---------|----------|
| `is_hybrid = False` | `modeling_apriel2.py:2805-2808` |
| `is_hybrid` flow | `vllm/config/model.py:1429` → `vllm/config/vllm.py:1292` |
| `MambaModelConfig` | `vllm/model_executor/models/config.py:319-352` |
| `HybridAttentionMambaModelConfig` | `vllm/model_executor/models/config.py:355+` |
| GDN attention backend | `vllm/v1/attention/backends/gdn_attn.py:93-94` |
