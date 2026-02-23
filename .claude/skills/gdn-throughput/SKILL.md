---
name: gdn-throughput
description: Use when analyzing GDN/KDA throughput regressions, investigating stochastic wrapper overhead, optimizing benchmarks for pure mixer configurations, or understanding why pure GDN is slower than expected compared to attention.
---

# GDN Throughput Analysis

Analysis of why pure GDN throughput is lower with the new Apriel2 modeling file compared to the old apriel-internal implementation.

## Quick Reference

### Current Throughput

| Variant | tok/s |
|---------|-------|
| Pure Sliding Window | 481.95 |
| Pure Attention | 430.35 |
| Pure GDN | 248.64 |
| Pure KDA | ~139.90 |

GDN was ~11x faster than attention in old benchmarks (~4750 vs ~427 tok/s). Now it's ~1.7x slower.

### Root Causes (Ranked by Impact)

| # | Cause | Impact | Fix |
|---|-------|--------|-----|
| 1 | **Stochastic wrapper loads ALL 4 sub-mixer weights** | ~4x memory → fewer cache slots → smaller batches | Use non-stochastic checkpoints |
| 2 | **PIECEWISE CUDA graphs** (48 graph breaks) | Kernel launch overhead per layer | FULL mode for pure placements |
| 3 | **Custom op dispatch overhead** | Python dict lookups escaping torch.compile | Direct layer calls |
| 4 | **Different GDN implementation** | Custom vs vLLM built-in Qwen3NextGatedDeltaNet | Profile and compare |
| 5 | **Virtual layer KV cache allocation** | Possible dormant mixer cache allocation | Verify with logging |

### Recommended Fix

**Option A** (recommended): Modify `create_checkpoints.sh` to produce non-stochastic configs where `layers_block_type` uses direct types (`"gdn"`, `"kda"`) instead of `"stochastic"` with `main_mixer_name`.

## When to Use This Skill

- Investigating throughput regressions for GDN/KDA models
- Creating fair benchmark comparisons between mixer types
- Understanding stochastic wrapper overhead
- Optimizing checkpoint configs for pure placements
- Deciding between stochastic vs non-stochastic model configurations

## Detailed Documentation

- [Full Throughput Analysis](analysis.md) - Complete root cause analysis with file references and verification steps

## Key File Locations

| File | Relevance |
|------|-----------|
| `modeling_apriel2.py:2394-2429` | `Apriel2StochasticMixer.__init__` — creates all sub-mixers |
| `modeling_apriel2.py:922-957` | `_force_piecewise_cudagraph_for_stochastic_mixers()` |
| `modeling_apriel2.py:870-884` | `stochastic_mixer_dispatch` custom op |
| `modeling_apriel2.py:2504-2510` | `get_kv_cache_spec` delegates to active mixer only |
| `create_checkpoints.sh` | Only changes `main_mixer_name`, keeps `type: stochastic` |
