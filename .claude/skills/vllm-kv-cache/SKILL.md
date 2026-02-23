---
name: vllm-kv-cache
description: Use when investigating vLLM KV cache memory management, paged attention, block allocation, hybrid model cache setup, supernet cache bugs, or understanding how vLLM manages memory for models with multiple mixer types.
---

# vLLM KV Cache Memory Management

Comprehensive guide to how vLLM v0.14.0 manages KV cache memory for transformer models, including hybrid architectures with multiple mixer types (attention, GDN, KDA, Mamba).

## Quick Reference

### Core Concepts

| Concept | Purpose |
|---------|---------|
| **Paged Attention** | OS-style paging for KV cache — fixed-size blocks instead of contiguous allocation |
| **KVCacheSpec** | Per-layer declaration of cache requirements (AttentionSpec vs MambaSpec) |
| **KVCacheGroupSpec** | Groups compatible layers to share one block table |
| **KVCacheTensor** | Physical GPU memory shared by layers at same position across groups |
| **BlockPool** | Global pool ensuring unique block IDs across all groups |
| **Block Tables** | Per-group mapping from (request, position) to block IDs |

### Supernet-Specific Concepts

| Concept | Purpose |
|---------|---------|
| **Virtual Layer Indices** | `real_idx + (mixer_index+1) * num_layers` — unique registration per sub-mixer |
| **Unified Page Size** | Convex hull across all mixer types for compatible page sizes |
| **Shared KVCacheTensor** | All sub-mixers at same real position share physical memory (mutually exclusive) |

## When to Use This Skill

- Debugging KV cache allocation issues or NaN errors after placement switching
- Understanding why memory is over/under-allocated for hybrid models
- Investigating block allocation overlap between mixer groups
- Modifying how supernet layers register in vLLM's cache system
- Understanding the relationship between block_size, page_size, and num_gpu_blocks

## Detailed Documentation

- [Full vLLM KV Cache Guide](guide.md) - Complete walkthrough of paged attention, specs, groups, tensors, block pool, block tables, and supernet bug analysis

## Key File Locations

| Component | File Path |
|-----------|-----------|
| KVCacheSpec types | `/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py` |
| Spec collection | `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/attn_utils.py` |
| KVCacheTensor allocation | `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py` |
| BlockPool | `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py` |
| KVCacheCoordinator | `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py` |
| SingleTypeKVCacheManager | `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py` |
| GPU Model Runner | `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py` |
| Apriel2 Model | `/home/toolkit/dev/Fast-LLM/fast_llm_external_models/apriel2/vllm/modeling_apriel2.py` |
