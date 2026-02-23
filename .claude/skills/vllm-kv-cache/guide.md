# vLLM KV Cache Memory Management

A comprehensive guide to how vLLM v0.14.0 manages KV cache memory for transformer models, including hybrid architectures with multiple mixer types.

---

## Table of Contents

1. [Introduction: Why KV Cache?](#introduction-why-kv-cache)
2. [Paged Attention and Blocks](#paged-attention-and-blocks)
3. [KV Cache Specification System](#kv-cache-specification-system)
4. [Grouping Layers by Cache Type](#grouping-layers-by-cache-type)
5. [Physical Memory: KVCacheTensor](#physical-memory-kvcachetensor)
6. [Block Pool: Managing Block IDs](#block-pool-managing-block-ids)
7. [Block Tables: Mapping Requests to Blocks](#block-tables-mapping-requests-to-blocks)
8. [The Complete Allocation Flow](#the-complete-allocation-flow)
9. [Accessing the Cache at Runtime](#accessing-the-cache-at-runtime)
10. [Hybrid Models: Multiple Mixer Types](#hybrid-models-multiple-mixer-types)
11. [Supernet Bug Analysis](#supernet-bug-analysis)
12. [Key File Locations](#key-file-locations)

---

## Introduction: Why KV Cache?

In autoregressive generation, a transformer generates one token at a time. Without caching, generating token N would require recomputing attention for all previous N-1 tokens - O(N²) complexity.

**KV Cache** stores the Key and Value projections from previous tokens so they don't need to be recomputed:

```
Without KV Cache:
  Token 1: Compute K₁, V₁, attend
  Token 2: Compute K₁, K₂, V₁, V₂, attend  (recompute K₁, V₁!)
  Token 3: Compute K₁, K₂, K₃, V₁, V₂, V₃, attend  (recompute again!)

With KV Cache:
  Token 1: Compute K₁, V₁, store in cache, attend
  Token 2: Compute K₂, V₂, store in cache, attend using cached K₁, V₁
  Token 3: Compute K₃, V₃, store in cache, attend using cached K₁, K₂, V₁, V₂
```

This reduces complexity from O(N²) to O(N) for the KV computation.

---

## Paged Attention and Blocks

### The Problem with Contiguous Memory

Traditional KV cache allocates contiguous memory for each request's maximum possible sequence length. This causes:
- **Memory waste**: A request might only use 100 tokens but reserves space for 4096
- **Fragmentation**: Memory becomes fragmented as requests of different lengths arrive/complete

### vLLM's Solution: Paged Attention

vLLM borrows the concept of **paging** from operating systems. Instead of contiguous allocation, memory is divided into fixed-size **blocks** (like memory pages).

### What is a Block?

A **block** is a fixed-size chunk of GPU memory that can hold KV cache for a certain number of tokens.

```
Block (e.g., block_size = 16 tokens):
┌─────────────────────────────────────────────────────────────────┐
│ K₀ V₀ │ K₁ V₁ │ K₂ V₂ │ ... │ K₁₅ V₁₅ │  (16 token slots)    │
└─────────────────────────────────────────────────────────────────┘
```

**Block properties**:
- **block_size**: Number of tokens per block (e.g., 16)
- **block_id**: Unique identifier (0, 1, 2, ..., N-1)
- **page_size_bytes**: Memory size = 2 × block_size × num_kv_heads × head_size × dtype_size
  - The "2" is for K and V

### Benefits of Blocks

1. **No fragmentation**: Any free block can be used by any request
2. **Fine-grained allocation**: Allocate blocks as needed, not max_seq_len upfront
3. **Memory sharing**: Multiple requests can share blocks (for prefix caching)

---

## KV Cache Specification System

### KVCacheSpec: Describing What a Layer Needs

Each layer tells vLLM what kind of cache it needs via `get_kv_cache_spec()`:

```python
@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int  # tokens per block

    @property
    def page_size_bytes(self) -> int:
        """Memory size of one block"""
        raise NotImplementedError
```

### Spec Types

**1. AttentionSpec** (for standard attention):
```python
@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

    @property
    def page_size_bytes(self) -> int:
        # 2 for K and V
        return 2 * self.block_size * self.num_kv_heads * self.head_size * dtype_size
```

**2. FullAttentionSpec** (with optional sliding window):
```python
@dataclass(frozen=True)
class FullAttentionSpec(AttentionSpec):
    sliding_window: int | None = None
```

**3. MambaSpec** (for state-based layers like Mamba, GDN, KDA):
```python
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]  # State tensor shapes
    dtypes: tuple[torch.dtype, ...]      # State tensor dtypes
```

State-based layers don't store K/V per token - they maintain a compressed state that gets updated.

### How Specs Are Collected

vLLM iterates over all layers that inherit from `AttentionLayerBase`:

```python
# File: vllm/v1/worker/gpu/attn_utils.py
def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    for layer_name, attn_module in attn_layers.items():
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec
```

Result: A dictionary mapping layer names to their specs:
```python
{
    "model.layers.0.self_attn": FullAttentionSpec(...),
    "model.layers.1.self_attn": FullAttentionSpec(...),
    ...
}
```

---

## Grouping Layers by Cache Type

### Why Group Layers?

In a standard transformer, all 48 attention layers have identical specs. Instead of managing 48 separate block tables:
- Group them together
- Use ONE block table for all 48 layers
- All layers in the group use the same blocks for a given request

### KVCacheGroupSpec

```python
@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """
    layer_names: list[str]    # Layers in this group
    kv_cache_spec: KVCacheSpec  # The shared spec
```

### Grouping Examples

**Standard Transformer (all attention)**:
```
Group 0 (FullAttentionSpec):
  layer_names: ["model.layers.0.self_attn", "model.layers.1.self_attn", ..., "model.layers.47.self_attn"]
  kv_cache_spec: FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=bf16)
```

One group, one block table, 48 layers share it.

**Hybrid Model (Attention + Mamba interleaved)**:
```
Group 0 (FullAttentionSpec):
  layer_names: ["model.layers.0.self_attn", "model.layers.2.self_attn", ...]

Group 1 (MambaSpec):
  layer_names: ["model.layers.1.mamba", "model.layers.3.mamba", ...]
```

Two groups, two block tables.

### What Groups Share

All layers in a group share:
1. **One block table** - Same mapping from requests to blocks
2. **One set of attention metadata** - Same slot_mapping, block_table_tensor
3. **One AttentionMetadataBuilder** - Builds the indexing structures

---

## Physical Memory: KVCacheTensor

### What is KVCacheTensor?

`KVCacheTensor` is a **specification** that tells vLLM how to allocate physical GPU memory:

```python
@dataclass
class KVCacheTensor:
    size: int              # Size in bytes
    shared_by: list[str]   # Layer names that share this memory
```

It's NOT the tensor itself - it's a blueprint.

### How KVCacheTensors Are Created

The allocation algorithm creates one KVCacheTensor per "position" across groups:

```python
# File: vllm/v1/core/kv_cache_utils.py (lines 1132-1139)
for i in range(group_size):  # group_size = max layers in any group
    shared_by = []
    for j in range(len(kv_cache_groups)):  # iterate over groups
        if i < len(kv_cache_groups[j].layer_names):
            shared_by.append(kv_cache_groups[j].layer_names[i])
    kv_cache_tensors.append(
        KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
    )
```

**Key insight**: The i-th layer from EACH group shares the SAME KVCacheTensor.

### Allocation Example: Hybrid Model

In a hybrid model like Jamba (attention at even layers, mamba at odd layers):

```
Group 0 (Attention): [layer.0.attn, layer.2.attn, ..., layer.46.attn]  (24 layers)
Group 1 (Mamba):     [layer.1.mamba, layer.3.mamba, ..., layer.47.mamba]  (24 layers)

KVCacheTensor 0: shared_by = [layer.0.attn, layer.1.mamba]   ← i=0 from each group (different real layers!)
KVCacheTensor 1: shared_by = [layer.2.attn, layer.3.mamba]   ← i=1 from each group
KVCacheTensor 2: shared_by = [layer.4.attn, layer.5.mamba]
...
KVCacheTensor 23: shared_by = [layer.46.attn, layer.47.mamba]
```

Note: These are **different** real layer positions sharing memory. Layer 0 (attention) and layer 1 (mamba) never execute at the same time for the same token, so sharing is safe.

### Physical Tensor Creation

```python
# File: vllm/v1/worker/gpu_model_runner.py (lines 5339-5344)
def _allocate_kv_cache_tensors(self, kv_cache_config):
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        # Create ONE physical tensor
        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
        # Assign the SAME tensor to all layers that share it
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor  # Same object!
    return kv_cache_raw_tensors
```

**Result**: Multiple layer names point to the SAME GPU memory.

### Memory Layout Within a Tensor

Each KVCacheTensor has space for `num_blocks` blocks:

```
KVCacheTensor (size = page_size_bytes × num_blocks):
┌──────────────┬──────────────┬──────────────┬─────┬──────────────┐
│   Block 0    │   Block 1    │   Block 2    │ ... │  Block N-1   │
│ (page_size)  │ (page_size)  │ (page_size)  │     │ (page_size)  │
└──────────────┴──────────────┴──────────────┴─────┴──────────────┘
     ↑                                                    ↑
  offset 0                                    offset (N-1) × page_size
```

A block ID determines the offset: `offset = block_id × page_size_bytes`

---

## Block Pool: Managing Block IDs

### What is the BlockPool?

The `BlockPool` manages the allocation and deallocation of block IDs. Think of it as a "free list" for blocks.

```python
# File: vllm/v1/core/block_pool.py
class BlockPool:
    def __init__(self, num_gpu_blocks, ...):
        # Create all block objects
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        # Queue of free blocks (initially all are free)
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
```

### Block Object

```python
class KVCacheBlock:
    block_id: int       # Unique identifier (0, 1, 2, ...)
    ref_cnt: int        # Reference count (for sharing)
    block_hash: str     # For prefix caching
```

### Allocating Blocks

```python
def get_new_blocks(self, num_blocks) -> list[KVCacheBlock]:
    # Pop blocks from the free queue
    ret = self.free_block_queue.popleft_n(num_blocks)
    for block in ret:
        block.ref_cnt += 1  # Mark as in-use
    return ret
```

**Key property**: Block IDs are globally unique. If block 5 is allocated, no one else can get block 5 until it's freed.

### Freeing Blocks

```python
def free_blocks(self, blocks):
    for block in blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self.free_block_queue.append(block)  # Return to free list
```

### Shared BlockPool

**Critical**: There is ONE BlockPool shared by ALL groups:

```python
# File: vllm/v1/core/kv_cache_coordinator.py (lines 49-68)
class KVCacheCoordinator:
    def __init__(self, ...):
        # ONE pool for everyone
        self.block_pool = BlockPool(num_blocks, ...)

        # Each group gets a manager, but they share the pool
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                block_pool=self.block_pool,  # SAME pool!
                kv_cache_group_id=i,
                ...
            )
            for i, group in enumerate(kv_cache_groups)
        )
```

---

## Block Tables: Mapping Requests to Blocks

### What is a Block Table?

A **block table** maps (request, token position) to block IDs:

```
Block Table for Group 0:
┌────────────┬─────────┬─────────┬─────────┬─────────┐
│ Request    │ Block 0 │ Block 1 │ Block 2 │ Block 3 │
├────────────┼─────────┼─────────┼─────────┼─────────┤
│ Request A  │    5    │   12    │   23    │   -     │  ← Uses blocks 5, 12, 23
│ Request B  │    3    │    8    │   -     │   -     │  ← Uses blocks 3, 8
│ Request C  │   17    │   42    │   51    │   67    │  ← Uses blocks 17, 42, 51, 67
└────────────┴─────────┴─────────┴─────────┴─────────┘
```

If block_size = 16:
- Request A with 40 tokens needs ceil(40/16) = 3 blocks
- Request C with 64 tokens needs ceil(64/16) = 4 blocks

### One Block Table Per Group

Each `KVCacheGroupSpec` has its own block table:

```
Group 0 (Attention):  block_table_0[request_id] = [5, 12, 23]
Group 1 (Mamba):      block_table_1[request_id] = [100, 105, 110]
```

Different groups allocate independently → different block IDs.

### SingleTypeKVCacheManager

Each group has a manager that maintains its block table:

```python
# File: vllm/v1/core/single_type_kv_cache_manager.py
class SingleTypeKVCacheManager:
    def __init__(self, block_pool, ...):
        self.block_pool = block_pool
        self.req_to_blocks: dict[str, list[KVCacheBlock]] = {}  # The block table!

    def allocate_new_blocks(self, request_id, num_tokens):
        req_blocks = self.req_to_blocks[request_id]
        num_required = ceil(num_tokens / self.block_size)
        num_new = num_required - len(req_blocks)
        if num_new > 0:
            new_blocks = self.block_pool.get_new_blocks(num_new)
            req_blocks.extend(new_blocks)
        return new_blocks
```

---

## The Complete Allocation Flow

### Step 1: Model Initialization

1. Model loads, each layer registers in `static_forward_context`
2. `get_kv_cache_spec()` collects specs from all `AttentionLayerBase` layers
3. Specs are grouped into `KVCacheGroupSpec` objects
4. `KVCacheTensor` specs are created (determining which layers share memory)
5. Physical tensors are allocated on GPU
6. `bind_kv_cache()` connects tensors to layers

### Step 2: Request Arrives

```python
# File: vllm/v1/core/kv_cache_coordinator.py
def allocate_new_blocks(self, request_id, num_tokens, ...):
    return tuple(
        manager.allocate_new_blocks(request_id, num_tokens)
        for manager in self.single_type_managers  # ALL groups!
    )
```

For a request with 50 tokens (block_size=16, needs 4 blocks):
```
Group 0 calls block_pool.get_new_blocks(4) → gets [0, 1, 2, 3]
Group 1 calls block_pool.get_new_blocks(4) → gets [4, 5, 6, 7]
Group 2 calls block_pool.get_new_blocks(4) → gets [8, 9, 10, 11]
```

### Step 3: Forward Pass

Each layer looks up its blocks and accesses the appropriate memory region.

**For Attention** (using slot_mapping):
```python
slot = block_id * block_size + token_offset_in_block
# E.g., token 20 with block_size=16: block_id=1, offset=4, slot=20
```

**For State-based (Mamba/GDN/KDA)** (using state_indices):
```python
state_index = block_id  # Direct block access
state = kv_cache[state_index]
```

### Step 4: Request Completes

```python
def free(self, request_id):
    for manager in self.single_type_managers:
        manager.free(request_id)  # Returns blocks to pool
```

---

## Accessing the Cache at Runtime

### Attention Metadata

Each group builds metadata for the forward pass:

```python
# File: vllm/v1/worker/gpu/attn_utils.py
def build_attn_metadata(...):
    for i, group in enumerate(kv_cache_groups):
        block_table = block_tables[i]      # This group's block table
        slot_mapping = slot_mappings[i]    # This group's slot mapping

        metadata = builder.build(
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            ...
        )

        # Assign same metadata to all layers in group
        for layer_name in group.layer_names:
            attn_metadata[layer_name] = metadata
```

### Layer Forward Pass

```python
# Attention layer
def forward(self, hidden_states, positions):
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata[self.layer_name]
    kv_cache = self.kv_cache[0]  # Bound during initialization

    # Use slot_mapping to index into kv_cache
    k_cache, v_cache = kv_cache[0], kv_cache[1]
    # ... attention computation using attn_metadata.slot_mapping
```

```python
# State-based layer (Mamba/GDN/KDA)
def forward(self, hidden_states):
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata[self.layer_name]
    state_cache = self.kv_cache[0]

    # Use state_indices to access state
    state = state_cache[attn_metadata.state_indices]
    # ... update state
    state_cache[attn_metadata.state_indices] = new_state
```

---

## Hybrid Models: Multiple Mixer Types

### Standard Hybrid (e.g., Jamba: Attention + Mamba interleaved)

```
Layer 0: Attention
Layer 1: Mamba
Layer 2: Attention
Layer 3: Mamba
...
```

**Groups**:
```
Group 0 (Attention): [layer.0.attn, layer.2.attn, layer.4.attn, ...]
Group 1 (Mamba):     [layer.1.mamba, layer.3.mamba, layer.5.mamba, ...]
```

**KVCacheTensor sharing** (i-th from each group):
```
KVCacheTensor 0: shared_by = [layer.0.attn, layer.1.mamba]
KVCacheTensor 1: shared_by = [layer.2.attn, layer.3.mamba]
...
```

Note: These are DIFFERENT model layers sharing memory (layer 0 and layer 1).

### Supernet (e.g., Apriel2: Multiple mixers per layer position)

```
Layer 0: Can be Attention OR GDN OR KDA (switchable at runtime)
Layer 1: Can be Attention OR GDN OR KDA
...
```

**Virtual Layer Indices** (to ensure unique registration):
```python
virtual_layer_idx = real_layer_idx + (mixer_index + 1) * num_layers
```

For 48-layer model:
- Attention at layer 0 → virtual 48
- GDN at layer 0 → virtual 96
- KDA at layer 0 → virtual 144

**Groups**:
```
Group 0 (Attention): [virtual.48.attn, virtual.49.attn, ..., virtual.95.attn]
Group 1 (GDN):       [virtual.96.gdn, virtual.97.gdn, ..., virtual.143.gdn]
Group 2 (KDA):       [virtual.144.kda, virtual.145.kda, ..., virtual.191.kda]
```

**KVCacheTensor sharing**:
```
KVCacheTensor 0: shared_by = [virtual.48.attn, virtual.96.gdn, virtual.144.kda]
                 → All correspond to REAL layer 0!

KVCacheTensor 1: shared_by = [virtual.49.attn, virtual.97.gdn, virtual.145.kda]
                 → All correspond to REAL layer 1!
```

**Key difference from standard hybrid**: In supernet, the layers sharing a KVCacheTensor are at the SAME real position but are mutually exclusive (only one active at a time).

---

## Supernet Bug Analysis

### The Bug

When placement switches in Apriel2 supernet (e.g., layer 0 changes from KDA to Attention), NaN errors occur from reading corrupted cache data.

### The Fix Applied

Clear all KV cache tensors when placement switches:

```python
def _clear_kv_cache(self):
    for cache_item in model_runner.kv_caches:
        if isinstance(cache_item, list):
            for tensor in cache_item:
                tensor.zero_()
        else:
            cache_item.zero_()
```

### Why It's Confusing

Based on the vLLM architecture described above, supernet **should work**:

1. **Different groups get different block IDs**
   - Group 0 (Attention) allocates → gets blocks [0, 1, 2, ...]
   - Group 2 (KDA) allocates → gets blocks [100, 101, 102, ...]
   - The shared BlockPool ensures no overlap

2. **Each group has its own block table**
   - attn.48 uses block_table[0] with blocks [0, 1, 2]
   - kda.144 uses block_table[2] with blocks [100, 101, 102]

3. **Shared tensor, different offsets**
   - Block 0 → offset 0 in tensor
   - Block 100 → offset 100 × page_size in tensor
   - No overlap!

### Expected Safe Scenario

```
KVCacheTensor 0 (shared by attn.48, gdn.96, kda.144):
┌────────┬────────┬────────┬─────┬─────────┬─────────┬─────────┬─────┐
│ Blk 0  │ Blk 1  │ Blk 2  │ ... │ Blk 100 │ Blk 101 │ Blk 102 │ ... │
└────────┴────────┴────────┴─────┴─────────┴─────────┴─────────┴─────┘
    ↑                                  ↑
 attn.48                            kda.144
 uses these                         uses these
```

Different blocks → Different memory regions → Should be safe.

### Possible Root Causes

#### Theory 1: Scheduler Optimization

Maybe vLLM recognizes that attn.48 and kda.144 are "alternatives" (same real layer) and shares block allocations for efficiency?

**Test**: Log block IDs for each group and check for overlap.

#### Theory 2: Dormant Mixer Block Allocation

If the scheduler only allocates for the ACTIVE mixer:
- Initially KDA is active → only KDA gets blocks
- Placement switches → Attention becomes active
- Attention has no blocks, or tries to reuse KDA's blocks

**Test**: Check block allocation status before/after placement switch.

#### Theory 3: Block Table Sharing

Maybe groups at the same real position share block tables?

**Test**: Compare block_table pointers across groups.

#### Theory 4: Index Interpretation Mismatch

KDA uses `state_indices` (block ID directly), Attention uses `slot_mapping` (block_id × block_size + offset).

If the same block ID is used by both:
- KDA interprets block 5 as: read/write entire block as state
- Attention interprets block 5 as: slots 80-95 (if block_size=16)

Same physical memory, different access patterns → corruption.

**Test**: Compare state_indices and slot_mapping values.

### Debugging Steps

```python
# Add to placement switch code:
def debug_block_allocation():
    for group_idx, manager in enumerate(coordinator.single_type_managers):
        for req_id, blocks in manager.req_to_blocks.items():
            block_ids = [b.block_id for b in blocks]
            print(f"Group {group_idx}, Request {req_id}: blocks = {block_ids}")

    # Check for overlap
    all_block_sets = [
        set(b.block_id for blocks in mgr.req_to_blocks.values() for b in blocks)
        for mgr in coordinator.single_type_managers
    ]
    for i, set_i in enumerate(all_block_sets):
        for j, set_j in enumerate(all_block_sets):
            if i < j:
                overlap = set_i & set_j
                if overlap:
                    print(f"BUG: Groups {i} and {j} share blocks: {overlap}")
```

---

## Key File Locations

| Component | File Path |
|-----------|-----------|
| KVCacheSpec types | `/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py` |
| Spec collection | `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/attn_utils.py` |
| KVCacheTensor allocation | `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py` |
| BlockPool | `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py` |
| KVCacheCoordinator | `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py` |
| SingleTypeKVCacheManager | `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py` |
| KVCacheManager | `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py` |
| GPU Model Runner | `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py` |
| Bind KV Cache | `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py` |
| Apriel2 Model | `/home/toolkit/dev/Fast-LLM/fast_llm_external_models/apriel2/vllm/modeling_apriel2.py` |

---

## Summary

vLLM's KV cache management uses:

1. **Paged memory** - Divided into fixed-size blocks for flexibility
2. **Specs** - Each layer declares its cache requirements
3. **Groups** - Compatible layers share block tables for efficiency
4. **KVCacheTensor** - Physical memory shared by layers at same position across groups
5. **BlockPool** - Global pool ensuring unique block IDs
6. **Block tables** - Per-group mapping from requests to blocks

For supernets, the architecture should prevent conflicts because different groups get different block IDs. The fact that the bug exists suggests either a scheduler optimization, a block allocation issue, or an index interpretation mismatch that needs further investigation.
