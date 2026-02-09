# vLite — Technical README (Implemented Components)

This repository contains a minimal, production-oriented implementation of
paged KV caching and attention for single-GPU inference. The README below
documents only the components that are implemented in this codebase.

## Purpose

vLite provides a compact implementation of paged KV cache management and a
pure-PyTorch PagedAttention kernel. The goal is to enable efficient
per-request KV storage (low internal/external fragmentation) and correct
attention computation when KV blocks are non-contiguous in memory.

## Implemented Components

1. `vlite.core.sequence`
   - `Sequence` and `SequenceStatus`
   - Tracks an individual request: prompt tokens, generated output tokens,
     and lifecycle status (WAITING, RUNNING, FINISHED_*).
   - Simple helpers: `append_token`, `is_finished`, and length properties.

2. `vlite.core.block_manager`
   - `BlockAllocator`
     - Low-level physical block pool with O(1) allocate/free.
     - Maintains per-block reference counts to allow safe sharing.
   - `BlockSpaceManager`
     - Per-sequence block tables mapping logical token indices to physical
       block IDs.
     - `allocate(seq_id, num_prompt_tokens)` — allocate blocks for a prompt.
     - `append_slot(seq_id)` — grow a sequence by one token and return the
       flat cache slot index.
     - `fork(parent_id, child_id)` — increase refcounts and share blocks
       (copy-on-write pattern).
     - `free(seq_id)` — return all blocks to the pool.

3. `vlite.core.kv_cache`
   - `CacheEngine`
     - Pre-allocates per-layer K and V tensors on the target device with
       shape `[num_blocks, block_size, num_kv_heads, head_dim]`.
     - `write_kv(...)` writes key/value vectors into slots determined by
       `BlockSpaceManager`'s slot mapping.
     - `copy_blocks(...)` performs block-level copy (used for CoW).

4. `vlite.kernels.paged_attention`
   - `PagedAttention` (pure PyTorch)
     - `forward_prefix(...)` — prefill phase:
       - Writes K/V for contiguous prompt tokens into the paged cache.
       - Computes causal self-attention for the prompt (returns attention
         outputs for the whole prompt).
     - `forward_decode(...)` — decode phase:
       - Accepts a single query token per sequence.
       - Reads scattered K/V blocks via a block table per sequence.
       - Implements numerically-stable online softmax across blocks so the
         result is identical to computing attention over a contiguous KV.
     - Supports GQA (expanding KV heads to query heads when necessary).

5. Tests
   - `tests/test_paged_attention.py` includes unit tests covering:
     - `BlockSpaceManager` lifecycle (alloc/append/free/fork/slot-mapping).
     - `CacheEngine` write/read semantics.
     - `PagedAttention` correctness (single-block, multi-block,
       batched mixed-lengths).
     - End-to-end prefill → decode cycles.

## Example (step-by-step)

This example shows how the implemented pieces work together for a single
request (prompt of 4 tokens, generate 3 tokens). The code below mirrors the
examples used in tests.

```python
from vlite.core.block_manager import BlockSpaceManager
from vlite.core.kv_cache import CacheEngine
from vlite.kernels.paged_attention import PagedAttention
from vlite.core.sequence import Sequence, SequenceStatus
import torch

# 1) Setup (tiny example parameters)
BS = 4          # block size (tokens per block)
NUM_BLOCKS = 8
NUM_LAYERS = 1
KV_HEADS = 4
HEAD_DIM = 8

engine = CacheEngine(NUM_LAYERS, KV_HEADS, HEAD_DIM, NUM_BLOCKS, BS, device="cpu")
mgr = BlockSpaceManager(block_size=BS, num_gpu_blocks=NUM_BLOCKS)
kc, vc = engine.gpu_cache[0]  # layer 0

# 2) New request arrives
seq = Sequence(seq_id=0, prompt_token_ids=[101,102,103,104])  # 4-token prompt
seq.status = SequenceStatus.RUNNING

# 3) Allocate blocks and prefill
mgr.allocate(seq.seq_id, seq.prompt_len)                     # allocate 1 block
slots = torch.tensor(mgr.get_slot_mapping(seq.seq_id, seq.prompt_len))

# model computes q,k,v for the prompt (shapes match PagedAttention API)
q = torch.randn(1, 4, KV_HEADS, HEAD_DIM)
k = torch.randn(1, 4, KV_HEADS, HEAD_DIM)
v = torch.randn(1, 4, KV_HEADS, HEAD_DIM)

out = PagedAttention.forward_prefix(q, k, v, kc, vc, block_size=BS, slot_mapping=slots)

# 4) Decode: append_slot to grow sequence, write new KV, then call decode
slot = mgr.append_slot(seq.seq_id)                            # alloc new block when needed
new_k = torch.randn(1, KV_HEADS, HEAD_DIM)
new_v = torch.randn(1, KV_HEADS, HEAD_DIM)
CacheEngine.write_kv(kc, vc, torch.tensor([slot]), new_k, new_v, BS)

bt = torch.tensor([mgr.get_block_table(seq.seq_id) + [-1]*6])  # padded block table
cl = torch.tensor([seq.total_len + 1])                        # current context length
query = torch.randn(1, 1, KV_HEADS, HEAD_DIM)
decoded = PagedAttention.forward_decode(query, kc, vc, bt, cl, BS)
```
Future work

- CUDA kernel for PagedAttention to improve throughput and reduce latency.
- Rust-based scheduler with lock-free queues for low CPU overhead.
- Iteration-level continuous batching (dynamic batch composition).
- KV cache quantization (INT8/INT4) and block-level compression.
- Minimal public API / CLI for running and benchmarking models.
