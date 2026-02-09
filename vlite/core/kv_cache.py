"""
Cache engine — pre-allocated KV cache tensors, mirrors vLLM's CacheEngine.

vLLM stores K and V as *separate* tensors per layer (not combined).
Each tensor: [num_blocks, block_size, num_kv_heads, head_dim]
"""

from __future__ import annotations

from typing import List, Tuple

import torch

# Type alias: one (key_cache, value_cache) pair per layer.
KVCache = Tuple[torch.Tensor, torch.Tensor]


class CacheEngine:
    """Allocate and manage the physical KV cache on a device."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        self.gpu_cache: List[KVCache] = self._allocate(num_blocks)

    # -- allocation ---------------------------------------------------------

    def _allocate(self, num_blocks: int) -> List[KVCache]:
        kv_cache: List[KVCache] = []
        for _ in range(self.num_layers):
            key_cache = torch.zeros(
                num_blocks, self.block_size, self.num_kv_heads, self.head_dim,
                dtype=self.dtype, device=self.device,
            )
            value_cache = torch.zeros(
                num_blocks, self.block_size, self.num_kv_heads, self.head_dim,
                dtype=self.dtype, device=self.device,
            )
            kv_cache.append((key_cache, value_cache))
        return kv_cache

    # -- write --------------------------------------------------------------

    @staticmethod
    def write_kv(
        key_cache: torch.Tensor,    # [num_blocks, block_size, H, D]
        value_cache: torch.Tensor,  # [num_blocks, block_size, H, D]
        slot_indices: torch.Tensor, # [num_tokens]
        keys: torch.Tensor,         # [num_tokens, H, D]
        values: torch.Tensor,       # [num_tokens, H, D]
        block_size: int,
    ) -> None:
        block_ids = slot_indices // block_size
        offsets = slot_indices % block_size
        key_cache[block_ids, offsets] = keys
        value_cache[block_ids, offsets] = values

    # -- copy (for CoW) -----------------------------------------------------

    @staticmethod
    def copy_blocks(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        src: int,
        dst: int,
    ) -> None:
        key_cache[dst] = key_cache[src]
        value_cache[dst] = value_cache[src]
