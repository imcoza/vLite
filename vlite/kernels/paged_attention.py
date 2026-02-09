"""
PagedAttention kernel — mirrors vLLM's ``PagedAttention`` class.

Two static entry points matching vLLM's naming:
    forward_prefix  — prefill phase (many Q tokens, causal self-attention).
    forward_decode  — decode phase  (one Q token, online softmax over blocks).

Both take raw key_cache / value_cache tensors directly, NOT a cache-engine
object.  This keeps the kernel layer independent of the management layer,
exactly like vLLM.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


class PagedAttention:
    """Pure-PyTorch paged attention matching vLLM's kernel interface."""

    # ------------------------------------------------------------------ #
    #  PREFILL  (prompt processing)                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,         # [B, S, num_q_heads, head_dim]
        key: torch.Tensor,           # [B, S, num_kv_heads, head_dim]
        value: torch.Tensor,         # [B, S, num_kv_heads, head_dim]
        key_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: torch.Tensor,   # [num_blocks, block_size, num_kv_heads, head_dim]
        block_size: int,
        slot_mapping: torch.Tensor,  # [total_tokens]
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Prefill: write K/V to cache, compute causal self-attention.

        Returns: [B, S, num_q_heads, head_dim]
        """
        B, S, num_q_heads, head_dim = query.shape
        num_kv_heads = key.shape[2]
        heads_per_kv = num_q_heads // num_kv_heads

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # 1. Write K/V into the paged cache.
        total = B * S
        block_ids = slot_mapping // block_size
        offsets = slot_mapping % block_size
        key_cache[block_ids, offsets] = key.reshape(total, num_kv_heads, head_dim)
        value_cache[block_ids, offsets] = value.reshape(total, num_kv_heads, head_dim)

        # 2. Causal self-attention (standard — tokens are contiguous at prefill).
        k, v = key, value
        if heads_per_kv > 1:
            k = k.repeat_interleave(heads_per_kv, dim=2)
            v = v.repeat_interleave(heads_per_kv, dim=2)

        # [B, H, S, D]
        q = query.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(
            torch.ones(S, S, device=query.device, dtype=torch.bool), diagonal=1,
        )
        scores.masked_fill_(causal_mask, float("-inf"))

        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        out = torch.matmul(weights, v)  # [B, H, S, D]

        return out.transpose(1, 2)      # [B, S, H, D]

    # ------------------------------------------------------------------ #
    #  DECODE  (one query token per sequence, online softmax over blocks) #
    # ------------------------------------------------------------------ #

    @staticmethod
    def forward_decode(
        query: torch.Tensor,         # [batch, 1, num_q_heads, head_dim]
        key_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: torch.Tensor,   # [num_blocks, block_size, num_kv_heads, head_dim]
        block_tables: torch.Tensor,  # [batch, max_num_blocks]  (-1 = padding)
        context_lens: torch.Tensor,  # [batch]
        block_size: int,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Decode: one query token, read scattered KV blocks, online softmax.

        Returns: [batch, 1, num_q_heads, head_dim]
        """
        batch_size, _, num_q_heads, head_dim = query.shape
        num_kv_heads = key_cache.shape[2]
        heads_per_kv = num_q_heads // num_kv_heads

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        output = torch.zeros_like(query, dtype=torch.float32)

        for b in range(batch_size):
            ctx_len = context_lens[b].item()
            if ctx_len == 0:
                continue

            num_blocks = (ctx_len + block_size - 1) // block_size
            q = query[b, 0].float()  # [num_q_heads, head_dim]

            # Online softmax accumulators — all heads at once.
            m = torch.full((num_q_heads,), float("-inf"),
                           dtype=torch.float32, device=query.device)
            l = torch.zeros(num_q_heads,
                            dtype=torch.float32, device=query.device)
            acc = torch.zeros(num_q_heads, head_dim,
                              dtype=torch.float32, device=query.device)

            for bi in range(num_blocks):
                phys = block_tables[b, bi].item()
                if phys < 0:
                    break

                toks = min(block_size, ctx_len - bi * block_size)

                # [T, kv_heads, D]
                k_blk = key_cache[phys, :toks].float()
                v_blk = value_cache[phys, :toks].float()

                # GQA expansion: [T, kv_heads, D] -> [T, q_heads, D]
                if heads_per_kv > 1:
                    k_blk = k_blk.repeat_interleave(heads_per_kv, dim=1)
                    v_blk = v_blk.repeat_interleave(heads_per_kv, dim=1)

                # -> [H, T, D]
                k_blk = k_blk.permute(1, 0, 2)
                v_blk = v_blk.permute(1, 0, 2)

                # scores: [H, T]
                scores = torch.bmm(k_blk, q.unsqueeze(-1)).squeeze(-1) * scale

                # -- online softmax --
                block_max, _ = scores.max(dim=-1)
                new_m = torch.maximum(m, block_max)

                rescale = torch.exp(m - new_m)
                l = l * rescale
                acc = acc * rescale.unsqueeze(-1)

                w = torch.exp(scores - new_m.unsqueeze(-1))  # [H, T]
                l = l + w.sum(dim=-1)
                acc = acc + torch.bmm(w.unsqueeze(1), v_blk).squeeze(1)

                m = new_m

            l = l.clamp(min=1e-9)
            output[b, 0] = acc / l.unsqueeze(-1)

        return output.to(query.dtype)
