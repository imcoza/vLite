"""Tests for the PagedAttention implementation.

Covers:
    1. BlockSpaceManager — alloc, append, free, fork, slots
    2. CacheEngine       — write / read round-trip
    3. PagedAttention     — numerical match against standard attention
    4. End-to-end         — prefill then decode cycle
"""

import math
import torch
import pytest

from vlite.core.block_manager import BlockAllocator, BlockSpaceManager
from vlite.core.kv_cache import CacheEngine
from vlite.core.sequence import Sequence, SequenceStatus
from vlite.kernels.paged_attention import PagedAttention


# ── helpers ────────────────────────────────────────────────────────────

def standard_attention(q, k, v, scale):
    """Reference non-paged attention (no causal mask, single head batch).

    q: [H, D]   k: [T, H, D]   v: [T, H, D]  ->  [H, D]
    """
    k = k.permute(1, 0, 2)           # [H, T, D]
    v = v.permute(1, 0, 2)           # [H, T, D]
    scores = torch.bmm(q.unsqueeze(1), k.transpose(-2, -1)) * scale  # [H,1,T]
    w = torch.softmax(scores.float(), dim=-1)
    out = torch.bmm(w.to(v.dtype), v)  # [H, 1, D]
    return out.squeeze(1)             # [H, D]


# ── 1  BlockSpaceManager ──────────────────────────────────────────────

class TestBlockSpaceManager:

    def test_allocate_and_free(self):
        mgr = BlockSpaceManager(block_size=4, num_gpu_blocks=16)
        table = mgr.allocate(seq_id=0, num_prompt_tokens=10)  # ceil(10/4) = 3
        assert len(table) == 3
        assert mgr.num_free_blocks == 13
        mgr.free(seq_id=0)
        assert mgr.num_free_blocks == 16

    def test_append_slot_within_block(self):
        mgr = BlockSpaceManager(block_size=4, num_gpu_blocks=16)
        mgr.allocate(seq_id=0, num_prompt_tokens=5)   # 2 blocks, last filled=1
        slot = mgr.append_slot(seq_id=0)
        assert len(mgr.get_block_table(0)) == 2  # no new block

    def test_append_slot_new_block(self):
        mgr = BlockSpaceManager(block_size=4, num_gpu_blocks=16)
        mgr.allocate(seq_id=0, num_prompt_tokens=8)   # 2 blocks, last filled=4
        mgr.append_slot(seq_id=0)
        assert len(mgr.get_block_table(0)) == 3  # new block

    def test_fork(self):
        mgr = BlockSpaceManager(block_size=4, num_gpu_blocks=16)
        parent = mgr.allocate(seq_id=0, num_prompt_tokens=10)
        free_before = mgr.num_free_blocks
        child = mgr.fork(parent_id=0, child_id=1)
        assert child == parent
        assert mgr.num_free_blocks == free_before  # shared, no extra blocks

    def test_slot_mapping(self):
        mgr = BlockSpaceManager(block_size=4, num_gpu_blocks=16)
        mgr.allocate(seq_id=0, num_prompt_tokens=6)
        slots = mgr.get_slot_mapping(seq_id=0, seq_len=6)
        assert len(slots) == 6

    def test_oom_raises(self):
        mgr = BlockSpaceManager(block_size=4, num_gpu_blocks=2)
        with pytest.raises(RuntimeError):
            mgr.allocate(seq_id=0, num_prompt_tokens=20)


# ── 2  CacheEngine ────────────────────────────────────────────────────

class TestCacheEngine:

    def test_write_read(self):
        engine = CacheEngine(
            num_layers=1, num_kv_heads=4, head_dim=8,
            num_blocks=8, block_size=4, dtype=torch.float32, device="cpu",
        )
        kc, vc = engine.gpu_cache[0]
        slots = torch.tensor([0, 1, 5])
        keys = torch.randn(3, 4, 8)
        vals = torch.randn(3, 4, 8)
        CacheEngine.write_kv(kc, vc, slots, keys, vals, block_size=4)

        # Read back by decomposing slots.
        bids = slots // 4
        offs = slots % 4
        assert torch.allclose(kc[bids, offs], keys)
        assert torch.allclose(vc[bids, offs], vals)


# ── 3  PagedAttention correctness ─────────────────────────────────────

class TestPagedAttentionCorrectness:

    def test_single_block(self):
        """One sequence fitting in one block."""
        torch.manual_seed(42)
        H, D, BS, T = 4, 8, 16, 10

        kc = torch.zeros(4, BS, H, D)
        vc = torch.zeros(4, BS, H, D)
        keys = torch.randn(T, H, D)
        vals = torch.randn(T, H, D)
        kc[0, :T] = keys
        vc[0, :T] = vals

        query = torch.randn(1, 1, H, D)
        bt = torch.tensor([[0, -1, -1, -1]])
        cl = torch.tensor([T])
        scale = 1.0 / math.sqrt(D)

        out = PagedAttention.forward_decode(
            query, kc, vc, bt, cl, block_size=BS, scale=scale,
        )
        ref = standard_attention(query[0, 0], keys, vals, scale)
        assert torch.allclose(out[0, 0].float(), ref.float(), atol=1e-4)

    def test_multi_block_non_contiguous(self):
        """KV scattered across non-contiguous blocks [1, 5, 3, 7]."""
        torch.manual_seed(123)
        H, D, BS, T = 4, 16, 4, 14

        kc = torch.zeros(8, BS, H, D)
        vc = torch.zeros(8, BS, H, D)
        keys = torch.randn(T, H, D)
        vals = torch.randn(T, H, D)

        phys = [1, 5, 3, 7]
        for i, pb in enumerate(phys):
            s, e = i * BS, min((i + 1) * BS, T)
            kc[pb, :e - s] = keys[s:e]
            vc[pb, :e - s] = vals[s:e]

        query = torch.randn(1, 1, H, D)
        bt = torch.tensor([phys + [-1] * 4])
        cl = torch.tensor([T])
        scale = 1.0 / math.sqrt(D)

        out = PagedAttention.forward_decode(
            query, kc, vc, bt, cl, block_size=BS, scale=scale,
        )
        ref = standard_attention(query[0, 0], keys, vals, scale)
        assert torch.allclose(out[0, 0].float(), ref.float(), atol=1e-4)

    def test_batch_mixed_lengths(self):
        """Batch of 3 sequences with different context lengths."""
        torch.manual_seed(7)
        H, D, BS = 2, 8, 4

        mgr = BlockSpaceManager(block_size=BS, num_gpu_blocks=32)
        kc = torch.zeros(32, BS, H, D)
        vc = torch.zeros(32, BS, H, D)

        lens = [5, 12, 3]
        all_k, all_v = [], []

        for sid, slen in enumerate(lens):
            k = torch.randn(slen, H, D);  all_k.append(k)
            v = torch.randn(slen, H, D);  all_v.append(v)
            mgr.allocate(sid, slen)
            slots = torch.tensor(mgr.get_slot_mapping(sid, slen))
            CacheEngine.write_kv(kc, vc, slots, k, v, BS)

        max_bl = max(len(mgr.get_block_table(i)) for i in range(3))
        tables = []
        for i in range(3):
            t = mgr.get_block_table(i)
            tables.append(t + [-1] * (max_bl - len(t)))

        bt = torch.tensor(tables)
        cl = torch.tensor(lens)
        queries = torch.randn(3, 1, H, D)
        scale = 1.0 / math.sqrt(D)

        out = PagedAttention.forward_decode(
            queries, kc, vc, bt, cl, block_size=BS, scale=scale,
        )

        for i in range(3):
            ref = standard_attention(queries[i, 0], all_k[i], all_v[i], scale)
            assert torch.allclose(out[i, 0].float(), ref.float(), atol=1e-4)


# ── 4  Prefill ────────────────────────────────────────────────────────

class TestPrefill:

    def test_writes_cache_and_returns_output(self):
        torch.manual_seed(0)
        H, D, BS, S = 2, 8, 4, 6

        kc = torch.zeros(4, BS, H, D)
        vc = torch.zeros(4, BS, H, D)

        q = torch.randn(1, S, H, D)
        k = torch.randn(1, S, H, D)
        v = torch.randn(1, S, H, D)
        slots = torch.arange(S)

        out = PagedAttention.forward_prefix(
            q, k, v, kc, vc, block_size=BS, slot_mapping=slots,
        )
        assert out.shape == (1, S, H, D)

        # Verify cache written.
        bids = slots // BS
        offs = slots % BS
        assert torch.allclose(kc[bids, offs], k[0])
        assert torch.allclose(vc[bids, offs], v[0])

    def test_causal_mask(self):
        """First token can only attend to itself -> output == value[0]."""
        torch.manual_seed(1)
        H, D, BS, S = 1, 4, 8, 4

        kc = torch.zeros(2, BS, H, D)
        vc = torch.zeros(2, BS, H, D)

        q = torch.randn(1, S, H, D)
        k = torch.randn(1, S, H, D)
        v = torch.randn(1, S, H, D)
        slots = torch.arange(S)

        out = PagedAttention.forward_prefix(
            q, k, v, kc, vc, block_size=BS, slot_mapping=slots,
        )
        # Token 0 attends only to token 0 -> softmax([x]) = 1 -> output = v0
        assert torch.allclose(out[0, 0, 0].float(), v[0, 0, 0].float(), atol=1e-5)


# ── 5  End-to-end prefill -> decode ───────────────────────────────────

class TestEndToEnd:

    def test_prefill_then_decode(self):
        torch.manual_seed(42)
        num_layers, H, D, BS, num_blocks = 1, 2, 8, 4, 16

        engine = CacheEngine(num_layers, H, D, num_blocks, BS,
                             dtype=torch.float32, device="cpu")
        mgr = BlockSpaceManager(block_size=BS, num_gpu_blocks=num_blocks)
        kc, vc = engine.gpu_cache[0]
        scale = 1.0 / math.sqrt(D)

        # -- prefill 6-token prompt --
        prompt_len = 6
        seq = Sequence(seq_id=0, prompt_token_ids=list(range(prompt_len)))
        seq.status = SequenceStatus.RUNNING

        mgr.allocate(seq.seq_id, prompt_len)
        slots = torch.tensor(mgr.get_slot_mapping(seq.seq_id, prompt_len))

        pq = torch.randn(1, prompt_len, H, D)
        pk = torch.randn(1, prompt_len, H, D)
        pv = torch.randn(1, prompt_len, H, D)

        pout = PagedAttention.forward_prefix(
            pq, pk, pv, kc, vc, BS, slots, scale,
        )
        assert pout.shape == (1, prompt_len, H, D)

        # -- decode 3 tokens --
        cur_len = prompt_len
        for step in range(3):
            slot = mgr.append_slot(seq.seq_id)
            nk = torch.randn(1, H, D)
            nv = torch.randn(1, H, D)
            CacheEngine.write_kv(kc, vc, torch.tensor([slot]), nk, nv, BS)
            cur_len += 1

            table = mgr.get_block_table(seq.seq_id)
            max_bl = len(table)
            bt = torch.tensor([table + [-1] * (8 - max_bl)])
            cl = torch.tensor([cur_len])
            dq = torch.randn(1, 1, H, D)

            dout = PagedAttention.forward_decode(
                dq, kc, vc, bt, cl, BS, scale,
            )
            assert dout.shape == (1, 1, H, D)
            assert not torch.isnan(dout).any()

            seq.append_token(1000 + step)

        assert seq.total_len == 9

        mgr.free(seq.seq_id)
        assert mgr.num_free_blocks == num_blocks


# ── run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
