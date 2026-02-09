"""
Block space manager for PagedAttention — mirrors vLLM's block_manager.

No wrapper dataclasses. Block tables are plain ``list[int]``.
Ref counts are a flat array. The allocator is a free-list stack.
"""

from __future__ import annotations

from typing import Dict, List


class BlockAllocator:
    """Low-level physical block pool.  O(1) alloc / free."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.ref_counts: List[int] = [0] * num_blocks
        self.free_blocks: List[int] = list(range(num_blocks - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("Out of free blocks")
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def free(self, block_id: int) -> None:
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)

    def increase_ref(self, block_id: int) -> None:
        self.ref_counts[block_id] += 1


class BlockSpaceManager:
    """Manages per-sequence block tables — mirrors vLLM's BlockSpaceManager.

    block_tables:  seq_id  ->  list[int]  (physical block ids)
    last_filled:   seq_id  ->  int        (tokens written in last block)
    """

    def __init__(self, block_size: int, num_gpu_blocks: int) -> None:
        self.block_size = block_size
        self.allocator = BlockAllocator(num_gpu_blocks, block_size)
        self.block_tables: Dict[int, List[int]] = {}
        self._last_filled: Dict[int, int] = {}

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    def can_allocate(self, num_tokens: int) -> bool:
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        return self.allocator.num_free >= num_blocks

    # -- lifecycle ----------------------------------------------------------

    def allocate(self, seq_id: int, num_prompt_tokens: int) -> List[int]:
        """Allocate blocks for a new prompt.  Returns the block table."""
        num_blocks = (num_prompt_tokens + self.block_size - 1) // self.block_size
        table = [self.allocator.allocate() for _ in range(num_blocks)]
        self.block_tables[seq_id] = table
        rem = num_prompt_tokens % self.block_size
        self._last_filled[seq_id] = rem if rem else self.block_size
        return table

    def append_slot(self, seq_id: int) -> int:
        """Grow the sequence by one token.  Returns the cache slot index."""
        table = self.block_tables[seq_id]
        filled = self._last_filled[seq_id]

        if not table or filled == self.block_size:
            table.append(self.allocator.allocate())
            filled = 0

        slot = table[-1] * self.block_size + filled
        self._last_filled[seq_id] = filled + 1
        return slot

    def free(self, seq_id: int) -> None:
        """Return all blocks of a finished sequence to the pool."""
        for blk in self.block_tables.pop(seq_id, []):
            self.allocator.free(blk)
        self._last_filled.pop(seq_id, None)

    def fork(self, parent_id: int, child_id: int) -> List[int]:
        """Fork via ref-count (Copy-on-Write).  Returns child's block table."""
        parent_table = self.block_tables[parent_id]
        for blk in parent_table:
            self.allocator.increase_ref(blk)
        child_table = list(parent_table)
        self.block_tables[child_id] = child_table
        self._last_filled[child_id] = self._last_filled[parent_id]
        return child_table

    # -- queries ------------------------------------------------------------

    def get_block_table(self, seq_id: int) -> List[int]:
        return self.block_tables[seq_id]

    def get_slot_mapping(self, seq_id: int, seq_len: int) -> List[int]:
        """Flat slot index for every token in a sequence."""
        table = self.block_tables[seq_id]
        return [
            table[i // self.block_size] * self.block_size + i % self.block_size
            for i in range(seq_len)
        ]
