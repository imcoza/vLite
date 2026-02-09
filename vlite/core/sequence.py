"""Sequence tracking — mirrors vLLM's sequence.py (minimal)."""

from __future__ import annotations

from enum import Enum, auto
from typing import List


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH = auto()
    FINISHED_ABORTED = auto()

    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


class Sequence:
    """Tracks one request: tokens, block table, and status."""

    def __init__(self, seq_id: int, prompt_token_ids: List[int]) -> None:
        self.seq_id = seq_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.status = SequenceStatus.WAITING

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        return self.status.is_finished()

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, prompt={self.prompt_len}, "
            f"output={self.output_len}, status={self.status.name})"
        )
