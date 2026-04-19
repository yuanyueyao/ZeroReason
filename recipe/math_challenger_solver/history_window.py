# Copyright 2026 the verl recipe authors
"""Rolling window of past problem texts only (no B-side stats, no majority label)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class HistoryEntry:
    """One prior problem snippet shown to A; no solver telemetry."""

    problem_excerpt: str


class ProblemHistoryWindow:
    """
    FIFO window of :class:`HistoryEntry` for building A's user prompt.

    Stores **only** past problem excerpts. Does not record or expose B 的作答情况、
    多数票或任何聚合统计。
    """

    def __init__(self, maxlen: int = 10) -> None:
        self._maxlen = max(0, int(maxlen))
        self._q: Deque[HistoryEntry] = deque(maxlen=self._maxlen if self._maxlen > 0 else None)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._q)

    def append(self, entry: HistoryEntry) -> None:
        if self._maxlen == 0:
            return
        self._q.append(entry)

    def format_for_user_prompt(self, max_excerpt_chars: int = 400) -> str:
        """
        One line per prior problem: index + truncated excerpt only (no stats).
        """
        if not self._q:
            return "No earlier records."

        lines: list[str] = []
        for i, e in enumerate(self._q):
            ex = e.problem_excerpt.replace("\n", " ").strip()
            if len(ex) > max_excerpt_chars:
                ex = ex[: max_excerpt_chars - 3] + "..."
            lines.append(f"[#{i + 1}] prior_problem: {ex}")
        return "\n".join(lines)
