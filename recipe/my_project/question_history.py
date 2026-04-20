# Copyright 2026 the verl recipe authors
#
# Sliding-window history of validated Model-A puzzles (code + input) for prompt conditioning.

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    # Reserve space for the marker; keep head + tail so embedded ``` cannot break outer fences easily
    marker = "\n... [truncated] ...\n"
    budget = max_chars - len(marker)
    if budget < 8:
        return s[:max_chars]
    head = budget // 2
    tail = budget - head
    return s[:head] + marker + s[-tail:]


class QuestionHistoryWindow:
    """Stores recent (func_code, input_code) pairs; formats a block for the A-side user prompt."""

    def __init__(
        self,
        *,
        enable: bool = True,
        max_entries: int = 32,
        max_chars_per_code: int = 6000,
        max_chars_per_input: int = 800,
    ) -> None:
        self.enable = bool(enable)
        self.max_entries = max(1, int(max_entries))
        self.max_chars_per_code = int(max_chars_per_code)
        self.max_chars_per_input = int(max_chars_per_input)
        self._order: deque[tuple[str, str]] = deque(maxlen=self.max_entries)
        self._seen: set[str] = set()

    def clear(self) -> None:
        self._order.clear()
        self._seen.clear()

    def _fingerprint(self, code: str, inp: str) -> str:
        blob = (code.strip() + "\n---\n" + inp.strip()).encode("utf-8", errors="replace")
        return hashlib.sha256(blob).hexdigest()

    def add(self, func_code: str, input_code: str) -> bool:
        """Append one puzzle if enabled and not duplicate of an entry already in the window. Returns True if added."""
        if not self.enable:
            return False
        fp = self._fingerprint(func_code, input_code)
        if fp in self._seen:
            return False
        if len(self._order) == self.max_entries:
            old_code, old_inp = self._order[0]
            self._seen.discard(self._fingerprint(old_code, old_inp))
        self._order.append((func_code, input_code))
        self._seen.add(fp)
        return True

    def add_many(self, pairs: list[tuple[str, str]]) -> int:
        """Add multiple (code, input) pairs; returns count newly added."""
        n = 0
        for c, i in pairs:
            if self.add(c, i):
                n += 1
        return n

    def format_for_prompt(self) -> str:
        """Non-empty markdown section listing past puzzles, or empty string if none / disabled."""
        if not self.enable or not self._order:
            return ""
        parts: list[str] = []
        for idx, (code, inp) in enumerate(self._order, start=1):
            c = _truncate(code, self.max_chars_per_code)
            ins = _truncate(inp, self.max_chars_per_input)
            parts.append(
                f"### Historical puzzle #{idx}\n\n"
                f"```python\n{c}\n```\n\n"
                f"```input\n{ins}\n```\n"
            )
        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self._order)

    def state_dict(self) -> dict[str, Any]:
        return {
            "enable": self.enable,
            "max_entries": self.max_entries,
            "max_chars_per_code": self.max_chars_per_code,
            "max_chars_per_input": self.max_chars_per_input,
            "entries": list(self._order),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.clear()
        self.enable = bool(d.get("enable", self.enable))
        self.max_entries = max(1, int(d.get("max_entries", self.max_entries)))
        self.max_chars_per_code = int(d.get("max_chars_per_code", self.max_chars_per_code))
        self.max_chars_per_input = int(d.get("max_chars_per_input", self.max_chars_per_input))
        self._order = deque(maxlen=self.max_entries)
        self._seen = set()
        entries = list(d.get("entries") or [])
        if len(entries) > self.max_entries:
            entries = entries[-self.max_entries :]
        for code, inp in entries:
            if not isinstance(code, str) or not isinstance(inp, str):
                continue
            self._order.append((code, inp))
            self._seen.add(self._fingerprint(code, inp))
