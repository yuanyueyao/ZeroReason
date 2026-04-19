# Copyright 2026 the verl recipe authors
"""Majority vote over normalized answer keys with deterministic tie-break."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MajorityResult:
    """Outcome of a vote over extracted answers."""

    label: Optional[str]
    """Normalized key that won the vote; None if every entry was invalid."""
    counts: dict[str, int]
    """Per-key vote counts (normalized keys)."""
    tie_broken: bool
    """True if multiple keys tied for max count and tie-break applied."""
    detail: dict[str, Any]


class MajorityVoteLabeler:
    """
    Given ``n`` optional normalized predictions (None = parse failure), return the majority label.

    Tie-break when multiple keys share the max count: pick the key with **lexicographically smallest**
    string representation (stable, reproducible).
    """

    def label(self, preds: list[Optional[str]]) -> MajorityResult:
        valid = [p for p in preds if p is not None]
        if not valid:
            return MajorityResult(
                label=None,
                counts={},
                tie_broken=False,
                detail={"n_in": len(preds), "n_valid": 0},
            )

        ctr = Counter(valid)
        max_c = max(ctr.values())
        top = [k for k, v in ctr.items() if v == max_c]
        tie_broken = len(top) > 1
        winner = sorted(top)[0]

        return MajorityResult(
            label=winner,
            counts=dict(ctr),
            tie_broken=tie_broken,
            detail={"n_in": len(preds), "n_valid": len(valid), "max_count": max_c},
        )
