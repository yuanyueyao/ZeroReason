# Copyright 2026 the verl recipe authors
"""Parse ``<score>...</score>`` from judge model output."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

_SCORE_RE = re.compile(
    r"<score>\s*([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)\s*</score>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ParseScoreResult:
    value: float | None
    parse_ok: bool


def extract_score_tag(response_text: str) -> ParseScoreResult:
    if not response_text or not str(response_text).strip():
        return ParseScoreResult(value=None, parse_ok=False)
    m = _SCORE_RE.search(str(response_text))
    if m is None:
        return ParseScoreResult(value=None, parse_ok=False)
    try:
        v = float(m.group(1).strip())
    except (TypeError, ValueError):
        return ParseScoreResult(value=None, parse_ok=False)
    return ParseScoreResult(value=v, parse_ok=True)


def normalize_score_to_01(
    value: float | None, *, score_min: float, score_max: float, parse_ok: bool
) -> float:
    if not parse_ok or value is None:
        return 0.0
    if score_max <= score_min:
        return 0.0
    t = (float(value) - float(score_min)) / (float(score_max) - float(score_min))
    return max(0.0, min(1.0, t))


def score_in_range(
    value: float | None, *, score_min: float, score_max: float, parse_ok: bool
) -> bool:
    if not parse_ok or value is None:
        return False
    v = float(value)
    return float(score_min) <= v <= float(score_max)


def rounded_score_bucket(
    value: float | None, *, parse_ok: bool, score_min: float, score_max: float
) -> int | None:
    """Same rounding/clamp as majority_vote_judge_score; for comparing a row to the voted mode."""
    if not parse_ok or value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    lo, hi = int(round(score_min)), int(round(score_max))
    ri = int(round(x))
    return max(lo, min(hi, ri))


def majority_vote_judge_score(
    values: list,
    parse_ok: list[bool],
    *,
    score_min: float,
    score_max: float,
) -> tuple[float | None, int | None]:
    """
    Round each parsed value to an integer in [score_min, score_max], then plurality vote.
    Tie-break: prefer the int with the most votes; if still tied, take the larger int.

    Returns:
        (representative, mode_int) where representative is the median of raw scores in the
        winning bucket (for A/B shaping), and mode_int is the winning integer (for per-row
        “matches majority” bonuses). Both None if no valid votes.
    """
    if not values or not parse_ok or len(values) != len(parse_ok):
        return None, None
    buckets: list[tuple[int, float]] = []
    for v, ok in zip(values, parse_ok):
        if not ok or v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        lo, hi = int(round(score_min)), int(round(score_max))
        ri = int(round(x))
        ri = max(lo, min(hi, ri))
        buckets.append((ri, x))
    if not buckets:
        return None, None
    c = Counter(r for r, _ in buckets)
    best = max(c.values())
    winners = sorted([k for k, n in c.items() if n == best], reverse=True)
    mode_int = winners[0]
    in_mode = [x for r, x in buckets if r == mode_int]
    rep = float(median(in_mode)) if in_mode else float(mode_int)
    return rep, mode_int
