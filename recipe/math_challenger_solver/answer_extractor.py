# Copyright 2026 the verl recipe authors
"""Generic numeric / symbolic answer extraction from free-form model text."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional


# \boxed{...} — inner may contain LaTeX; we take a best-effort slice
_BOXED_RE = re.compile(r"\\boxed\s*\{([^}]*)\}", re.DOTALL)
# Final answer style lines
_ANSWER_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s*answer|answer)\s*[:：]\s*([^\n]+)",
    re.IGNORECASE | re.MULTILINE,
)
# Integer or float (optional sign / scientific)
_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
# Simple fraction written as a/b (digits only)
_FRAC_RE = re.compile(r"[+-]?\d+\s*/\s*[+-]?\d+")


def _strip_wrappers(s: str) -> str:
    s = s.strip().strip("$").strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    return s


def normalize_answer_key(raw: Optional[str]) -> Optional[str]:
    """
    Normalize an extracted answer string to a canonical key for equality / voting.

    Returns None if empty or unparseable.
    """
    if raw is None:
        return None
    s = _strip_wrappers(raw.strip())
    if not s:
        return None

    # Try exact rational
    frac_m = _FRAC_RE.fullmatch(s.replace(" ", ""))
    if frac_m:
        try:
            parts = s.replace(" ", "").split("/")
            if len(parts) == 2:
                return str(Fraction(int(parts[0]), int(parts[1])))
        except (ValueError, ZeroDivisionError):
            pass

    # Try float / int
    try:
        v = float(s.replace(",", ""))
        if v == int(v):
            return str(int(v))
        return f"{v:.12g}"
    except ValueError:
        pass

    # Fallback: stable lowercase strip for non-numeric leftovers
    return s.lower().strip() or None


class AnswerExtractor:
    """
    Heuristic extraction of a single scalar answer from arbitrary completion text.

    Order: ``\\boxed{...}`` → ``Answer:`` line → last simple fraction → last float-like token.
    """

    def __init__(self) -> None:
        pass

    def extract(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None

        # 1) \\boxed{...}
        boxed = _BOXED_RE.findall(text)
        if boxed:
            inner = boxed[-1].strip()
            # inner might be "42" or "1/2" or messy TeX — try normalize on raw inner first
            nk = normalize_answer_key(inner)
            if nk is not None:
                return nk
            # strip nested braces one level
            nk2 = normalize_answer_key(inner.replace("{", "").replace("}", ""))
            if nk2 is not None:
                return nk2

        # 2) "Answer:" line
        al = _ANSWER_LINE_RE.findall(text)
        if al:
            nk = normalize_answer_key(al[-1].strip())
            if nk is not None:
                return nk

        # 3) Last a/b fraction
        fracs = _FRAC_RE.findall(text)
        if fracs:
            nk = normalize_answer_key(fracs[-1])
            if nk is not None:
                return nk

        # 4) Last float-like number in the string (common for CoT)
        nums = _FLOAT_RE.findall(text)
        if nums:
            nk = normalize_answer_key(nums[-1])
            if nk is not None:
                return nk

        return None
