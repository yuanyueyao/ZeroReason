# Copyright 2026 the verl recipe authors
"""Generic numeric / symbolic answer extraction from free-form model text."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional


def _extract_last_boxed_inner(text: str) -> Optional[str]:
    """
    Inner text of the last ``\\boxed{...}``, with brace depth so nested ``{ }``
    (e.g. ``\\sqrt{3}``, ``\\frac{a}{b}``) does not terminate the match early.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    j = idx + len("\\boxed")
    while j < len(text) and text[j].isspace():
        j += 1
    if j >= len(text) or text[j] != "{":
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k].strip()
    return None


# Simple fraction written as a/b (digits only) — used by normalize_answer_key
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
    Extracts the scalar answer from the last ``\\boxed{...}`` in the text.
    Returns ``None`` if no ``\\boxed`` is found.
    """

    def __init__(self) -> None:
        pass

    def extract(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None

        inner = _extract_last_boxed_inner(text)
        if not inner:
            return None
        nk = normalize_answer_key(inner)
        if nk is not None:
            return nk
        # strip nested braces one level
        nk2 = normalize_answer_key(inner.replace("{", "").replace("}", ""))
        return nk2
