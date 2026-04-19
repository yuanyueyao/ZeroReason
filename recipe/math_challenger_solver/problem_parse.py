# Copyright 2026 the verl recipe authors
"""Extract the challenger problem from model A output: prefer **Problem:** …, else ```problem``` fence."""

from __future__ import annotations

import re

_PROBLEM_FENCE_RE = re.compile(
    r"```problem\s*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Line starting with **Problem:** or Problem: — note common markdown is * * Problem : * * (colon before closing **).
_PROBLEM_HEADER_RE = re.compile(
    r"(?:^|\n)\s*\*{0,2}\s*Problem\s*:\s*\*{0,2}\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Stop body before a following Solution / Answer / Hint section (markdown headings or bold)
_STOP_AFTER_PROBLEM_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*\*\s*)?(?:Solution|Answer|Hints?)\b(?:\s*\*\*)?\s*[:.]?",
    re.IGNORECASE | re.MULTILINE,
)

# GSM8K-style: stem then "---" then a repeated **Question:** line — not part of the task for B.
_HR_THEN_REST_RE = re.compile(r"\n\s*---\s*\r?\n", re.MULTILINE)

# Reject problem bodies that echo telemetry / meta-quizzes (see history_window + prompt).
_META_SUBSTRINGS: tuple[str, ...] = (
    "on that task the student",
    "independent attempts",
    "parsed numeric answer",
    "distinct answers appeared",
    "majority vote could",
    "earlier item",
    "usable majority vote",
    "successful numeric extracts",
    "solver side:",
    "which scenario presents",
    "task inconsistency",
    "prior_snippet:",
    "stats:r=",
)


def _sanitize_problem_body(body: str) -> str:
    """Trim GSM8K-like duplicate tails and trailing chat special tokens."""
    if not body:
        return body
    m = _HR_THEN_REST_RE.search(body)
    if m:
        body = body[: m.start()].rstrip()
    # vLLM/HF sometimes leave template tokens glued to the last line, e.g. <|im_end|>
    while True:
        stripped = re.sub(r"<\|[^>]+\|>\s*\Z", "", body, flags=re.MULTILINE)
        if stripped == body:
            break
        body = stripped.strip()
    return body.strip()


def problem_content_ok(body: str) -> bool:
    """True if the statement is not obviously a meta-question or history-echo spam."""
    if not body or not body.strip():
        return False
    low = body.lower()
    for s in _META_SUBSTRINGS:
        if s in low:
            return False
    return True


def extract_problem_fence(response: str) -> tuple[str, bool]:
    """
    Returns ``(problem_body, ok)``. ``ok`` is True iff there is **exactly** one ```problem``` block
    and the inner body is non-empty after strip.
    """
    matches = _PROBLEM_FENCE_RE.findall(response)
    if len(matches) != 1:
        return "", False
    body = _sanitize_problem_body(matches[0].strip())
    if not body:
        return "", False
    return body, True


def extract_challenger_problem(response: str) -> tuple[str, bool]:
    """
    Preferred: a single ``**Problem:**`` (or ``Problem:``) section; body runs until Solution/Answer/Hint or end.

    Fallback: exactly one ```problem``` … ``` fence (legacy).

    Returns ``(problem_body, ok)``.
    """
    if not response or not str(response).strip():
        return "", False
    text = str(response)

    prob_iter = list(_PROBLEM_HEADER_RE.finditer(text))
    if len(prob_iter) > 1:
        return "", False
    if len(prob_iter) == 1:
        start = prob_iter[0].end()
        tail = text[start:]
        stop_m = _STOP_AFTER_PROBLEM_RE.search(tail)
        body = _sanitize_problem_body((tail[: stop_m.start()] if stop_m else tail).strip())
        if body:
            return body, True
        # empty after header: try legacy fence in same response
        return extract_problem_fence(response)

    return extract_problem_fence(response)


def extract_problem_fence_batch(responses: list[str]) -> tuple[list[str], list[bool]]:
    problems: list[str] = []
    oks: list[bool] = []
    for r in responses:
        p, ok = extract_challenger_problem(r)
        problems.append(p)
        oks.append(ok)
    return problems, oks
