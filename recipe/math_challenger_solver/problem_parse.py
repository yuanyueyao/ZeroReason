# Copyright 2026 the verl recipe authors
"""Extract the challenger problem from model A output: exactly one non-empty <question>…</question> block."""

from __future__ import annotations

import re

_QUESTION_TAG_RE = re.compile(
    r"<question>\s*(.*?)\s*</question>",
    re.DOTALL | re.IGNORECASE,
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


def extract_challenger_problem(response: str) -> tuple[str, bool]:
    """
    Accept **only** a single non-empty ``<question>…</question>`` block.

    Returns ``(problem_body, ok)``. ``ok`` is True iff there is exactly one such block and, after
    :func:`_sanitize_problem_body`, the inner text is non-empty. Zero, multiple, or empty blocks all fail.
    """
    if not response or not str(response).strip():
        return "", False
    text = str(response)
    q_tags = _QUESTION_TAG_RE.findall(text)
    if len(q_tags) != 1:
        return "", False
    body = _sanitize_problem_body(q_tags[0].strip())
    if not body:
        return "", False
    return body, True
