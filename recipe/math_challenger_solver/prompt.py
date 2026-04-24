# Copyright 2026 the verl recipe authors
"""System / user prompts for math challenger (A) and solver (B)."""

from __future__ import annotations

SYSTEM_A = (
    "You are a skilled mathematics problem author. You propose one standalone math question for another model. "
    "Past rounds may appear only as short problem excerpts—use them to judge difficulty and avoid repetition; "
    "do not copy their wording into your problem. End with a line **Problem:** (exact label) followed by the full statement."
)

# Used when ``math_challenger.use_problem_history`` is false: no rolling context, i.i.d. problems each step.
SYSTEM_A_NO_HISTORY = (
    "You are a skilled mathematics problem author. Each time you write one clear, self-contained mathematics problem "
    "for another model to solve. End with a line **Problem:** (exact label) followed by the full statement."
)

SYSTEM_B = (
    "You are a helpful assistant. Solve the given mathematics problem. Show reasoning if needed; "
    "then put your final answer inside \\boxed{} at the end, e.g. \\boxed{42}."
)


def build_user_prompt_challenger(history_block: str) -> str:
    """
    User message for model A: task + optional history (no ground-truth numbers from voting).
    """
    return f"""Propose **one** new mathematics problem for a solver model.

Context (read-only excerpts from earlier problems you posed; **not** material to paste into your question):
Each line is: [#i] prior_problem: <truncated excerpt> (no statistics or solver outputs are shown.)
{history_block}

How to format your reply:
- You may think or plan **above** the problem.
- Then output a the question after **Problem:** 

Example:
**Problem:** what is the square root of 16?

Now give me a new problem after **Problem:**
"""


def build_user_prompt_challenger_no_history() -> str:
    """
    User message for model A when the rolling history window is disabled: same task, no past-round block.
    """
    return """Propose **one** new mathematics problem: challenging but fair, self-contained, with a clear question (exam or contest style).

How to format your reply:
- You may think or plan **above** the problem.
- Then a single line exactly: **Problem:**
- Below it, state **only** the mathematical task (no solution, no rubric).
- Do not add **Solution:** after the problem.

Prefer one well-defined question; you may use displayed equations and short notation."""


def build_user_prompt_solver(problem_text: str) -> str:
    """User message for model B given the problem body."""
    return f"""## Problem

{problem_text}

Think step by step, then solve the problem. Put your final answer inside \\boxed{{}}. """
