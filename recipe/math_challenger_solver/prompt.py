# Copyright 2026 the verl recipe authors
"""System / user prompts for math challenger (A) and solver (B)."""

from __future__ import annotations

# Model A: competition-style problem author; output must include <question>…</question> and \boxed{…}.
_SYSTEM_A_COMPETITION = (
    "You are an expert competition-math problem setter.\n"
    "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
    "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
    "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
    "Avoid re-using textbook clichés or famous contest problems.\n"
    "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
    "<question>\n"
    "{The full problem statement on one or more lines}\n"
    "</question>\n\n"
    r"\boxed{final_answer}"
    "\n\n"
    "Do NOT output anything else—no explanations, no extra markup."
)

SYSTEM_A = _SYSTEM_A_COMPETITION

# Used when ``math_challenger.use_problem_history`` is false: no rolling context, i.i.d. problems each step.
SYSTEM_A_NO_HISTORY = _SYSTEM_A_COMPETITION

USER_A_GENERATE = (
    "Generate one new, challenging reasoning question now. "
    "Remember to format the output exactly as instructed."
)

SYSTEM_B = (
    "You are a helpful assistant. Solve the given mathematics problem. Show reasoning if needed; "
    "then put your final answer inside \\boxed{} at the end, e.g. \\boxed{42}."
)


def build_user_prompt_challenger(history_block: str) -> str:
    """
    User message for model A: task + optional history (no ground-truth numbers from voting).
    """
    return f"""Context (read-only excerpts from earlier problems you posed; **not** material to copy into your question):
Each line is: [#i] prior_problem: <truncated excerpt> (no statistics or solver outputs are shown.)
{history_block}

{USER_A_GENERATE}
"""


def build_user_prompt_challenger_no_history() -> str:
    """
    User message for model A when the rolling history window is disabled: same task, no past-round block.
    """
    return USER_A_GENERATE


def build_user_prompt_solver(problem_text: str) -> str:
    """User message for model B given the problem body."""
    return f"""## Problem

{problem_text}

Think step by step, then solve the problem. Put your final answer inside \\boxed{{}}. """
