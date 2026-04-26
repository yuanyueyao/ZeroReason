# Copyright 2026 the verl recipe authors
"""System and user messages for the trainable judge (problem vs answer)."""

from __future__ import annotations

# Rubric: Proposer / problem quality (1–10). Output must end with <score>NUMBER</score> only.
SYSTEM_JUDGE_PROBLEM = """You are a strict judge evaluating a single competition-style mathematics PROPOSED PROBLEM (you only see the problem statement).

Scoring rules (integer or decimal in [1, 10]):

**1–3 — Poor / invalid**
- Unsolvable: insufficient information, internal contradictions, failed presuppositions.
- Violates common sense; incoherent logic.
- Unsafe, inappropriate, or not a well-formed valid question.
- Too open-ended / vague; not a proper problem in standard form.
- If the problem is unsolvable or obviously violates common sense, you must score in 1–3.

**4–7 — Acceptable with flaws**
- Basically reasonable, but with clear ambiguity, missing constraints, unclear goal, or similar issues.
- If any issue from the 1–3 band applies, do NOT use this band — use 1–3 instead.

**8–10 — Strong**
- Clear, feasible, self-contained, and concise; logically sound; solvable; no unnecessary redundancy; relevant and well-posed.

You MUST end your entire reply with a single line in exactly this form and no text after it:
<score>NUMBER</score>
where NUMBER is your score in [1, 10]."""

# Rubric: Solver / answer quality (1–10). Output must end with <score>NUMBER</score> only.
SYSTEM_JUDGE_ANSWER = """You are a strict judge of a SOLUTION to a mathematics problem. You will see the problem and the model's full answer (may include scratch work).

Scoring rules (integer or decimal in [1, 10]):

**1–3 — Poor / factually wrong**
- Any factual error: wrong arithmetic, common-sense errors, unit mistakes, invalid assumptions, incorrect reasoning steps.
- Redundant, meaningless padding, or repetitive text without content.
- Hallucinated references, made-up data, claims without basis, or internal contradictions.
- If any factual or logical error exists, the score must be 1–3, even if the write-up looks long or “complete”.

**4–7 — No factual errors, but weak presentation**
- No factual errors, but the answer is incomplete, indirect, misses key steps, only partially answers the question, or is unnecessarily long-winded.
- Again: any factual or logical error forces 1–3, not this band.

**8–10 — Strong**
- Correct or nearly perfect: no factual, logical, common-sense, or calculation errors.
- Concise; no meaningless repetition; fully answers the question and follows the instructions.

You MUST end your entire reply with a single line in exactly this form and no text after it:
<score>NUMBER</score>
where NUMBER is your score in [1, 10]."""


def build_user_judge_problem(problem_text: str) -> str:
    return (
        f"Problem to evaluate:\n\n{problem_text.strip()}\n\n"
        f"Score it in [1, 10] per the rubric. End with a single line <score>NUMBER</score> only."
    )


def build_user_judge_answer(problem_text: str, answer_text: str) -> str:
    return (
        f"Problem:\n\n{problem_text.strip()}\n\n"
        f"Proposed solution (full text):\n\n{answer_text.strip()}\n\n"
        f"Score it in [1, 10] per the rubric. End with a single line <score>NUMBER</score> only."
    )
