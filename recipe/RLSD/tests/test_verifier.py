"""is_correct：仅最后一个 \\boxed{} 内与 GT 做 math_verify。"""

import pytest

pytest.importorskip("math_verify")

from recipe.RLSD.rlsd.verifier import extract_boxed_answer, is_correct


def test_no_boxed_false():
    txt = r"""Solve: \[ x = 12 \] So answer is 12."""
    assert extract_boxed_answer(txt) is None
    assert is_correct(txt, "12") is False


def test_boxed_matches_gt():
    txt = r"Steps... Final \(\boxed{12}\)."
    assert extract_boxed_answer(txt) == "12"
    assert is_correct(txt, "12") is True


def test_last_boxed_wins():
    txt = r"Draft \(\boxed{3}\). Correct: \(\boxed{12}\)"
    assert extract_boxed_answer(txt) == "12"
    assert is_correct(txt, "12") is True
    assert is_correct(txt, "3") is False


def test_empty_boxed_false():
    txt = r"\boxed{}"
    assert extract_boxed_answer(txt) == ""
    assert is_correct(txt, "12") is False


def test_fbox_accepted():
    txt = r"Answer \fbox{$\frac{6}{2}$}"
    inner = extract_boxed_answer(txt)
    assert inner is not None
    assert is_correct(txt, "3") is True
