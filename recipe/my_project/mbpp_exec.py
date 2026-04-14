# Copyright 2026 the verl recipe authors
"""MBPP-style execution check: run model-generated code with MBPP assert lines in a subprocess."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

_CODE_FENCE_PY = re.compile(r"```(?:python|py)\s*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_ANY = re.compile(r"```\s*\r?\n(.*?)```", re.DOTALL)


def extract_python_code(solution_str: str) -> str | None:
    """Take ```python ... ``` if present; else first ``` ... ```; else stripped full string."""
    if not solution_str or not solution_str.strip():
        return None
    m = _CODE_FENCE_PY.search(solution_str)
    if m:
        return m.group(1).strip()
    m = _CODE_FENCE_ANY.search(solution_str)
    if m:
        return m.group(1).strip()
    return solution_str.strip()


def run_mbpp_exec_checks(
    user_code: str,
    *,
    test_setup_code: str,
    test_list: list[str],
    challenge_test_list: list[str],
    timeout_sec: float = 10.0,
) -> tuple[bool, str | None]:
    """
    Execute ``test_setup_code`` + ``user_code`` + asserts (test_list then challenge_test_list).

    Returns (all_passed, error_message_or_None).
    """
    setup = (test_setup_code or "").strip()
    body = (user_code or "").strip()
    tests = [t.strip() for t in (test_list or []) if t and str(t).strip()]
    ch = [t.strip() for t in (challenge_test_list or []) if t and str(t).strip()]
    all_lines = tests + ch
    if not body:
        return False, "empty_extracted_code"
    if not all_lines:
        return False, "no_tests"

    parts = []
    if setup:
        parts.append(setup)
    parts.append(body)
    parts.append("\n".join(all_lines))
    full_src = "\n\n".join(parts) + "\n"

    fd, path = tempfile.mkstemp(suffix="_mbpp_eval.py", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(full_src)
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=None,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            return False, (err[:500] if err else f"exit_{proc.returncode}")
        return True, None
    except subprocess.TimeoutExpired:
        return False, f"timeout ({timeout_sec}s)"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def compute_mbpp_score_dict(
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None,
    *,
    timeout_sec: float = 10.0,
) -> dict:
    """
    Return a dict compatible with NaiveRewardManager: must include ``score`` (float).

    Uses ``extra_info["mbpp"]`` for test lists; ``ground_truth`` is unused for pass/fail
    (reference is only for logging / other metrics).
    """
    extra_info = extra_info or {}
    mbpp = extra_info.get("mbpp")
    if not isinstance(mbpp, dict):
        return {"score": 0.0, "mbpp_ok": False, "mbpp_err": "missing extra_info.mbpp"}

    test_list = list(mbpp.get("test_list") or [])
    setup = str(mbpp.get("test_setup_code") or "")
    challenge = list(mbpp.get("challenge_test_list") or [])

    user = extract_python_code(solution_str)
    if user is None:
        return {"score": 0.0, "mbpp_ok": False, "mbpp_err": "no_code_extracted"}

    ok, err = run_mbpp_exec_checks(
        user,
        test_setup_code=setup,
        test_list=test_list,
        challenge_test_list=challenge,
        timeout_sec=timeout_sec,
    )
    score = 1.0 if ok else 0.0
    out: dict = {"score": score, "mbpp_ok": ok}
    if err:
        out["mbpp_err"] = err[:300]
    return out
