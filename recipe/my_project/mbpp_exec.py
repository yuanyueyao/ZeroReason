# Copyright 2026 the verl recipe authors
"""MBPP-style execution check: run model-generated code with MBPP assert lines in a subprocess."""

from __future__ import annotations

import ast
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


def _first_top_level_function_name(src: str) -> str | None:
    """First ``def`` at module level (MBPP reference / student entry point)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def _is_dunder_name_main_guard(node: ast.If) -> bool:
    t = node.test
    if not isinstance(t, ast.Compare) or len(t.ops) != 1 or not isinstance(t.ops[0], ast.Eq):
        return False
    if not isinstance(t.left, ast.Name) or t.left.id != "__name__":
        return False
    if len(t.comparators) != 1:
        return False
    c = t.comparators[0]
    if isinstance(c, ast.Constant) and isinstance(c.value, str):
        return c.value == "__main__"
    return False


def strip_mbpp_student_code(src: str) -> str:
    """
    Drop module-level side effects that break MBPP harness execution:

    - bare expressions (e.g. ``check_solution()``, demo ``print(...)``)
    - ``if __name__ == "__main__":`` blocks

    Official asserts are appended *after* this block; those must be the first code that runs
    after definitions (besides imports / assignments).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    new_body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.If) and _is_dunder_name_main_guard(node):
            continue
        if isinstance(node, ast.Expr):
            continue
        new_body.append(node)
    if not new_body:
        return src
    out = ast.Module(body=new_body, type_ignores=getattr(tree, "type_ignores", []))
    try:
        return ast.unparse(out)
    except Exception:
        return src


def maybe_entrypoint_alias_line(reference_code: str, student_code: str) -> str:
    """
    MBPP ``test_list`` asserts call the **reference** function name (e.g. ``remove_Occ``).
    Models often rename (e.g. ``remove_first_last_occurrence``). Emit ``ref = student`` when
    both sides expose a single top-level name and they differ.
    """
    ref = (reference_code or "").strip()
    stu = (student_code or "").strip()
    if not ref or not stu:
        return ""
    rname = _first_top_level_function_name(ref)
    sname = _first_top_level_function_name(stu)
    if not rname or not sname or rname == sname:
        return ""
    return f"{rname} = {sname}"


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

    Uses ``extra_info["mbpp"]`` for test lists. ``ground_truth`` (reference solution) is used to:

    - strip module-level demo calls / ``__main__`` blocks from student code;
    - add ``<ref_name> = <student_name>`` when MBPP asserts use the dataset entrypoint name.
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

    ref = (ground_truth or "").strip() or str((extra_info or {}).get("reference_code") or "").strip()
    student = strip_mbpp_student_code(user)
    alias = maybe_entrypoint_alias_line(ref, student)
    body = student if not alias else f"{student}\n\n{alias}"

    ok, err = run_mbpp_exec_checks(
        body,
        test_setup_code=setup,
        test_list=test_list,
        challenge_test_list=challenge,
        timeout_sec=timeout_sec,
    )
    score = 1.0 if ok else 0.0
    # Always return the same keys; use "" when no stderr (avoids None in numpy stats paths).
    err_s = err[:300] if err else ""
    return {"score": score, "mbpp_ok": ok, "mbpp_err": err_s}
