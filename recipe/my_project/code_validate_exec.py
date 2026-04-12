# Copyright 2026 the verl recipe authors
#
# Stdlib-only: executed in a spawned subprocess for competition code validation (avoids importing torch).

from __future__ import annotations

import contextlib
import io
import multiprocessing
from typing import Any


def validate_exec_and_call(func_code: str, input_code: str) -> tuple[bool, str | None]:
    """
    exec ``func_code`` (defines ``f``), eval ``input_code`` as ``f``'s arguments, call ``f(*args)``.
    Discards stdout/stderr from user code.
    """
    _devnull = io.StringIO()
    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        try:
            ns: dict[str, Any] = {"__builtins__": __builtins__}
            exec(compile(func_code, "<competition_func>", "exec"), ns)
        except Exception as e:
            return False, f"exec: {e}"

        f = ns.get("f")
        if f is None:
            return False, "missing def f"
        if not callable(f):
            return False, "f is not callable"

        try:
            args = eval(f"({input_code})", {"__builtins__": __builtins__}, ns)
        except Exception as e:
            return False, f"input eval: {e}"

        if not isinstance(args, tuple):
            args = (args,)

        try:
            f(*args)
        except Exception as e:
            return False, f"f(*args): {e}"

    return True, None


def validate_exec_and_call_with_capture(
    func_code: str, input_code: str
) -> tuple[bool, str | None, str]:
    """
    Same as ``validate_exec_and_call`` but returns captured stdout+stderr as the third element
    (for logging). Plain ``return`` values from ``f`` are not printed by Python; on success we
    append a line ``[return] <repr>`` so logs show the call result. Must stay stdlib-only so
    ``spawn`` workers do not import torch.
    """
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        try:
            ns: dict[str, Any] = {"__builtins__": __builtins__}
            exec(compile(func_code, "<competition_func>", "exec"), ns)
        except Exception as e:
            return False, f"exec: {e}", captured.getvalue()

        f = ns.get("f")
        if f is None:
            return False, "missing def f", captured.getvalue()
        if not callable(f):
            return False, "f is not callable", captured.getvalue()

        try:
            args = eval(f"({input_code})", {"__builtins__": __builtins__}, ns)
        except Exception as e:
            return False, f"input eval: {e}", captured.getvalue()

        if not isinstance(args, tuple):
            args = (args,)

        try:
            ret = f(*args)
        except Exception as e:
            return False, f"f(*args): {e}", captured.getvalue()

        captured.write(f"{ret!r}\n")

    return True, None, captured.getvalue()


def _subprocess_entry(result_queue: multiprocessing.Queue, func_code: str, input_code: str) -> None:
    """Picklable entry for ``multiprocessing.Process`` (spawn); imports only this stdlib module."""
    try:
        result_queue.put(validate_exec_and_call_with_capture(func_code, input_code))
    except Exception as e:
        result_queue.put((False, f"worker: {e}", ""))
