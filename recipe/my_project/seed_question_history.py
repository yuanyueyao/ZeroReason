# Copyright 2026 the verl recipe authors
#
# 内置若干道简单、可执行的 f + input，用于在训练开始时预填 A 的历史窗口（与 validate 规则一致）。

from __future__ import annotations


def builtin_seed_pairs() -> list[tuple[str, str]]:
    """
    返回 (func_code, input_code) 列表；每段均可被 code_validate_exec 成功 exec/eval/call。
    保持短小，便于占满窗口前的「初试」参考。
    """
    fib = """def f(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"""

    sum_sq = """def f(nums: list) -> int:
    return sum(x * x for x in nums)"""

    return [
        (fib, "10"),
        (sum_sq, "[1, 2, 3, 4]"),
    ]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from recipe.my_project.code_validate_exec import validate_exec_and_call

    for i, (code, inp) in enumerate(builtin_seed_pairs(), 1):
        ok, err = validate_exec_and_call(code, inp)
        print(f"seed #{i}: ok={ok} err={err}")
