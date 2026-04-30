"""
数学答案验证器。

从模型输出中提取 \\boxed{} 答案并与 ground-truth 比较。

修复点：
  1. remove_boxed 支持 \\fbox{} 前缀
  2. LaTeX→sympy 转换使用递归括号提取，正确处理嵌套括号
     （如 \\frac{\\sqrt{2}}{3}，[^}]+ 正则无法处理此类情况）
"""

import re
from typing import Optional


# ══════════════════════════════════════════════════════════════
# 括号感知工具函数
# ══════════════════════════════════════════════════════════════

def _find_matching_brace(s: str, open_pos: int) -> int:
    """
    给定字符串 s 和左花括号位置 open_pos（s[open_pos] == '{'），
    返回对应右花括号的位置。若括号不平衡则返回 -1。
    """
    assert s[open_pos] == "{", f"Expected '{{' at pos {open_pos}, got {s[open_pos]!r}"
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1  # 不平衡


def _extract_brace_group(s: str, pos: int) -> tuple[str, int]:
    """
    从 pos 处（s[pos] 应为 '{'）提取一个完整括号组。
    返回 (内容, 括号组结束后的位置)。
    """
    if pos >= len(s) or s[pos] != "{":
        return "", pos
    close = _find_matching_brace(s, pos)
    if close == -1:
        return s[pos + 1 :], len(s)
    return s[pos + 1 : close], close + 1


# ══════════════════════════════════════════════════════════════
# 提取 \\boxed{} / \\fbox{} 内容
# ══════════════════════════════════════════════════════════════

def last_boxed_only_string(string: str) -> Optional[str]:
    """
    返回最后一个 \\boxed{...} 或 \\fbox{...} 的完整字符串（含嵌套括号）。
    使用括号计数而非正则，正确处理嵌套括号。
    """
    # 按优先级搜索：先找 \boxed，再找 \fbox
    for marker in (r"\boxed", r"\fbox"):
        idx = string.rfind(marker)
        if idx < 0:
            continue
        # 找到 marker 后的第一个 '{'
        brace_start = string.find("{", idx + len(marker))
        if brace_start == -1:
            continue
        brace_end = _find_matching_brace(string, brace_start)
        if brace_end == -1:
            continue
        return string[idx : brace_end + 1]
    return None


def remove_boxed(s: str) -> str:
    """
    去掉 \\boxed{ / \\fbox{ 前缀和匹配的右括号，返回内部内容。
    正确处理嵌套括号。
    """
    for prefix in (r"\boxed{", r"\fbox{"):
        if s.startswith(prefix):
            inner = s[len(prefix):]
            # 去掉最后一个匹配的 '}'
            # 因为 last_boxed_only_string 保证 s 整体括号平衡，
            # 直接去掉末尾 '}' 是安全的。
            if inner.endswith("}"):
                return inner[:-1]
            return inner
    # 兜底：去掉首尾的 \boxed/\fbox 包装（非标准格式）
    for prefix in (r"\boxed", r"\fbox"):
        if s.startswith(prefix):
            brace_start = s.find("{", len(prefix))
            if brace_start != -1:
                content, _ = _extract_brace_group(s, brace_start)
                return content
    return s


def extract_boxed_answer(solution: str) -> Optional[str]:
    """从模型生成文本中提取最终 boxed 答案字符串。"""
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return None
    return remove_boxed(boxed)


# ══════════════════════════════════════════════════════════════
# LaTeX → sympy 可解析字符串（括号感知转换）
# ══════════════════════════════════════════════════════════════

def _latex_to_expr(s: str) -> str:
    """
    将 LaTeX 数学表达式转换为 sympy 可解析的字符串。
    使用递归括号提取，正确处理嵌套括号。

    支持：
      \\frac{num}{den}  → (num_expr)/(den_expr)
      \\sqrt{arg}       → sqrt(arg_expr)
      \\sqrt[n]{arg}    → (arg_expr)**(1/n)
      \\left( \\right)  → ( )
      \\cdot            → *
      \\times           → *
      \\pi              → pi
      \\infty           → oo
      \\pm              → +  (取正值，仅作兜底)
      { }               → ( )  兜底替换
    """
    s = s.strip()

    # ── 处理 \frac{num}{den} ──
    # 不能用 [^}]+ 正则，改用括号感知提取
    result = []
    i = 0
    while i < len(s):
        # \frac
        if s[i:].startswith(r"\frac") and i + 5 < len(s):
            after = i + len(r"\frac")
            # 跳过空白
            while after < len(s) and s[after] == " ":
                after += 1
            if after < len(s) and s[after] == "{":
                num_content, after = _extract_brace_group(s, after)
                # 跳过空白
                while after < len(s) and s[after] == " ":
                    after += 1
                if after < len(s) and s[after] == "{":
                    den_content, after = _extract_brace_group(s, after)
                    num_expr = _latex_to_expr(num_content)
                    den_expr = _latex_to_expr(den_content)
                    result.append(f"({num_expr})/({den_expr})")
                    i = after
                    continue
        # \sqrt[n]{arg}  或  \sqrt{arg}
        if s[i:].startswith(r"\sqrt"):
            after = i + len(r"\sqrt")
            # 跳过空白
            while after < len(s) and s[after] == " ":
                after += 1
            n_str = None
            # 可选 [n]
            if after < len(s) and s[after] == "[":
                close_bracket = s.find("]", after + 1)
                if close_bracket != -1:
                    n_str = s[after + 1 : close_bracket]
                    after = close_bracket + 1
            if after < len(s) and s[after] == "{":
                arg_content, after = _extract_brace_group(s, after)
                arg_expr = _latex_to_expr(arg_content)
                if n_str is not None:
                    result.append(f"({arg_expr})**(1/({n_str}))")
                else:
                    result.append(f"sqrt({arg_expr})")
                i = after
                continue
        # \left( \right) → ( )
        if s[i:].startswith(r"\left"):
            result.append("(")
            i += len(r"\left")
            # 跳过紧跟的 ( 或 [
            if i < len(s) and s[i] in "([|":
                i += 1
            continue
        if s[i:].startswith(r"\right"):
            result.append(")")
            i += len(r"\right")
            if i < len(s) and s[i] in ")]|":
                i += 1
            continue
        # 常用常量和符号
        if s[i:].startswith(r"\pi"):
            result.append("pi")
            i += 3
            continue
        if s[i:].startswith(r"\infty"):
            result.append("oo")
            i += 6
            continue
        if s[i:].startswith(r"\cdot") or s[i:].startswith(r"\times"):
            result.append("*")
            i += 6 if s[i:].startswith(r"\times") else 5
            continue
        if s[i:].startswith(r"\pm"):
            result.append("+")
            i += 3
            continue
        if s[i:].startswith(r"\mp"):
            result.append("-")
            i += 3
            continue
        # 跳过其他 \command（不识别的命令）
        if s[i] == "\\":
            j = i + 1
            while j < len(s) and s[j].isalpha():
                j += 1
            # 跳过该命令（包括后续的可能括号组）
            i = j
            continue
        # 花括号→圆括号（兜底）
        if s[i] == "{":
            result.append("(")
            i += 1
            continue
        if s[i] == "}":
            result.append(")")
            i += 1
            continue
        # 其他字符直接传递
        result.append(s[i])
        i += 1

    return "".join(result)


# ══════════════════════════════════════════════════════════════
# 规范化
# ══════════════════════════════════════════════════════════════

def _normalize(expr: str) -> Optional[str]:
    """将表达式规范化（去掉空白、美元符、千分位逗号等）。"""
    if expr is None:
        return None
    expr = expr.strip()
    expr = expr.replace("$", "")
    # 去掉千分位逗号（但保留小数点后的逗号情形不处理，仅去掉数字间的）
    expr = re.sub(r"(?<=\d),(?=\d)", "", expr)
    # \text{...} → 内容
    # 使用括号感知提取，避免 [^}]+ 的嵌套问题
    expr = _strip_text_commands(expr)
    # 百分号
    if expr.endswith("%"):
        try:
            return str(float(expr[:-1]) / 100)
        except Exception:
            pass
    return expr


def _strip_text_commands(s: str) -> str:
    """
    将 \\text{...} 替换为其内容（括号感知）。
    """
    result = []
    i = 0
    while i < len(s):
        if s[i:].startswith(r"\text"):
            after = i + len(r"\text")
            if after < len(s) and s[after] == "{":
                content, after = _extract_brace_group(s, after)
                result.append(content)
                i = after
                continue
        result.append(s[i])
        i += 1
    return "".join(result)


def _try_sympy_equal(a: str, b: str) -> bool:
    """
    用 sympy 判断两个 LaTeX 表达式是否相等。
    先用括号感知的 _latex_to_expr 转换，再调用 sympy 化简。
    """
    try:
        import sympy  # type: ignore[import]
        from sympy.parsing.sympy_parser import (  # type: ignore[import]
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )

        transformations = standard_transformations + (implicit_multiplication_application,)

        def _parse(latex_str: str):
            expr_str = _latex_to_expr(latex_str)
            return parse_expr(expr_str, transformations=transformations)

        diff = _parse(a) - _parse(b)
        return sympy.simplify(diff) == 0
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════

def is_correct(prediction: str, ground_truth: str) -> bool:
    """
    判断模型预测是否正确。
    先从 prediction 中提取 \\boxed{} / \\fbox{} 答案，再与 ground_truth 比较。
    比较顺序：字符串精确匹配 → 数值匹配 → sympy 符号等价。
    """
    extracted = extract_boxed_answer(prediction)
    if extracted is None:
        return False

    gt_norm = _normalize(ground_truth)
    pred_norm = _normalize(extracted)

    if gt_norm is None or pred_norm is None:
        return False

    # 1. 精确字符串匹配
    if gt_norm == pred_norm:
        return True

    # 2. 数值匹配（整数、浮点数）
    try:
        if abs(float(gt_norm) - float(pred_norm)) < 1e-6:
            return True
    except Exception:
        pass

    # 3. sympy 符号等价（括号感知转换后再判断）
    return _try_sympy_equal(gt_norm, pred_norm)


def compute_pass_at_k(correct_flags: list[bool], k: int) -> float:
    """
    给定一道题的 n 次采样结果（True/False 列表），计算 pass@k 无偏估计。
    公式：pass@k = 1 - C(n-c, k) / C(n, k)
    其中 c = 正确次数，n = 总次数。
    """
    n = len(correct_flags)
    c = sum(correct_flags)
    if n == 0:
        return 0.0
    if n < k:
        return float(c > 0)
    if n - c < k:
        return 1.0
    # Π_{i=0}^{k-1} (n-c-i)/(n-i)
    ratio = 1.0
    for i in range(k):
        ratio *= (n - c - i) / (n - i)
    return 1.0 - ratio
