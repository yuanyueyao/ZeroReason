"""
数学答案验证器。

使用 HuggingFace math-verify 库：仅将模型回复中最后一个 \\boxed{}（或 \\fbox{}）
内的字符串与 ground_truth 做数学等价比较；全文其它位置的 LaTeX/数字不参与判对。
extract_boxed_answer 与判题使用同一套抽取规则，便于 eval 日志与 is_correct 一致。
"""

from typing import Optional

from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

_GOLD_CONFIGS = [LatexExtractionConfig(), ExprExtractionConfig()]


# ══════════════════════════════════════════════════════════════
# 括号感知工具函数（extract_boxed_answer / is_correct 共用）
# ══════════════════════════════════════════════════════════════

def _find_matching_brace(s: str, open_pos: int) -> int:
    assert s[open_pos] == "{", f"Expected '{{' at pos {open_pos}, got {s[open_pos]!r}"
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _extract_brace_group(s: str, pos: int) -> tuple[str, int]:
    if pos >= len(s) or s[pos] != "{":
        return "", pos
    close = _find_matching_brace(s, pos)
    if close == -1:
        return s[pos + 1 :], len(s)
    return s[pos + 1 : close], close + 1


# ══════════════════════════════════════════════════════════════
# 提取 \\boxed{} / \\fbox{} 内容（与 is_correct 共用，亦可用于日志）
# ══════════════════════════════════════════════════════════════

def last_boxed_only_string(string: str) -> Optional[str]:
    for marker in (r"\boxed", r"\fbox"):
        idx = string.rfind(marker)
        if idx < 0:
            continue
        brace_start = string.find("{", idx + len(marker))
        if brace_start == -1:
            continue
        brace_end = _find_matching_brace(string, brace_start)
        if brace_end == -1:
            continue
        return string[idx : brace_end + 1]
    return None


def remove_boxed(s: str) -> str:
    for prefix in (r"\boxed{", r"\fbox{"):
        if s.startswith(prefix):
            inner = s[len(prefix):]
            if inner.endswith("}"):
                return inner[:-1]
            return inner
    for prefix in (r"\boxed", r"\fbox"):
        if s.startswith(prefix):
            brace_start = s.find("{", len(prefix))
            if brace_start != -1:
                content, _ = _extract_brace_group(s, brace_start)
                return content
    return s


def extract_boxed_answer(solution: str) -> Optional[str]:
    """从模型回复中取最后一个 \\boxed{} / \\fbox{} 内的纯内容（无外层 \\boxed）。"""
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return None
    return remove_boxed(boxed)


# ══════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════

def _wrap_latex(s: str) -> str:
    """将 ground truth 包裹在 $...$ 中以确保 LatexExtractionConfig 能解析。"""
    s = s.strip()
    if not s:
        return s
    if s.startswith("$") or s.startswith("\\(") or s.startswith("\\["):
        return s
    return f"${s}$"


def is_correct(prediction: str, ground_truth: str) -> bool:
    """
    判断模型预测是否正确。

    仅从回复全文中的**最后一个** \\boxed{} / \\fbox{} 内取出答案串，再与 ground_truth
    做 math_verify 等价比较；无 boxed、内容为空或解析失败均视为错误。
    """
    boxed_inner = extract_boxed_answer(prediction)
    if boxed_inner is None:
        return False
    boxed_inner = boxed_inner.strip()
    if not boxed_inner:
        return False
    try:
        gold_parsed = parse(_wrap_latex(ground_truth), extraction_config=_GOLD_CONFIGS)
        if not gold_parsed:
            return False

        pred_parsed = parse(_wrap_latex(boxed_inner), extraction_config=_GOLD_CONFIGS)
        if not pred_parsed:
            return False
        return verify(gold_parsed, pred_parsed)
    except Exception:
        return False


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
