"""
数学答案验证器。

使用 HuggingFace math-verify 库做数学等价性判断，
支持 LaTeX / 纯表达式 / 元组 / 集合 / 分数等多种格式的鲁棒比较。
同时保留 extract_boxed_answer 用于日志展示。
"""

from typing import Optional

from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

_GOLD_CONFIGS = [LatexExtractionConfig(), ExprExtractionConfig()]
_PRED_CONFIGS_STRICT = [LatexExtractionConfig(boxed_match_priority=0)]
_PRED_CONFIGS_FALLBACK = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]


# ══════════════════════════════════════════════════════════════
# 括号感知工具函数（用于 extract_boxed_answer 展示用途）
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
# 提取 \\boxed{} / \\fbox{} 内容（仅用于日志展示）
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
    """从模型生成文本中提取最终 boxed 答案字符串（用于日志展示）。"""
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
    使用 math_verify 库：从 prediction 中提取答案并与 ground_truth 做数学等价性比较。

    提取策略（两级）：
      1. 仅用 LatexExtractionConfig（从 \\boxed{} / LaTeX 环境提取），避免从正文误匹配数字
      2. 若第 1 级未提取到内容，回退到 LatexExtraction + ExprExtraction
    """
    try:
        gold_parsed = parse(_wrap_latex(ground_truth), extraction_config=_GOLD_CONFIGS)
        if not gold_parsed:
            return False

        pred_parsed = parse(prediction, extraction_config=_PRED_CONFIGS_STRICT)
        if pred_parsed:
            return verify(gold_parsed, pred_parsed)

        pred_parsed = parse(prediction, extraction_config=_PRED_CONFIGS_FALLBACK)
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
