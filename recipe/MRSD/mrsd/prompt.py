"""
MRSD 的 prompt 模板。

包含：
  - Student prompt（仅问题）
  - Teacher context A：OPSD 风格（问题 + 正确答案）
  - Teacher context B：MRSD 风格（问题 + 错误轨迹 + 正确答案）
"""

# ──────────────────────────────────────────────────────────────
# 系统提示
# ──────────────────────────────────────────────────────────────

SYSTEM_STUDENT = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step and put your final answer within \\boxed{}."
)

SYSTEM_TEACHER = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step and put your final answer within \\boxed{}."
)

# ──────────────────────────────────────────────────────────────
# 模板函数
# ──────────────────────────────────────────────────────────────

MAX_WRONG_TRAJ_TOKENS = 1024  # §6.3：错误轨迹最多截断到此 token 数（字符数估算约 4x）
MAX_WRONG_TRAJ_CHARS = MAX_WRONG_TRAJ_TOKENS * 4  # 粗略估算


def build_student_messages(question: str) -> list[dict]:
    """Student 的输入：仅包含问题，不含任何 hint。"""
    return [
        {"role": "system", "content": SYSTEM_STUDENT},
        {"role": "user", "content": f"Problem: {question}"},
    ]


def _truncate_wrong_traj(wrong_traj: str, max_chars: int = MAX_WRONG_TRAJ_CHARS) -> str:
    """
    截断过长的错误轨迹（§6.3）。
    策略：保留前 512 tokens + 后 512 tokens（字符估算）。
    """
    if len(wrong_traj) <= max_chars:
        return wrong_traj
    half = max_chars // 2
    return wrong_traj[:half] + "\n...[truncated]...\n" + wrong_traj[-half:]


def build_teacher_context_a(question: str, correct_answer: str) -> list[dict]:
    """
    Context A（OPSD 风格）：问题 + 正确答案。
    用于对照实验（§3.1 Context A）。
    """
    user_content = (
        f"Problem: {question}\n\n"
        f"I was told the correct answer is: {correct_answer}\n\n"
        "Now let me carefully work out a correct step-by-step solution:"
    )
    return [
        {"role": "system", "content": SYSTEM_TEACHER},
        {"role": "user", "content": user_content},
    ]


def build_teacher_context_b(
    question: str,
    wrong_traj: str,
    correct_answer: str,
) -> list[dict]:
    """
    Context B（MRSD 风格，§2.4）：问题 + 错误轨迹 + 正确答案。
    这是 MRSD 的核心 teacher context。
    """
    wrong_traj_truncated = _truncate_wrong_traj(wrong_traj)
    user_content = (
        f"Problem: {question}\n\n"
        f"My previous attempt (which was incorrect):\n{wrong_traj_truncated}\n\n"
        f"I was told the correct answer is: {correct_answer}\n\n"
        "Now let me carefully reconsider the problem and provide a correct step-by-step solution:"
    )
    return [
        {"role": "system", "content": SYSTEM_TEACHER},
        {"role": "user", "content": user_content},
    ]
