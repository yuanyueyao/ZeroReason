"""
测试 prompt.py 中 student / teacher 模板的输出是否符合预期。

测试项：
  1. build_student_messages：结构、角色、内容
  2. build_teacher_context_a：OPSD 风格（问题 + 正确答案）
  3. build_teacher_context_b：MRSD 风格（问题 + 错误轨迹 + 正确答案）
  4. _truncate_wrong_traj：截断逻辑（短 / 长 / 边界）
  5. 与 tokenizer chat_template 的集成（apply_chat_template 后是否可解析）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from recipe.MRSD.mrsd.prompt import (
    SYSTEM_STUDENT,
    SYSTEM_TEACHER,
    MAX_WRONG_TRAJ_CHARS,
    build_student_messages,
    build_teacher_context_a,
    build_teacher_context_b,
    _truncate_wrong_traj,
)

# ═══════════════════════════════════════════════════════════
# 测试数据
# ═══════════════════════════════════════════════════════════

SAMPLE_QUESTION = (
    "Let $f(x) = x^3 - 3x + 1$. Find all real roots of $f(x) = 0$."
)
SAMPLE_ANSWER = "x = 2\\cos(20°),\\; 2\\cos(140°),\\; 2\\cos(260°)"
SAMPLE_WRONG_TRAJ = (
    "Step 1: I'll try to factor $x^3 - 3x + 1$.\n"
    "Step 2: Testing $x=1$: $1 - 3 + 1 = -1 \\neq 0$. Not a root.\n"
    "Step 3: Testing $x=-1$: $-1 + 3 + 1 = 3 \\neq 0$. Not a root.\n"
    "Step 4: Since there's no rational root, I'll use the quadratic formula... "
    "wait, this is cubic. Let me try $x = 3$: $27 - 9 + 1 = 19$. Nope.\n"
    "Step 5: I'm stuck, so let me guess the answer is $\\boxed{1}$."
)

SEPARATOR = "─" * 60


def pprint_messages(messages: list[dict], label: str):
    """漂亮地打印 chat messages。"""
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(SEPARATOR)
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[{role}]")
        print(content)
    print()


# ═══════════════════════════════════════════════════════════
# Test 1: build_student_messages
# ═══════════════════════════════════════════════════════════

def test_student_messages():
    print("=" * 60)
    print("TEST 1: build_student_messages")
    print("=" * 60)

    msgs = build_student_messages(SAMPLE_QUESTION)
    pprint_messages(msgs, "Student Prompt")

    assert len(msgs) == 2, f"期望 2 条消息，实际 {len(msgs)}"
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[0]["content"] == SYSTEM_STUDENT
    assert "\\boxed{}" in msgs[0]["content"], "system prompt 应包含 \\boxed{}"
    assert msgs[1]["content"] == f"Problem: {SAMPLE_QUESTION}"
    assert SAMPLE_ANSWER not in msgs[1]["content"], "student prompt 不应泄漏答案"

    print("✓ 结构正确：2 条消息 (system + user)")
    print("✓ system prompt 包含 \\boxed{} 格式要求")
    print("✓ user 内容仅包含问题，无答案泄漏")


# ═══════════════════════════════════════════════════════════
# Test 2: build_teacher_context_a (OPSD 风格)
# ═══════════════════════════════════════════════════════════

def test_teacher_context_a():
    print("\n" + "=" * 60)
    print("TEST 2: build_teacher_context_a (OPSD)")
    print("=" * 60)

    msgs = build_teacher_context_a(SAMPLE_QUESTION, SAMPLE_ANSWER)
    pprint_messages(msgs, "Teacher Context A (OPSD)")

    assert len(msgs) == 2, f"期望 2 条消息，实际 {len(msgs)}"
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[0]["content"] == SYSTEM_TEACHER

    user_content = msgs[1]["content"]
    assert f"Problem: {SAMPLE_QUESTION}" in user_content, "应包含原始问题"
    assert SAMPLE_ANSWER in user_content, "应包含正确答案"
    assert "correct answer is:" in user_content, "应有正确答案的引导词"
    assert "step-by-step" in user_content, "应引导逐步求解"
    assert "incorrect" not in user_content.lower(), "Context A 不应包含'错误'相关词汇"

    print("✓ 结构正确：2 条消息 (system + user)")
    print("✓ 包含原始问题")
    print("✓ 包含正确答案")
    print("✓ 不含错误轨迹（与 Context B 区分）")


# ═══════════════════════════════════════════════════════════
# Test 3: build_teacher_context_b (MRSD 核心)
# ═══════════════════════════════════════════════════════════

def test_teacher_context_b():
    print("\n" + "=" * 60)
    print("TEST 3: build_teacher_context_b (MRSD)")
    print("=" * 60)

    msgs = build_teacher_context_b(SAMPLE_QUESTION, SAMPLE_WRONG_TRAJ, SAMPLE_ANSWER)
    pprint_messages(msgs, "Teacher Context B (MRSD)")

    assert len(msgs) == 2, f"期望 2 条消息，实际 {len(msgs)}"
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[0]["content"] == SYSTEM_TEACHER

    user_content = msgs[1]["content"]
    assert f"Problem: {SAMPLE_QUESTION}" in user_content, "应包含原始问题"
    assert SAMPLE_ANSWER in user_content, "应包含正确答案"
    assert "incorrect" in user_content.lower(), "应标注之前的尝试是错误的"
    assert "reconsider" in user_content.lower(), "应引导重新思考"

    assert "I'll try to factor" in user_content, "应包含错误轨迹内容"
    assert "I'm stuck" in user_content, "应包含错误轨迹内容"

    sections = user_content.split("\n\n")
    problem_idx = next(i for i, s in enumerate(sections) if s.startswith("Problem:"))
    traj_idx = next(i for i, s in enumerate(sections) if "previous attempt" in s.lower())
    answer_idx = next(i for i, s in enumerate(sections) if "correct answer" in s.lower())

    assert problem_idx < traj_idx < answer_idx, \
        f"顺序应为：问题({problem_idx}) < 错误轨迹({traj_idx}) < 正确答案({answer_idx})"

    print("✓ 结构正确：2 条消息 (system + user)")
    print("✓ 包含原始问题")
    print("✓ 包含错误轨迹（且标注为 incorrect）")
    print("✓ 包含正确答案")
    print("✓ 内容顺序正确：问题 → 错误轨迹 → 正确答案 → 引导语")


# ═══════════════════════════════════════════════════════════
# Test 4: _truncate_wrong_traj 截断逻辑
# ═══════════════════════════════════════════════════════════

def test_truncation():
    print("\n" + "=" * 60)
    print("TEST 4: _truncate_wrong_traj")
    print("=" * 60)

    # 4a: 短文本不截断
    short = "This is a short trajectory."
    result = _truncate_wrong_traj(short)
    assert result == short, "短文本不应被截断"
    print(f"✓ 短文本 ({len(short)} chars) 未截断")

    # 4b: 恰好等于上限，不截断
    exact = "x" * MAX_WRONG_TRAJ_CHARS
    result = _truncate_wrong_traj(exact)
    assert result == exact, "恰好等于上限时不应截断"
    print(f"✓ 边界文本 ({len(exact)} chars = MAX) 未截断")

    # 4c: 超长文本截断
    long_text = "A" * 1000 + "MIDDLE" + "Z" * (MAX_WRONG_TRAJ_CHARS + 1000)
    result = _truncate_wrong_traj(long_text)

    assert len(result) < len(long_text), "超长文本应被截断"
    assert "...[truncated]..." in result, "截断标记应存在"
    assert result.startswith("A"), "应保留开头内容"
    assert result.endswith("Z"), "应保留结尾内容"

    half = MAX_WRONG_TRAJ_CHARS // 2
    head = result.split("\n...[truncated]...\n")[0]
    tail = result.split("\n...[truncated]...\n")[1]
    assert len(head) == half, f"头部长度应为 {half}，实际 {len(head)}"
    assert len(tail) == half, f"尾部长度应为 {half}，实际 {len(tail)}"

    print(f"✓ 超长文本 ({len(long_text)} chars) 被截断到 ~{len(result)} chars")
    print(f"  头部 {len(head)} chars + 截断标记 + 尾部 {len(tail)} chars")

    # 4d: 截断后传入 context_b 仍能正常工作
    msgs = build_teacher_context_b("Q?", long_text, "42")
    user_content = msgs[1]["content"]
    assert "...[truncated]..." in user_content, "截断标记应出现在最终 prompt 中"
    print("✓ 截断后的轨迹在 build_teacher_context_b 中正常工作")


# ═══════════════════════════════════════════════════════════
# Test 5: 与 tokenizer chat_template 集成测试
# ═══════════════════════════════════════════════════════════

def test_with_tokenizer():
    print("\n" + "=" * 60)
    print("TEST 5: tokenizer chat_template 集成")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("⚠ transformers 未安装，跳过 tokenizer 集成测试")
        return

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"加载 tokenizer: {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    for label, msgs_fn in [
        ("Student", lambda: build_student_messages(SAMPLE_QUESTION)),
        ("Teacher-A", lambda: build_teacher_context_a(SAMPLE_QUESTION, SAMPLE_ANSWER)),
        ("Teacher-B", lambda: build_teacher_context_b(SAMPLE_QUESTION, SAMPLE_WRONG_TRAJ, SAMPLE_ANSWER)),
    ]:
        msgs = msgs_fn()
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        token_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

        print(f"\n--- {label} ---")
        print(f"  消息数: {len(msgs)}")
        print(f"  渲染后字符数: {len(text)}")
        print(f"  Token 数: {len(token_ids)}")
        print(f"  渲染后文本预览 (前 300 chars):")
        print(f"    {text[:300]}...")

        assert len(text) > 0, f"{label}: 渲染后文本不应为空"
        assert len(token_ids) > 0, f"{label}: token 数不应为 0"

        decoded = tok.decode(token_ids, skip_special_tokens=False)
        assert len(decoded) > 0, f"{label}: decode 后不应为空"

    print("\n✓ 所有 prompt 模板与 tokenizer chat_template 兼容")


# ═══════════════════════════════════════════════════════════
# Test 6: Context A vs Context B 对比
# ═══════════════════════════════════════════════════════════

def test_context_a_vs_b_diff():
    print("\n" + "=" * 60)
    print("TEST 6: Context A vs Context B 对比")
    print("=" * 60)

    msgs_a = build_teacher_context_a(SAMPLE_QUESTION, SAMPLE_ANSWER)
    msgs_b = build_teacher_context_b(SAMPLE_QUESTION, SAMPLE_WRONG_TRAJ, SAMPLE_ANSWER)

    content_a = msgs_a[1]["content"]
    content_b = msgs_b[1]["content"]

    assert msgs_a[0] == msgs_b[0], "system prompt 应相同"
    print("✓ system prompt 相同")

    assert len(content_b) > len(content_a), \
        f"Context B ({len(content_b)} chars) 应比 Context A ({len(content_a)} chars) 长"
    print(f"✓ Context B ({len(content_b)} chars) > Context A ({len(content_a)} chars)")

    assert "previous attempt" not in content_a, "Context A 不应包含错误轨迹"
    assert "previous attempt" in content_b, "Context B 应包含错误轨迹"
    print("✓ 只有 Context B 包含错误轨迹引用")

    assert "reconsider" not in content_a.lower(), "Context A 不应有 reconsider"
    assert "reconsider" in content_b.lower(), "Context B 应有 reconsider"
    print("✓ 引导语区分：A='work out' vs B='reconsider'")

    print(f"\n  Context A 长度: {len(content_a)} chars")
    print(f"  Context B 长度: {len(content_b)} chars")
    print(f"  差值（≈错误轨迹+引导语）: {len(content_b) - len(content_a)} chars")


# ═══════════════════════════════════════════════════════════
# 运行所有测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_student_messages()
    test_teacher_context_a()
    test_teacher_context_b()
    test_truncation()
    test_with_tokenizer()
    test_context_a_vs_b_diff()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
