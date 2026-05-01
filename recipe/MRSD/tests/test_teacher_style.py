"""
对比 Student（无 hint）与 Teacher Context A/B 的输出风格。

目的：验证 teacher 在得到正确答案提示后，生成的解题过程是否
与 student 直接生成的风格一致（格式、口吻、推理方式）。
如果风格差异大，OPSD 蒸馏效果会打折扣。

用法：
  python recipe/MRSD/tests/test_teacher_style.py [--n_questions 3] [--max_tokens 8192]

输出保存到：logs/mrsd/test_teacher_style_<timestamp>.txt
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from recipe.MRSD.mrsd.prompt import (
    build_student_messages,
    build_teacher_context_a,
    build_teacher_context_b,
)

MODEL_PATH = "/data3/yyy/models/Qwen3-4B-Instruct-2507"
DATA_PATH = "/data3/yyy/verl/data/mrsd/pass_at_k_pass1_resp8192_20260501_095948_dead_zone.jsonl"

SEP = "=" * 72
SUBSEP = "─" * 72


class TeeWriter:
    """同时写到文件和 stdout。"""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def load_questions(path: str, n: int):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


@torch.no_grad()
def generate(model, tokenizer, messages_list: list[list[dict]], max_new_tokens: int, temperature: float) -> list[str]:
    """逐条生成（避免 padding 对小规模测试的干扰）。"""
    results = []
    for i, msgs in enumerate(messages_list):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        n_prompt_tokens = inputs["input_ids"].shape[1]

        t0 = time.time()
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
        )
        elapsed = time.time() - t0
        new_tokens = out[0][n_prompt_tokens:]
        n_gen = len(new_tokens)
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"  生成 [{i+1}/{len(messages_list)}]: "
              f"prompt={n_prompt_tokens} tokens, "
              f"gen={n_gen} tokens, "
              f"{elapsed:.1f}s ({n_gen/elapsed:.1f} tok/s)",
              flush=True)
        results.append(resp)
    return results


def analyze_style(text: str) -> dict:
    """提取风格指标。"""
    lower = text.lower()
    return {
        "len_chars": len(text),
        "has_boxed": "\\boxed{" in text or "\\boxed " in text,
        "has_think": "<think>" in lower,
        "n_steps": lower.count("step "),
        "n_newlines": text.count("\n"),
        "first_person_I": lower[:800].count(" i "),
        "has_therefore": "therefore" in lower or "thus" in lower or "hence" in lower,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_questions", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    log_dir = Path("/data3/yyy/verl/logs/mrsd")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = log_dir / f"test_teacher_style_{ts}.txt"

    tee = TeeWriter(str(out_path))
    sys.stdout = tee

    print(f"模型: {MODEL_PATH}")
    print(f"数据: {DATA_PATH}")
    print(f"参数: n_questions={args.n_questions}, max_tokens={args.max_tokens}, temperature={args.temperature}")
    print(f"输出: {out_path}")
    print()

    print("加载模型...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("模型加载完成\n")

    questions = load_questions(DATA_PATH, args.n_questions)
    print(f"加载了 {len(questions)} 道死区题目\n")

    for qi, q in enumerate(questions):
        question = q["question"]
        gt = q["ground_truth"]
        wrong_traj = q.get("first_wrong_traj", "")

        student_msgs = build_student_messages(question)
        teacher_a_msgs = build_teacher_context_a(question, gt)
        teacher_b_msgs = build_teacher_context_b(question, wrong_traj, gt) if wrong_traj else None

        all_msgs = [student_msgs, teacher_a_msgs]
        labels = ["Student", "Teacher-A"]
        if teacher_b_msgs:
            all_msgs.append(teacher_b_msgs)
            labels.append("Teacher-B")

        print(SEP)
        print(f"  题目 #{qi} (index={q['index']}, difficulty={q['difficulty']})")
        print(SEP)
        print(f"Question:\n{question}\n")
        print(f"Ground Truth: {gt}")
        print()

        responses = generate(model, tokenizer, all_msgs, args.max_tokens, args.temperature)
        print()

        styles = {}
        for label, resp in zip(labels, responses):
            styles[label] = analyze_style(resp)

            print(SUBSEP)
            print(f"  [{label}] (len={len(resp)} chars)")
            print(SUBSEP)
            print(resp)
            print()

        # 风格对比表
        print(SUBSEP)
        print("  风格对比")
        print(SUBSEP)
        header = f"{'指标':<20}" + "".join(f"{lb:>14}" for lb in labels)
        print(header)
        print("─" * len(header))

        metrics = ["len_chars", "has_boxed", "has_think", "n_steps", "n_newlines", "first_person_I", "has_therefore"]
        metric_names = ["长度(chars)", "有\\boxed", "有<think>", "Step提及数", "换行数", "第一人称I(前800)", "有therefore/thus"]
        for name, key in zip(metric_names, metrics):
            row = f"{name:<20}"
            for lb in labels:
                v = styles[lb][key]
                if isinstance(v, bool):
                    row += f"{'✓' if v else '✗':>14}"
                else:
                    row += f"{v:>14}"
            print(row)

        s_style = styles["Student"]
        a_style = styles["Teacher-A"]
        warnings = []
        if s_style["has_boxed"] != a_style["has_boxed"]:
            warnings.append("⚠ \\boxed 使用不一致")
        if s_style["has_think"] != a_style["has_think"]:
            warnings.append("⚠ <think> 标签使用不一致")
        if abs(s_style["len_chars"] - a_style["len_chars"]) / max(s_style["len_chars"], 1) > 0.5:
            warnings.append(f"⚠ 长度差异较大 (Student={s_style['len_chars']}, Teacher-A={a_style['len_chars']})")

        if warnings:
            print("\n  ⚠ 风格差异警告:")
            for w in warnings:
                print(f"    {w}")
        else:
            print("\n  ✓ Student 与 Teacher-A 风格基本一致")
        print()

    print(SEP)
    print("  测试完成")
    print(SEP)
    print(f"\n完整输出已保存到: {out_path}")

    sys.stdout = tee.stdout
    tee.close()
    print(f"\n完整输出已保存到: {out_path}")


if __name__ == "__main__":
    main()
