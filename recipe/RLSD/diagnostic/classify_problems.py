"""
Step 3/3 诊断实验：汇总分析，生成最终诊断报告，输出 MRSD 训练数据集。

输入：
  - pass@k 完整结果（pass_at_k_results.jsonl）
  - Context A/B 测试结果（context_ab_results.jsonl）

输出：
  - diagnostic_report.txt   诊断报告
  - mrsd_train.jsonl        MRSD 正式训练集（Type-B 题目 + 错误轨迹 + 教师轨迹）
  - mrsd_train.parquet      同上，verl parquet 格式

用法：
    conda run -n verl python recipe/RLSD/diagnostic/classify_problems.py \
        --pass_at_k /data3/yyy/verl/data/rlsd/pass_at_k_results.jsonl \
        --context_ab /data3/yyy/verl/data/rlsd/diagnostic/context_ab_results.jsonl \
        --output_dir /data3/yyy/verl/data/rlsd/diagnostic
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pass_at_k",
        default="/data3/yyy/verl/data/rlsd/pass_at_k_results.jsonl",
    )
    p.add_argument(
        "--context_ab",
        default="/data3/yyy/verl/data/rlsd/diagnostic/context_ab_results.jsonl",
    )
    p.add_argument(
        "--output_dir",
        default="/data3/yyy/verl/data/rlsd/diagnostic",
    )
    p.add_argument(
        "--type_b_acc_threshold",
        type=float,
        default=0.0,
        help="Context B 正确率 > 此阈值才算 Type-B（默认 0 = 有一条正确即可）",
    )
    return p.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载数据 ──
    pak_results = load_jsonl(args.pass_at_k)
    ab_results = load_jsonl(args.context_ab)

    n_total_pak = len(pak_results)
    dead_zone = [r for r in pak_results if r["is_dead_zone"]]
    n_dead = len(dead_zone)

    n_ab = len(ab_results)
    type_b = [r for r in ab_results if r["problem_type"] == "Type-B"]
    type_a = [r for r in ab_results if r["problem_type"] == "Type-A"]

    # ── 构建报告 ──
    lines = []
    lines.append("=" * 60)
    lines.append("MRSD 诊断实验报告")
    lines.append("=" * 60)
    lines.append("")
    lines.append("── §3.1 Pass@K 采样结果 ──")
    lines.append(f"  数据集总题数:       {n_total_pak}")
    lines.append(f"  pass@64=0 题数:     {n_dead}  ({100*n_dead/n_total_pak:.1f}%)")

    pass1_all = [r["pass_at_1"] for r in pak_results]
    pass64_all = [r["pass_at_64"] for r in pak_results]
    lines.append(f"  pass@1  均值:       {sum(pass1_all)/len(pass1_all):.3f}")
    lines.append(f"  pass@64 均值:       {sum(pass64_all)/len(pass64_all):.3f}")
    lines.append("")

    # 按 topic 的死区率
    by_topic: dict[str, list] = defaultdict(list)
    for r in pak_results:
        by_topic[r.get("topic", "unknown")].append(r)
    lines.append("  Topic 死区率:")
    for topic, rs in sorted(by_topic.items()):
        dead = sum(1 for r in rs if r["is_dead_zone"])
        lines.append(f"    {topic:30s}: {dead:4d}/{len(rs):4d}  = {100*dead/len(rs):5.1f}%")

    lines.append("")
    lines.append("── §3.2 Context A/B 测试结果 ──")
    lines.append(f"  测试题目数:                 {n_ab}")
    acc_a_vals = [r["acc_a"] for r in ab_results]
    acc_b_vals = [r["acc_b"] for r in ab_results]
    lines.append(f"  Context A 平均正确率:       {sum(acc_a_vals)/len(acc_a_vals):.3f}")
    lines.append(f"  Context B 平均正确率:       {sum(acc_b_vals)/len(acc_b_vals):.3f}")
    lines.append(f"  B 优于 A 的题目数:          {sum(1 for a,b in zip(acc_a_vals, acc_b_vals) if b > a)}")
    lines.append(f"  Type-A（知识盲区）:         {len(type_a)} 道  ({100*len(type_a)/n_ab:.1f}%)")
    lines.append(f"  Type-B（搜索死区）:         {len(type_b)} 道  ({100*len(type_b)/n_ab:.1f}%)")
    lines.append("")

    lines.append("── 结论 ──")
    avg_b = sum(acc_b_vals) / len(acc_b_vals) if acc_b_vals else 0
    type_b_ratio = len(type_b) / n_ab if n_ab else 0
    if avg_b > 0.10 and type_b_ratio > 0.10:
        lines.append("  ✅ MRSD 假设成立：")
        lines.append(f"     Context B 平均正确率 = {avg_b:.1%} > 10%")
        lines.append(f"     搜索死区（Type-B）占比 = {type_b_ratio:.1%} > 10%")
        lines.append("     → 建议启动 MRSD 训练，使用 Type-B 题目作为训练集")
    else:
        lines.append("  ⚠️  MRSD 假设存疑：")
        if avg_b <= 0.10:
            lines.append(f"     Context B 正确率 = {avg_b:.1%} ≤ 10%（模型知识盲区过多）")
        if type_b_ratio <= 0.10:
            lines.append(f"     Type-B 占比 = {type_b_ratio:.1%} ≤ 10%（死区主要是知识盲区）")
        lines.append("     → 建议换更大模型（7B）或更换数据集")

    lines.append("")
    lines.append("=" * 60)

    report_str = "\n".join(lines)
    print(report_str)

    report_path = output_dir / "diagnostic_report.txt"
    with open(report_path, "w") as f:
        f.write(report_str)
    print(f"\n[classify] 报告已保存到: {report_path}")

    # ── 构建 MRSD 训练集（Type-B 题目） ──
    # 从全量 pass@k 结果中，对已测试并确认为 Type-B 的题目，
    # 使用 wrong_trajs + correct_teacher_trajs 构建训练样本
    from recipe.RLSD.rlsd.prompt import build_student_messages, build_teacher_context_b

    mrsd_samples = []
    for rec in type_b:
        question = rec["question"]
        gt = rec["ground_truth"]
        wrong_trajs = rec.get("wrong_trajs", [])
        correct_teacher_trajs = rec.get("correct_teacher_trajs", [])

        if not wrong_trajs or not correct_teacher_trajs:
            continue

        # 每条错误轨迹 × 每条正确教师轨迹 → 一个训练样本
        for wrong_traj in wrong_trajs[:4]:
            for teacher_traj in correct_teacher_trajs[:2]:
                mrsd_samples.append(
                    {
                        "index": rec["index"],
                        "question": question,
                        "ground_truth": gt,
                        "difficulty": rec.get("difficulty", -1),
                        "topic": rec.get("topic", ""),
                        "problem_type": "Type-B",
                        # 训练所需字段
                        "student_prompt": build_student_messages(question),
                        "teacher_context": build_teacher_context_b(question, wrong_traj, gt),
                        "teacher_response": teacher_traj,
                        "wrong_traj": wrong_traj,
                    }
                )

    mrsd_jsonl = output_dir / "mrsd_train.jsonl"
    with open(mrsd_jsonl, "w") as f:
        for s in mrsd_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # parquet 格式（供 trainer 直接读取）
    if mrsd_samples:
        mrsd_df = pd.DataFrame(mrsd_samples)
        mrsd_parquet = output_dir / "mrsd_train.parquet"
        mrsd_df.to_parquet(str(mrsd_parquet), index=False)
        print(f"[classify] MRSD 训练集: {len(mrsd_samples)} 条样本 → {mrsd_parquet}")
    else:
        print("[classify] ⚠️  Type-B 题目中没有足够的训练样本（需要同时有错误轨迹和正确教师轨迹）")

    print("[classify] 完成")


if __name__ == "__main__":
    main()
