"""
Step 2/3 诊断实验：对 pass@K=0 的死区题目，测试 Context A vs Context B 的修正成功率。

输入：run_pass_at_k.py 生成的 dead_zone_problems.jsonl
输出：context_ab_results.jsonl + 分类报告

用法：
    conda run -n verl python recipe/MRSD/diagnostic/run_context_ab_test.py \
        --dead_zone /data3/yyy/verl/data/mrsd/dead_zone_problems.jsonl \
        --model /data3/yyy/models/Qwen3-4B-Instruct-2507 \
        --output_dir /data3/yyy/verl/data/mrsd/diagnostic \
        --n_problems 50 \
        --n_samples_per_context 16
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

# ── 添加项目根目录到路径 ──
_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dead_zone",
        default="/data3/yyy/verl/data/mrsd/dead_zone_problems.jsonl",
        help="run_pass_at_k.py 输出的死区题目 jsonl",
    )
    p.add_argument(
        "--model",
        default="/data3/yyy/models/Qwen3-4B-Instruct-2507",
    )
    p.add_argument(
        "--output_dir",
        default="/data3/yyy/verl/data/mrsd/diagnostic",
    )
    p.add_argument("--n_problems", type=int, default=50, help="随机抽取多少道死区题目")
    p.add_argument("--n_samples_per_context", type=int, default=16, help="每种 context 采样次数")
    p.add_argument("--n_gpus", type=int, default=8)
    p.add_argument("--max_new_tokens_a", type=int, default=2048, help="Context A 最大生成 tokens")
    p.add_argument("--max_new_tokens_b", type=int, default=3072, help="Context B 最大生成 tokens")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    from recipe.MRSD.mrsd.verifier import is_correct
    from recipe.MRSD.mrsd.prompt import (
        build_teacher_context_a,
        build_teacher_context_b,
    )

    # ── 加载死区题目 ──
    dead_zone = []
    with open(args.dead_zone) as f:
        for line in f:
            try:
                dead_zone.append(json.loads(line))
            except Exception:
                pass
    print(f"[context_ab] 加载死区题目: {len(dead_zone)} 道")

    if len(dead_zone) == 0:
        print("[context_ab] 没有死区题目，退出")
        return

    # 随机抽样
    sample = random.sample(dead_zone, min(args.n_problems, len(dead_zone)))
    print(f"[context_ab] 抽取 {len(sample)} 道题进行 Context A/B 测试")

    # ── 初始化 vllm ──
    print(f"[context_ab] 加载模型: {args.model}")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.n_gpus,
        gpu_memory_utilization=0.85,
        max_model_len=6144,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params_a = SamplingParams(
        n=args.n_samples_per_context,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens_a,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    sampling_params_b = SamplingParams(
        n=args.n_samples_per_context,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens_b,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # ── 构建 Context A/B prompts ──
    prompts_a, prompts_b = [], []
    for rec in sample:
        question = rec["question"]
        gt = rec["ground_truth"]
        # Context A：OPSD 风格
        msgs_a = build_teacher_context_a(question, gt)
        prompts_a.append(
            tokenizer.apply_chat_template(msgs_a, tokenize=False, add_generation_prompt=True)
        )
        # Context B：MRSD 风格，使用 pass@k 阶段保存的第一条错误轨迹
        wrong_traj = rec.get("first_wrong_traj", "")
        if not wrong_traj and rec.get("wrong_trajs"):
            wrong_traj = rec["wrong_trajs"][0]
        msgs_b = build_teacher_context_b(question, wrong_traj or "[No previous attempt available]", gt)
        prompts_b.append(
            tokenizer.apply_chat_template(msgs_b, tokenize=False, add_generation_prompt=True)
        )

    # ── 运行推理 ──
    print(f"[context_ab] 运行 Context A ({len(prompts_a)} prompts × {args.n_samples_per_context} samples)...")
    t0 = time.time()
    outputs_a = llm.generate(prompts_a, sampling_params_a)
    print(f"[context_ab] Context A 完成，耗时 {time.time()-t0:.1f}s")

    print(f"[context_ab] 运行 Context B ({len(prompts_b)} prompts × {args.n_samples_per_context} samples)...")
    t0 = time.time()
    outputs_b = llm.generate(prompts_b, sampling_params_b)
    print(f"[context_ab] Context B 完成，耗时 {time.time()-t0:.1f}s")

    # ── 验证并统计 ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "context_ab_results.jsonl"
    type_b_path = output_dir / "type_b_problems.jsonl"  # 搜索死区（MRSD 训练集）
    type_a_path = output_dir / "type_a_problems.jsonl"  # 知识盲区

    n_type_a = 0
    n_type_b = 0
    acc_a_total = 0.0
    acc_b_total = 0.0

    with open(results_path, "w") as out_f, \
         open(type_b_path, "w") as tb_f, \
         open(type_a_path, "w") as ta_f:

        for rec, out_a, out_b in zip(sample, outputs_a, outputs_b):
            gt = rec["ground_truth"]
            question = rec["question"]

            resps_a = [o.text for o in out_a.outputs]
            resps_b = [o.text for o in out_b.outputs]

            correct_a = [is_correct(r, gt) for r in resps_a]
            correct_b = [is_correct(r, gt) for r in resps_b]

            acc_a = sum(correct_a) / len(correct_a) if correct_a else 0.0
            acc_b = sum(correct_b) / len(correct_b) if correct_b else 0.0
            acc_a_total += acc_a
            acc_b_total += acc_b

            # §3.2 分层标准：conditioned 后 pass@16 > 0 → Type-B（搜索死区）
            is_type_b = sum(correct_b) > 0

            result = {
                "index": rec["index"],
                "question": question,
                "ground_truth": gt,
                "difficulty": rec.get("difficulty", -1),
                "topic": rec.get("topic", ""),
                # pass@k 阶段结果
                "original_n_correct": rec.get("n_correct", 0),
                "original_n_samples": rec.get("n_samples", 0),
                # Context A
                "acc_a": round(acc_a, 4),
                "n_correct_a": sum(correct_a),
                "n_samples_a": len(correct_a),
                # Context B
                "acc_b": round(acc_b, 4),
                "n_correct_b": sum(correct_b),
                "n_samples_b": len(correct_b),
                # 分类
                "problem_type": "Type-B" if is_type_b else "Type-A",
                # 保存 Context B 生成的正确轨迹（用于 MRSD 训练）
                "correct_teacher_trajs": [
                    r for r, c in zip(resps_b, correct_b) if c
                ][:4],
                "wrong_trajs": rec.get("wrong_trajs", []),
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if is_type_b:
                n_type_b += 1
                tb_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                n_type_a += 1
                ta_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    n_total = len(sample)
    print("\n[context_ab] === 诊断结果 ===")
    print(f"  测试题目数:         {n_total}")
    print(f"  Context A 平均正确率: {acc_a_total/n_total:.3f}")
    print(f"  Context B 平均正确率: {acc_b_total/n_total:.3f}")
    print(f"  B 优于 A 的题目数:    {sum(1 for r in open(results_path) for d in [json.loads(r)] if d['acc_b'] > d['acc_a'])}")
    print()
    print(f"  Type-A（知识盲区）: {n_type_a} 道  ({100*n_type_a/n_total:.1f}%)")
    print(f"  Type-B（搜索死区）: {n_type_b} 道  ({100*n_type_b/n_total:.1f}%)")
    print()
    print("  判断：", end="")
    if acc_b_total / n_total > 0.10 and n_type_b > n_total * 0.1:
        print("✅ Context B 正确率 > 10%，MRSD 假设成立，方法可行！")
    else:
        print("⚠️  Context B 正确率偏低，请考虑更换更大模型或更难的数据集。")

    print(f"\n[context_ab] 结果文件:")
    print(f"  完整结果:   {results_path}")
    print(f"  Type-B 题目: {type_b_path}  （MRSD 训练集）")
    print(f"  Type-A 题目: {type_a_path}  （知识盲区，跳过）")


if __name__ == "__main__":
    main()
