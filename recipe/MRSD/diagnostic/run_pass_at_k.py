"""
Step 1/3 诊断实验：对数据集做 pass@k 采样，找出 pass@K=0 的死区题目。

用法：
    conda run -n verl python recipe/MRSD/diagnostic/run_pass_at_k.py \
        --data /data3/yyy/verl/data/mrsd/train_level45.parquet \
        --model /data3/yyy/models/Qwen3-4B-Instruct-2507 \
        --output /data3/yyy/verl/data/mrsd/pass_at_k_results.jsonl \
        --n_samples 64 \
        --n_gpus 8 \
        --batch_size 256
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# 禁用 torch.compile（系统 PATH 里默认 nvcc 版本与 torch cu128 不匹配）
# 请在 shell 里预先 export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default="/data3/yyy/verl/data/mrsd/train_level45.parquet",
        help="输入 parquet，必须包含 prompt 和 reward_model 字段",
    )
    p.add_argument(
        "--model",
        default="/data3/yyy/models/Qwen3-4B-Instruct-2507",
        help="模型路径",
    )
    p.add_argument(
        "--output",
        default="/data3/yyy/verl/data/mrsd/pass_at_k_results.jsonl",
        help="输出 jsonl 路径",
    )
    p.add_argument("--n_samples", type=int, default=64, help="每道题采样次数")
    p.add_argument("--n_gpus", type=int, default=8, help="使用 GPU 数量")
    p.add_argument("--batch_size", type=int, default=256, help="vllm 并发请求数")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_problems", type=int, default=None, help="调试用：只处理前 N 道题")
    p.add_argument("--resume", action="store_true", help="跳过已有输出中的题目")
    return p.parse_args()


def load_done_indices(output_path: str) -> set[int]:
    """加载已完成的题目索引（用于断点续传）。"""
    done = set()
    p = Path(output_path)
    if not p.exists():
        return done
    with open(p) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["index"])
            except Exception:
                pass
    return done


def main():
    args = parse_args()

    # ── 加载数据 ──
    print(f"[pass@k] 加载数据: {args.data}")
    df = pd.read_parquet(args.data)
    if args.max_problems is not None:
        df = df.iloc[: args.max_problems]
    print(f"[pass@k] 共 {len(df)} 道题")

    # ── 断点续传 ──
    done_indices: set[int] = set()
    if args.resume:
        done_indices = load_done_indices(args.output)
        print(f"[pass@k] 已完成 {len(done_indices)} 道，跳过")

    # ── 初始化 vllm ──
    print(f"[pass@k] 加载模型: {args.model}  tensor_parallel={args.n_gpus}")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.n_gpus,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # ── 导入验证器 ──
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from recipe.MRSD.mrsd.verifier import is_correct, compute_pass_at_k

    # ── 构建 prompts ──
    records = df.to_dict(orient="records")
    pending = [
        (i, rec)
        for i, rec in enumerate(records)
        if i not in done_indices
    ]
    print(f"[pass@k] 待处理: {len(pending)} 道题")

    # ── 准备输出 ──
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output, "a", encoding="utf-8")

    # ── 分批处理 ──
    total = len(pending)
    n_done = 0
    t0 = time.time()

    # 每批 batch_size // n_samples 道题（避免超显存）
    prob_batch_size = max(1, args.batch_size // args.n_samples)

    for batch_start in range(0, total, prob_batch_size):
        batch = pending[batch_start : batch_start + prob_batch_size]

        # 将 chat messages 应用 tokenizer template → 字符串 prompt
        raw_prompts = []
        for _, rec in batch:
            messages = rec["prompt"]
            if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
                pass  # 已是 list[dict]
            else:
                messages = list(messages)
            # apply_chat_template: add_generation_prompt=True
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Qwen3 thinking mode: 关闭 /think 以加速诊断
            if "<|im_start|>" in prompt_str:
                prompt_str += ""  # 不额外加前缀
            raw_prompts.append(prompt_str)

        # vllm 批量生成（每道题 n_samples 条）
        outputs = llm.generate(raw_prompts, sampling_params)

        for (idx, rec), output in zip(batch, outputs):
            ground_truth = rec["reward_model"]["ground_truth"]
            question = rec["extra_info"].get("question", "")
            responses = [o.text for o in output.outputs]

            correct_flags = [is_correct(resp, ground_truth) for resp in responses]
            n_correct = sum(correct_flags)

            pass_at_1 = compute_pass_at_k(correct_flags, 1)
            pass_at_8 = compute_pass_at_k(correct_flags, 8)
            pass_at_64 = compute_pass_at_k(correct_flags, min(64, len(correct_flags)))

            result = {
                "index": idx,
                "question": question[:200],
                "ground_truth": ground_truth,
                "difficulty": rec["extra_info"].get("difficulty", -1),
                "topic": rec["extra_info"].get("topic", ""),
                "n_samples": len(responses),
                "n_correct": n_correct,
                "pass_at_1": round(pass_at_1, 4),
                "pass_at_8": round(pass_at_8, 4),
                "pass_at_64": round(pass_at_64, 4),
                "is_dead_zone": (n_correct == 0),
                # 保存第一条错误轨迹（供 context B 测试用）
                "first_wrong_traj": responses[0] if n_correct < len(responses) else "",
                "wrong_trajs": [r for r, c in zip(responses, correct_flags) if not c][:4],
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += 1

        elapsed = time.time() - t0
        speed = n_done / elapsed if elapsed > 0 else 0
        eta = (total - n_done) / speed if speed > 0 else 0
        print(
            f"[pass@k] {n_done}/{total}  "
            f"speed={speed:.1f} prob/s  ETA={eta/60:.1f}min"
        )

    out_f.close()

    # ── 汇总统计 ──
    print("\n[pass@k] === 结果汇总 ===")
    results = []
    with open(args.output) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    total_probs = len(results)
    dead_zone = [r for r in results if r["is_dead_zone"]]
    pass1_mean = sum(r["pass_at_1"] for r in results) / total_probs if total_probs else 0
    pass64_mean = sum(r["pass_at_64"] for r in results) / total_probs if total_probs else 0

    print(f"  总题数:       {total_probs}")
    print(f"  死区题数:     {len(dead_zone)}  ({100*len(dead_zone)/total_probs:.1f}%)")
    print(f"  pass@1 均值:  {pass1_mean:.3f}")
    print(f"  pass@64 均值: {pass64_mean:.3f}")

    # 按 topic 分
    from collections import defaultdict
    by_topic: dict[str, list] = defaultdict(list)
    for r in results:
        by_topic[r.get("topic", "unknown")].append(r)
    print("\n  Topic 死区率:")
    for topic, rs in sorted(by_topic.items()):
        dead = sum(1 for r in rs if r["is_dead_zone"])
        print(f"    {topic}: {dead}/{len(rs)} = {100*dead/len(rs):.1f}%")

    # 保存死区题目列表
    dead_output = Path(args.output).parent / "dead_zone_problems.jsonl"
    with open(dead_output, "w") as f:
        for r in dead_zone:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[pass@k] 死区题目已保存到: {dead_output}")
    print(f"[pass@k] 完整结果: {args.output}")


if __name__ == "__main__":
    main()
