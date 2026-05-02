"""
Step 1/3 诊断实验：对数据集做 pass@k 采样，找出 pass@K=0 的死区题目。

优化策略：两阶段筛选
  Phase A: 对所有题目做 n=1 单次采样（快速），答对的直接排除
  Phase B: 仅对 Phase A 没答对的题目做 n=63 补采样（凑满 64 次）

这样大约一半的题在 Phase A 即可排除，省掉 63 倍无用计算。

用法：
    conda run -n verl python recipe/RLSD/diagnostic/run_pass_at_k.py \\
        --data /data3/yyy/verl/data/mrsd/train_level45.parquet \\
        --model /data3/yyy/models/Qwen3-4B-Instruct-2507 \\
        --output /data3/yyy/verl/data/mrsd/pass_at_k_results.jsonl \\
        --n_samples 64 \\
        --n_gpus 8 \\
        --batch_size 256

不含 ``--resume`` 时对 ``--output`` **覆盖写入**（新一轮结果）；死区列表写入同目录 ``<stem>_dead_zone.jsonl``。
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
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
    p.add_argument("--n_samples", type=int, default=64, help="每道题总采样次数")
    p.add_argument("--n_gpus", type=int, default=8, help="使用 GPU 数量")
    p.add_argument("--batch_size", type=int, default=256, help="vllm 并发请求数")
    p.add_argument("--max_new_tokens", type=int, default=4096, help="单次生成上限（输出 token）")
    p.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="vLLM 上下文总长（prompt+生成）。默认 max(8192, max_new_tokens+2048)",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
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


def build_prompt(tokenizer, rec) -> str:
    """将 chat messages 应用 tokenizer template → 字符串 prompt。"""
    messages = rec["prompt"]
    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
        pass
    else:
        messages = list(messages)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def main():
    args = parse_args()
    if args.n_samples < 1:
        raise SystemExit("--n_samples must be >= 1")

    max_model_len = args.max_model_len
    if max_model_len is None:
        max_model_len = max(8192, args.max_new_tokens + 2048)
    if max_model_len < args.max_new_tokens + 256:
        raise SystemExit(
            f"--max_model_len ({max_model_len}) 过小，无法容纳 prompt + "
            f"--max_new_tokens ({args.max_new_tokens})，请增大 --max_model_len"
        )

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
    print(
        f"[pass@k] 加载模型: {args.model}  tensor_parallel={args.n_gpus}  "
        f"max_model_len={max_model_len}  max_new_tokens={args.max_new_tokens}"
    )
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.n_gpus,
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    # ── 导入验证器 ──
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from recipe.RLSD.mrsd.verifier import is_correct, compute_pass_at_k

    # ── 构建待处理列表 ──
    records = df.to_dict(orient="records")
    pending = [
        (i, rec)
        for i, rec in enumerate(records)
        if i not in done_indices
    ]
    print(f"[pass@k] 待处理: {len(pending)} 道题")

    # ── 准备输出 ──
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        print(f"[pass@k] 输出: 续写 {args.output}")
        out_mode = "a"
    else:
        print(f"[pass@k] 输出: 覆盖写入 {args.output}（与旧结果无关）")
        out_mode = "w"
    out_f = open(args.output, out_mode, encoding="utf-8")

    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════
    # Phase A: 快速单次筛选（n=1）
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[Phase A] 单次采样快速筛选...")
    phase_a_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # 构建所有 prompt
    all_prompts = [build_prompt(tokenizer, rec) for _, rec in pending]

    # 分批做 Phase A
    phase_a_correct = set()  # 索引集合：Phase A 答对的
    phase_a_responses: dict[int, str] = {}  # idx → first response

    batch_size_a = args.batch_size
    for batch_start in range(0, len(pending), batch_size_a):
        batch_prompts = all_prompts[batch_start: batch_start + batch_size_a]
        batch_items = pending[batch_start: batch_start + batch_size_a]
        outputs = llm.generate(batch_prompts, phase_a_params)

        for (idx, rec), output in zip(batch_items, outputs):
            resp = output.outputs[0].text
            gt = rec["reward_model"]["ground_truth"]
            phase_a_responses[idx] = resp
            if is_correct(resp, gt):
                phase_a_correct.add(idx)

        done_so_far = min(batch_start + batch_size_a, len(pending))
        print(f"  [Phase A] {done_so_far}/{len(pending)}  "
              f"已淘汰(答对): {len(phase_a_correct)}")

    n_eliminated = len(phase_a_correct)
    n_remaining = len(pending) - n_eliminated
    elapsed_a = time.time() - t0
    print(f"\n[Phase A] 完成  用时 {elapsed_a:.0f}s  "
          f"淘汰 {n_eliminated}/{len(pending)} ({100*n_eliminated/len(pending):.1f}%)  "
          f"剩余需全量采样: {n_remaining}")

    # ── 写入 Phase A 答对的结果（非死区） ──
    for idx in phase_a_correct:
        rec = records[idx]
        gt = rec["reward_model"]["ground_truth"]
        result = {
            "index": idx,
            # 勿截取：下游 dead_zone jsonl / MRSDDataset 依赖完整题干（曾与 parquet prompt 不一致）。
            "question": rec["extra_info"].get("question", ""),
            "ground_truth": gt,
            "difficulty": rec["extra_info"].get("difficulty", -1),
            "topic": rec["extra_info"].get("topic", ""),
            "n_samples": 1,
            "n_correct": 1,
            "pass_at_1": 1.0,
            "pass_at_8": 1.0,
            "pass_at_64": 1.0,
            "is_dead_zone": False,
            "first_wrong_traj": "",
            "wrong_trajs": [],
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
    out_f.flush()

    # ══════════════════════════════════════════════════════════════════
    # Phase B: 对 Phase A 未通过的题目做补采样（n_samples - 1 次）
    # ══════════════════════════════════════════════════════════════════
    remaining = [(idx, rec) for idx, rec in pending if idx not in phase_a_correct]

    if remaining:
        n_extra = args.n_samples - 1  # 已有 1 次（Phase A 的），补 n_samples-1 次

        def write_phase_b_row(idx: int, rec: dict, all_responses: list[str]) -> None:
            gt = rec["reward_model"]["ground_truth"]
            correct_flags = [is_correct(r, gt) for r in all_responses]
            n_correct = sum(correct_flags)
            pass_at_1 = compute_pass_at_k(correct_flags, 1)
            pass_at_8 = compute_pass_at_k(correct_flags, 8)
            k64 = min(64, len(correct_flags))
            pass_at_64 = compute_pass_at_k(correct_flags, k64)
            result = {
                "index": idx,
                # 勿截取：下游 dead_zone jsonl / MRSDDataset 依赖完整题干（曾与 parquet prompt 不一致）。
                "question": rec["extra_info"].get("question", ""),
                "ground_truth": gt,
                "difficulty": rec["extra_info"].get("difficulty", -1),
                "topic": rec["extra_info"].get("topic", ""),
                "n_samples": len(all_responses),
                "n_correct": n_correct,
                "pass_at_1": round(pass_at_1, 4),
                "pass_at_8": round(pass_at_8, 4),
                "pass_at_64": round(pass_at_64, 4),
                "is_dead_zone": (n_correct == 0),
                "first_wrong_traj": next(
                    (r for r, c in zip(all_responses, correct_flags) if not c), ""
                ),
                "wrong_trajs": [r for r, c in zip(all_responses, correct_flags) if not c][:4],
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if n_extra <= 0:
            # pass@1-only：不再调用 vLLM（避免 SamplingParams(n=0)）；必须把错题也写入 jsonl。
            print(f"\n[Phase B] n_samples=1，跳过补采样；将 {len(remaining)} 道题按单次结果写入...")
            for idx, rec in remaining:
                first_resp = phase_a_responses[idx]
                write_phase_b_row(idx, rec, [first_resp])
                out_f.flush()
        else:
            print(f"\n[Phase B] 对 {len(remaining)} 道题补采样 {n_extra} 次...")

            phase_b_params = SamplingParams(
                n=n_extra,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                stop=["<|im_end|>", "<|endoftext|>"],
            )

            remaining_prompts = [build_prompt(tokenizer, rec) for _, rec in remaining]
            prob_batch_size = max(1, args.batch_size // n_extra)
            n_done_b = 0

            for batch_start in range(0, len(remaining), prob_batch_size):
                batch_items = remaining[batch_start: batch_start + prob_batch_size]
                batch_prompts = remaining_prompts[batch_start: batch_start + prob_batch_size]
                outputs = llm.generate(batch_prompts, phase_b_params)

                for (idx, rec), output in zip(batch_items, outputs):
                    first_resp = phase_a_responses[idx]
                    extra_resps = [o.text for o in output.outputs]
                    all_responses = [first_resp] + extra_resps
                    write_phase_b_row(idx, rec, all_responses)
                    out_f.flush()
                    n_done_b += 1

                elapsed = time.time() - t0
                speed = n_done_b / (elapsed - elapsed_a) if (elapsed - elapsed_a) > 0 else 0
                eta = (len(remaining) - n_done_b) / speed if speed > 0 else 0
                print(
                    f"  [Phase B] {n_done_b}/{len(remaining)}  "
                    f"speed={speed:.1f} prob/s  ETA={eta/60:.1f}min"
                )

    out_f.close()

    # ── 汇总统计 ──
    total_time = time.time() - t0
    print(f"\n[pass@k] === 结果汇总 ===  总用时: {total_time/60:.1f}min")
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

    # 死区列表：与本次 --output 同名前缀，避免覆盖历史上其它诊断生成的 dead_zone_problems.jsonl
    dead_output = Path(args.output).with_name(Path(args.output).stem + "_dead_zone.jsonl")
    with open(dead_output, "w") as f:
        for r in dead_zone:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[pass@k] 死区题目已保存到: {dead_output}")
    print(f"[pass@k] 完整结果: {args.output}")


if __name__ == "__main__":
    main()
