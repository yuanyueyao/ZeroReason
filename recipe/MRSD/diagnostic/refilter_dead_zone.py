"""
对已有的死区题目用更长 token 和更高温度重新筛选，确认真正的 pass@64=0 死区。

参数统一：max_tokens=8196, temperature=1.0, top_p=1.0

两阶段策略：
  Phase A: n=1 快速排除能答对的
  Phase B: 对 Phase A 失败的补 63 次，确认 pass@64=0

用法：
    conda run -n verl python recipe/MRSD/diagnostic/refilter_dead_zone.py \
        --input /data3/yyy/verl/data/mrsd/dead_zone_problems.jsonl \
        --model /data3/yyy/models/Qwen3-4B-Instruct-2507 \
        --output /data3/yyy/verl/data/mrsd/dead_zone_refiltered.jsonl \
        --n_gpus 8
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/data3/yyy/verl/data/mrsd/dead_zone_problems.jsonl")
    p.add_argument("--model", default="/data3/yyy/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--output", default="/data3/yyy/verl/data/mrsd/dead_zone_refiltered.jsonl")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--n_gpus", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=8196)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    return p.parse_args()


SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step and put your final answer in \\boxed{}."
)


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()

    # ── 加载死区题目 ──
    print(f"[refilter] 加载: {args.input}")
    problems = []
    with open(args.input) as f:
        for line in f:
            problems.append(json.loads(line))
    print(f"[refilter] 共 {len(problems)} 道原死区题")
    print(f"[refilter] 参数: max_tokens={args.max_new_tokens}, temp={args.temperature}, top_p={args.top_p}")

    # ── 初始化 vLLM ──
    print(f"[refilter] 加载模型: {args.model}  TP={args.n_gpus}")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.n_gpus,
        gpu_memory_utilization=0.90,
        max_model_len=10240,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    # ── 验证器 ──
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from recipe.MRSD.mrsd.verifier import is_correct, compute_pass_at_k

    # ── 构建所有 prompt ──
    all_prompts = [build_prompt(tokenizer, p["question"]) for p in problems]
    t0 = time.time()

    # ══════════════════════════════════════════════════════════════
    # Phase A: n=1 快速筛选
    # ══════════════════════════════════════════════════════════════
    print(f"\n[Phase A] 单次采样快速筛选 ({len(problems)} 题)...")
    params_a = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    phase_a_correct = set()
    phase_a_responses: dict[int, str] = {}

    for batch_start in range(0, len(problems), args.batch_size):
        batch_prompts = all_prompts[batch_start: batch_start + args.batch_size]
        batch_problems = problems[batch_start: batch_start + args.batch_size]
        outputs = llm.generate(batch_prompts, params_a)

        for i, (prob, output) in enumerate(zip(batch_problems, outputs)):
            idx = batch_start + i
            resp = output.outputs[0].text
            phase_a_responses[idx] = resp
            if is_correct(resp, prob["ground_truth"]):
                phase_a_correct.add(idx)

        done = min(batch_start + args.batch_size, len(problems))
        print(f"  [Phase A] {done}/{len(problems)}  淘汰(答对): {len(phase_a_correct)}")

    elapsed_a = time.time() - t0
    n_remaining = len(problems) - len(phase_a_correct)
    print(f"\n[Phase A] 完成  用时 {elapsed_a:.0f}s  "
          f"淘汰 {len(phase_a_correct)}/{len(problems)} ({100*len(phase_a_correct)/len(problems):.1f}%)  "
          f"剩余: {n_remaining}")

    # ══════════════════════════════════════════════════════════════
    # Phase B: 补采样 n-1 次
    # ══════════════════════════════════════════════════════════════
    remaining_indices = [i for i in range(len(problems)) if i not in phase_a_correct]

    confirmed_dead = []

    if remaining_indices:
        n_extra = args.n_samples - 1
        print(f"\n[Phase B] 对 {len(remaining_indices)} 道题补采样 {n_extra} 次...")

        params_b = SamplingParams(
            n=n_extra,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        remaining_prompts = [all_prompts[i] for i in remaining_indices]
        prob_batch_size = max(1, args.batch_size // n_extra)
        n_done_b = 0

        for batch_start in range(0, len(remaining_indices), prob_batch_size):
            batch_idx_list = remaining_indices[batch_start: batch_start + prob_batch_size]
            batch_prompts = remaining_prompts[batch_start: batch_start + prob_batch_size]
            outputs = llm.generate(batch_prompts, params_b)

            for orig_idx, output in zip(batch_idx_list, outputs):
                prob = problems[orig_idx]
                gt = prob["ground_truth"]
                first_resp = phase_a_responses[orig_idx]
                extra_resps = [o.text for o in output.outputs]
                all_resps = [first_resp] + extra_resps

                correct_flags = [is_correct(r, gt) for r in all_resps]
                n_correct = sum(correct_flags)

                if n_correct == 0:
                    prob_out = dict(prob)
                    prob_out["n_samples"] = len(all_resps)
                    prob_out["n_correct"] = 0
                    prob_out["pass_at_1"] = 0.0
                    prob_out["pass_at_8"] = 0.0
                    prob_out["pass_at_64"] = 0.0
                    prob_out["is_dead_zone"] = True
                    prob_out["first_wrong_traj"] = first_resp
                    prob_out["wrong_trajs"] = [r for r in all_resps if not is_correct(r, gt)][:4]
                    confirmed_dead.append(prob_out)

                n_done_b += 1

            elapsed = time.time() - t0
            speed_b = n_done_b / (elapsed - elapsed_a) if (elapsed - elapsed_a) > 0 else 0
            eta = (len(remaining_indices) - n_done_b) / speed_b if speed_b > 0 else 0
            print(
                f"  [Phase B] {n_done_b}/{len(remaining_indices)}  "
                f"确认死区: {len(confirmed_dead)}  "
                f"speed={speed_b:.2f} prob/s  ETA={eta/60:.1f}min"
            )

    # ── 写入结果 ──
    total_time = time.time() - t0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in confirmed_dead:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[refilter] === 结果 ===")
    print(f"  原死区题数:     {len(problems)}")
    print(f"  Phase A 淘汰:   {len(phase_a_correct)} (给够 token 后能答对)")
    print(f"  Phase B 确认死区: {len(confirmed_dead)}")
    print(f"  淘汰率:         {100*(len(problems)-len(confirmed_dead))/len(problems):.1f}%")
    print(f"  总用时:         {total_time/60:.1f}min")
    print(f"  输出: {args.output}")


if __name__ == "__main__":
    main()
