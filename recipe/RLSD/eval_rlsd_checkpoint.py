"""
快速评估 RLSD checkpoint 实际输出质量。

用法：
    python recipe/RLSD/eval_rlsd_checkpoint.py \
        --ckpt /data3/yyy/verl/checkpoints/rlsd/global_step_450 \
        --data /data3/yyy/verl/data/rlsd/pass_at_k_pass1_resp8192_20260501_095948_dead_zone.jsonl \
        --n 10 \
        --out /data3/yyy/verl/checkpoints/rlsd/eval_step450_samples.jsonl
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def merge_checkpoint(ckpt_actor_dir: str, target_dir: str):
    print(f"[merge] {ckpt_actor_dir} → {target_dir}")
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", ckpt_actor_dir,
        "--target_dir", target_dir,
    ]
    subprocess.run(cmd, check=True)
    print("[merge] 完成")


def run_inference(model_dir: str, data_path: str, n: int, out_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[infer] 加载模型 {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 加载 dead-zone 样题
    samples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            samples.append(json.loads(line))

    print(f"[infer] 对 {len(samples)} 道题做 greedy decoding\n")

    SYSTEM = "You are a mathematical reasoning assistant. Solve the problem step by step and put your final answer within \\boxed{}."

    results = []
    for i, s in enumerate(samples):
        question = s["question"]
        gt = s["ground_truth"]

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"Problem: {question}\n\nNow provide a detailed step-by-step solution:"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # 提取 \boxed{} 答案
        import re
        m = re.search(r"\\boxed\{([^}]*)\}", response)
        extracted = m.group(1) if m else ""

        correct = (extracted.strip() == str(gt).strip())
        status = "✓" if correct else "✗"

        print(f"[{status}] Q{i}  gt={gt}  pred={extracted or '(none)'}")
        # 打印 response 前300字符，看思路是否正常
        preview = response[:300].replace("\n", " ")
        print(f"     response: {preview}...")
        print()

        results.append({
            "idx": i,
            "question": question,
            "ground_truth": gt,
            "response": response,
            "extracted": extracted,
            "correct": correct,
        })

    n_correct = sum(r["correct"] for r in results)
    print(f"\n[infer] pass@1 = {n_correct}/{len(results)} = {n_correct/max(len(results),1):.3f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[infer] 结果已保存到 {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data3/yyy/verl/checkpoints/rlsd/global_step_450")
    ap.add_argument("--data", default="/data3/yyy/verl/data/rlsd/pass_at_k_pass1_resp8192_20260501_095948_dead_zone.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="/data3/yyy/verl/checkpoints/rlsd/eval_step450_samples.jsonl")
    ap.add_argument("--skip_merge", action="store_true", help="跳过 merge，直接从 --merged_dir 加载")
    ap.add_argument("--merged_dir", default="/data3/yyy/verl/checkpoints/rlsd/global_step_450_hf")
    args = ap.parse_args()

    merged_dir = args.merged_dir

    if not args.skip_merge:
        actor_dir = str(Path(args.ckpt) / "actor")
        Path(merged_dir).mkdir(parents=True, exist_ok=True)
        merge_checkpoint(actor_dir, merged_dir)

    run_inference(merged_dir, args.data, args.n, args.out)


if __name__ == "__main__":
    main()
