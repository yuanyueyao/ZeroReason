"""
基于已保存的模型回答，用新的 math_verify 验证器重新筛选死区题目。

读取 dead_zone_phase_a.jsonl，对其中保存的 first_wrong_traj 和 wrong_trajs
使用新 is_correct（math_verify）重新判定，去除误判的假死区。
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from recipe.RLSD.mrsd.verifier import is_correct


def main():
    input_path = "/data3/yyy/verl/data/mrsd/dead_zone_phase_a.jsonl"
    output_path = "/data3/yyy/verl/data/mrsd/dead_zone_verified.jsonl"

    print(f"[recheck] 读取: {input_path}")
    with open(input_path) as f:
        records = [json.loads(line) for line in f]
    print(f"[recheck] 共 {len(records)} 条记录")

    confirmed_dead = []
    flipped = []

    for rec in records:
        gt = rec["ground_truth"]
        all_trajs = []

        if rec.get("first_wrong_traj"):
            all_trajs.append(rec["first_wrong_traj"])
        for t in rec.get("wrong_trajs", []):
            if t and t not in all_trajs:
                all_trajs.append(t)

        any_correct = any(is_correct(traj, gt) for traj in all_trajs)

        if any_correct:
            flipped.append(rec)
        else:
            confirmed_dead.append(rec)

    print(f"\n[recheck] 结果:")
    print(f"  原死区题数:     {len(records)}")
    print(f"  翻转(实际正确): {len(flipped)}")
    print(f"  确认死区:       {len(confirmed_dead)}")

    print(f"\n[recheck] 翻转的题目示例 (前 10 条):")
    for rec in flipped[:10]:
        from recipe.RLSD.mrsd.verifier import extract_boxed_answer
        traj = rec.get("first_wrong_traj", "")
        extracted = extract_boxed_answer(traj) or "(none)"
        print(f"  idx={rec['index']}  gt={rec['ground_truth']!r}  pred={extracted!r}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in confirmed_dead:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[recheck] 确认死区已保存到: {output_path}")


if __name__ == "__main__":
    main()
