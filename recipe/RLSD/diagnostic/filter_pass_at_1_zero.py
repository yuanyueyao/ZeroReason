"""
从 pass@k 结果 jsonl 中筛选 pass_at_1 == 0 的题目（单次采样估计为全错）。

典型输入：run_pass_at_k.py 生成的 pass_at_k_results.jsonl（含 pass_at_1 字段）。

用法：
    conda run -n verl python recipe/RLSD/diagnostic/filter_pass_at_1_zero.py \\
        --input /data3/yyy/verl/data/mrsd/pass_at_k_results.jsonl \\
        --output /data3/yyy/verl/data/mrsd/pass_at_1_zero.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="筛选 pass_at_1=0 的记录")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("/data3/yyy/verl/data/mrsd/pass_at_k_results.jsonl"),
        help="含 pass_at_1 字段的 jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/data3/yyy/verl/data/mrsd/pass_at_1_zero.jsonl"),
        help="输出 jsonl（仅保留 pass_at_1=0）",
    )
    return p.parse_args()


def is_zero_pass_at_1(v) -> bool:
    if v is None:
        return False
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return abs(x) < 1e-9


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"[filter] 找不到输入文件: {args.input}")

    total = 0
    kept = 0
    missing = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            pa1 = rec.get("pass_at_1")
            if pa1 is None:
                missing += 1
                continue
            if is_zero_pass_at_1(pa1):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[filter] 读取: {args.input}")
    print(f"[filter] 总条数: {total}  |  pass_at_1=0: {kept}  |  缺 pass_at_1 跳过: {missing}")
    print(f"[filter] 已写入: {args.output}")


if __name__ == "__main__":
    main()
