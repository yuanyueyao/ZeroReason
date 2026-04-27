#!/usr/bin/env python3
"""将验证阶段 dump 的 JSONL 按规则分数拆成 correct / wrong 两个文件。

默认读 ``0.jsonl``（与 RayPPOTrainer 在 global_steps=0 时写出的一致）；
每行应含 ``score`` 字段（token 上规则奖励之和，GSM8K 下为 0 或 1）。"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _is_correct(row: dict[str, Any]) -> bool:
    s = row.get("score", row.get("reward", None))
    if s is None:
        return False
    try:
        return float(s) >= 1.0
    except (TypeError, ValueError):
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input_jsonl",
        nargs="?",
        default="0.jsonl",
        help="输入 JSONL 路径（默认 0.jsonl）",
    )
    p.add_argument(
        "-d",
        "--dir",
        default=".",
        help="若 input 为相对路径，先在此目录下解析（默认当前目录）",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        default=None,
        help="输出目录（默认同输入文件所在目录）",
    )
    p.add_argument(
        "--correct-name",
        default="correct.jsonl",
        help="答对行写入的文件名",
    )
    p.add_argument(
        "--wrong-name",
        default="wrong.jsonl",
        help="答错行写入的文件名",
    )
    args = p.parse_args()

    base = os.path.expanduser(args.dir)
    in_path = args.input_jsonl
    if not os.path.isabs(in_path):
        in_path = os.path.join(base, in_path)
    in_path = os.path.normpath(in_path)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(in_path) or "."
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cor_path = os.path.join(out_dir, args.correct_name)
    wrg_path = os.path.join(out_dir, args.wrong_name)

    n_c = n_w = 0
    with open(in_path, encoding="utf-8") as f, open(cor_path, "w", encoding="utf-8") as fc, open(
        wrg_path, "w", encoding="utf-8"
    ) as fw:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if _is_correct(row):
                fc.write(line + "\n")
                n_c += 1
            else:
                fw.write(line + "\n")
                n_w += 1

    print(f"read: {in_path}  total={n_c + n_w}  correct={n_c}  wrong={n_w}")
    print(f"wrote: {cor_path}")
    print(f"wrote: {wrg_path}")


if __name__ == "__main__":
    main()
