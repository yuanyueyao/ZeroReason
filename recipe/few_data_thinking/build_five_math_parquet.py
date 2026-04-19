# Copyright 2026 the recipe authors
#
# Build RLHF parquet: five fixed math problems, explicitly repeated in order
# ( ... Q1..Q5, Q1..Q5, ... ) for deterministic sequential training.

from __future__ import annotations

import argparse
import json
import os

from datasets import Dataset

# GSM8K-style rule reward: numeric string in reward_model.ground_truth; data_source must be openai/gsm8k.
_INSTRUCTION = (
    "Let's think step by step and put the final answer after '####' (digits only, e.g. #### 42)."
)

_FIVE_PROBLEMS: list[tuple[str, str]] = [
    (
        "Tom has 24 apples. He gives one third of them to Jane. Jane then gives him 5 apples back. "
        "How many apples does Tom have now?",
        "21",
    ),
    (
        "A train travels 120 kilometers in 2 hours at constant speed. "
        "How many kilometers does it travel in 5 hours at the same speed?",
        "300",
    ),
    (
        "Three different books are placed in a row on a shelf. How many different orderings are possible?",
        "6",
    ),
    (
        "A jacket costs 80 dollars. It is on sale at 25% off. What is the sale price in dollars?",
        "60",
    ),
    (
        "The sum of two consecutive even integers is 46. What is the smaller integer?",
        "22",
    ),
]


def _row(split: str, repeat_block: int, index_in_parquet: int, question: str, ground_truth: str) -> dict:
    content = f"{question} {_INSTRUCTION}"
    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": index_in_parquet,
            "repeat_block": repeat_block,
            "question": question,
            "ground_truth": ground_truth,
        },
    }


def build_rows(*, num_cycles: int, split: str) -> list[dict]:
    """For each cycle k in 0..num_cycles-1, append Q1..Q5 in fixed order."""
    rows: list[dict] = []
    idx = 0
    for c in range(num_cycles):
        for question, gt in _FIVE_PROBLEMS:
            rows.append(_row(split, c, idx, question, gt))
            idx += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write train.parquet with 5 fixed math items × num_cycles.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory for train.parquet (default: <this_recipe>/data).",
    )
    parser.add_argument(
        "--num_cycles",
        type=int,
        default=200,
        help="How many times to repeat the full Q1..Q5 block (default 200 → 1000 rows).",
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(here, "data")
    os.makedirs(out_dir, exist_ok=True)

    rows = build_rows(num_cycles=args.num_cycles, split="train")
    out_path = os.path.join(out_dir, "train.parquet")
    Dataset.from_list(rows).to_parquet(out_path)

    meta = {
        "num_cycles": args.num_cycles,
        "num_rows": len(rows),
        "order": "For each cycle: Q1, Q2, Q3, Q4, Q5 (fixed).",
        "problems": [{"q": q, "ground_truth": g} for q, g in _FIVE_PROBLEMS],
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Meta written to {os.path.join(out_dir, 'train_meta.json')}")


if __name__ == "__main__":
    main()
