# Copyright 2026 the recipe authors
#
# Build RLHF parquet for full GSM8K (train + test), matching few_data_thinking prompt style
# and verl GSM8K rule-reward schema (data_source openai/gsm8k, reward_model.ground_truth).

from __future__ import annotations

import argparse
import json
import os
import re

from datasets import Dataset, load_dataset

_INSTRUCTION = (
    "Let's think step by step and put the final answer after '####' (digits only, e.g. #### 42)."
)
_DATA_SOURCE = "openai/gsm8k"


def extract_solution(solution_str: str) -> str:
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None, f"no #### answer in: {solution_str[:200]!r}..."
    final_solution = solution.group(0).split("#### ")[1].replace(",", "")
    return final_solution


def _make_map_fn(split: str):
    def process_fn(example: dict, idx: int) -> dict:
        question_raw = example["question"]
        answer_raw = example["answer"]
        solution = extract_solution(answer_raw)
        content = f"{question_raw} {_INSTRUCTION}"
        return {
            "data_source": _DATA_SOURCE,
            "prompt": [{"role": "user", "content": content}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }

    return process_fn


def rows_for_split(split_name: str, hf_split) -> list[dict]:
    rows: list[dict] = []
    fn = _make_map_fn(split_name)
    for i in range(len(hf_split)):
        rows.append(fn(hf_split[i], i))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write GSM8K train.parquet + test.parquet for GRPO (verl schema).")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <this_recipe>/data).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=_DATA_SOURCE,
        help="HuggingFace dataset id (default: openai/gsm8k).",
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(here, "data")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset(args.dataset_name, "main")
    train_rows = rows_for_split("train", ds["train"])
    test_rows = rows_for_split("test", ds["test"])

    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "test.parquet")
    Dataset.from_list(train_rows).to_parquet(train_path)
    Dataset.from_list(test_rows).to_parquet(test_path)

    meta = {
        "dataset_name": args.dataset_name,
        "config": "main",
        "instruction_suffix": _INSTRUCTION,
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "train_parquet": train_path,
        "test_parquet": test_path,
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(train_rows)} train rows -> {train_path}")
    print(f"Wrote {len(test_rows)} test rows -> {test_path}")
    print(f"Meta -> {os.path.join(out_dir, 'train_meta.json')}")


if __name__ == "__main__":
    main()
