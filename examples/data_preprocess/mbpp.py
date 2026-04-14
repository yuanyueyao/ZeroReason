# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess MBPP (google-research-datasets/mbpp, config ``full``) to verl Parquet format.

Output columns align with ``examples/data_preprocess/gsm8k.py`` + MBPP-specific ``reward_model.mbpp``.
"""

from __future__ import annotations

import argparse
import os

import shutil

import datasets

DATA_SOURCE = "google-research-datasets/mbpp"
CONFIG_NAME = "full"

# 题干里不含参考答案；评测用 reward_model / extra_info 中的标答与单测
DEFAULT_USER_PREFIX = (
    "Write a correct Python solution for the task below. Use only the Python standard library.\n"
    "Respond with complete, runnable code (functions or script as appropriate).\n\n"
    "Task:\n"
)


def _build_row(example: dict, split: str, idx: int) -> dict:
    task_id = example["task_id"]
    text = example["text"]
    code = (example["code"] or "").strip()
    test_list = list(example["test_list"])
    test_setup_code = example.get("test_setup_code") or ""
    challenge_test_list = list(example.get("challenge_test_list") or [])

    user_content = DEFAULT_USER_PREFIX + text

    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": user_content}],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": code,
            "mbpp": {
                "task_id": int(task_id),
                "test_list": test_list,
                "test_setup_code": test_setup_code,
                "challenge_test_list": challenge_test_list,
            },
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "task_id": int(task_id),
            "question": text,
            "reference_code": code,
            # 与 reward_model.mbpp 一致，供 NaiveRewardManager / val_b_compute_score 执行 assert 评测
            "mbpp": {
                "task_id": int(task_id),
                "test_list": test_list,
                "test_setup_code": test_setup_code,
                "challenge_test_list": challenge_test_list,
            },
        },
    }


def _make_map_fn(split: str):
    def process_fn(example: dict, idx: int) -> dict:
        return _build_row(example, split, idx)

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        "--local_save_dir",
        default="~/data/mbpp",
        dest="local_dir",
        help="Directory for train.parquet, test.parquet, and optional validation.parquet.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Optional HuggingFace ``cache_dir`` for ``load_dataset`` (e.g. repo ``data/`` if MBPP is already cached).",
    )
    parser.add_argument(
        "--write_validation",
        action="store_true",
        help="Also write validation.parquet from the MBPP validation split.",
    )
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    cache_dir = os.path.expanduser(args.hf_cache_dir) if args.hf_cache_dir else None
    if cache_dir:
        dataset = datasets.load_dataset(DATA_SOURCE, CONFIG_NAME, cache_dir=cache_dir)
    else:
        dataset = datasets.load_dataset(DATA_SOURCE, CONFIG_NAME)

    def _map(split: str, subset: datasets.Dataset) -> datasets.Dataset:
        cols = subset.column_names
        return subset.map(
            function=_make_map_fn(split),
            with_indices=True,
            remove_columns=cols,
        )

    train_dataset = _map("train", dataset["train"])
    test_dataset = _map("test", dataset["test"])

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.write_validation:
        val_dataset = _map("validation", dataset["validation"])
        val_dataset.to_parquet(os.path.join(local_dir, "validation.parquet"))

    if hdfs_dir is not None:
        if hdfs_dir.startswith("hdfs://"):
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(hdfs_dir)
            copy(src=local_dir, dst=hdfs_dir)
        else:
            shutil.copytree(local_dir, hdfs_dir, dirs_exist_ok=True)
