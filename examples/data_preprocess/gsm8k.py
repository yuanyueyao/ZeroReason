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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets


def _load_gsm8k_from_huggingface(hf_endpoint=None):
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint.rstrip("/")
    return datasets.load_dataset("openai/gsm8k", "main")


def _load_gsm8k_from_modelscope(cache_dir=None):
    try:
        from modelscope.msdatasets import MsDataset
        from modelscope.utils.constant import Hubs
    except ImportError as e:
        raise ImportError(
            "使用 --source modelscope 需安装: pip install modelscope addict"
        ) from e
    # 与 openai/gsm8k 的 config 名一致；数据从 ModelScope Hub 拉取，不经 Hugging Face 端点
    raw = MsDataset.load(
        "AI-ModelScope/gsm8k",
        subset_name="main",
        hub=Hubs.modelscope,
        cache_dir=cache_dir,
    )
    if isinstance(raw, datasets.DatasetDict):
        return raw
    # 少数类型会包一层 MsDataset
    inner = getattr(raw, "ds_instance", None)
    if inner is not None and isinstance(inner, datasets.DatasetDict):
        return inner
    raise TypeError(f"未预期的 ModelScope 返回类型: {type(raw)}")


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        "--local_save_dir",
        default="/data3/yyy/verl/data/gsm8k",
        dest="local_dir",
        help="Output directory for train.parquet and test.parquet (same as --local_save_dir).",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
        help="huggingface: 从 Hub 下载；modelscope: 从 ModelScope（国内源）下载同名数据。",
    )
    parser.add_argument(
        "--hf_endpoint",
        default=None,
        help="仅 --source huggingface：覆盖 HF 端点（镜像连不上时可设为 https://huggingface.co）。",
    )
    parser.add_argument(
        "--ms_cache_dir",
        default=None,
        help="仅 --source modelscope：ModelScope 数据集缓存目录（默认走 ModelScope 全局配置）。",
    )

    args = parser.parse_args()

    # 与下游 verl 配置一致：内容同 openai/gsm8k，仅下载渠道不同
    data_source = "openai/gsm8k"

    if args.source == "modelscope":
        dataset = _load_gsm8k_from_modelscope(cache_dir=args.ms_cache_dir)
    else:
        dataset = _load_gsm8k_from_huggingface(hf_endpoint=args.hf_endpoint)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = ''

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw if not instruction_following else question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
