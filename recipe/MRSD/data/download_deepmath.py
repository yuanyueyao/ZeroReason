"""
下载 DeepMath-103K 数据集到本地。

用法：
    conda run -n verl python recipe/MRSD/data/download_deepmath.py \
        --output_dir /data3/yyy/verl/data/deepmath

网络要求：需要能访问 HuggingFace（hf-mirror.com 或 huggingface.co）。
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/data3/yyy/verl/data/deepmath")
    parser.add_argument(
        "--hf_endpoint",
        default=None,
        help="HuggingFace endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--repo_id",
        default="zwhe99/DeepMath-103K",
        help="HuggingFace dataset repo ID",
    )
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download_deepmath] 从 {args.repo_id} 下载数据...")
    print(f"[download_deepmath] HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'default')}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 请安装 datasets 库: pip install datasets")
        sys.exit(1)

    try:
        ds = load_dataset(args.repo_id, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: 下载失败: {e}")
        print("提示：请确认网络可以访问 HuggingFace。")
        print("      可以尝试设置 --hf_endpoint https://hf-mirror.com")
        sys.exit(1)

    print(f"[download_deepmath] 下载完成，共 {len(ds)} 条数据")
    print(f"[download_deepmath] 字段: {ds.column_names}")

    # 查看 difficulty 字段分布
    import collections
    levels = collections.Counter(
        round(float(x), 0) for x in ds["difficulty"]
    )
    print("[download_deepmath] difficulty 分布（取整）:")
    for k in sorted(levels.keys()):
        print(f"  difficulty={k:.0f}: {levels[k]} 条")

    # 保存完整训练集
    full_path = output_dir / "train_full.parquet"
    ds.to_parquet(str(full_path))
    print(f"[download_deepmath] 完整数据已保存到 {full_path}")

    # 保存 Level 7-9 子集
    hard_ds = ds.filter(lambda x: float(x["difficulty"]) >= 7.0)
    hard_path = output_dir / "train_level7to9.parquet"
    hard_ds.to_parquet(str(hard_path))
    print(f"[download_deepmath] Level 7-9 子集（{len(hard_ds)} 条）已保存到 {hard_path}")


if __name__ == "__main__":
    main()
