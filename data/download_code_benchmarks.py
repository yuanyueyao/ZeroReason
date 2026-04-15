#!/usr/bin/env python3
"""Download LiveBench、MBPP、LiveCodeBench、CodeContests、CodeForces 到本目录子文件夹。"""
from __future__ import annotations

import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(msg, flush=True)


def save_hf_dataset(repo_id: str, subdir: str) -> None:
    out = ROOT / subdir
    if out.exists() and any(out.iterdir()):
        log(f"[skip] {subdir} 已存在且非空 -> {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    log(f"=== load_dataset {repo_id} -> {out} ===")
    ds = load_dataset(repo_id)
    ds.save_to_disk(str(out))
    log(f"[ok] {subdir}")


def snap(repo_id: str, subdir: str, ignore_patterns: list[str] | None = None) -> None:
    out = ROOT / subdir
    if out.exists() and any(out.iterdir()):
        log(f"[skip] {subdir} 已存在且非空 -> {out}")
        return
    log(f"=== snapshot_download {repo_id} -> {out} ===")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out),
        ignore_patterns=ignore_patterns,
    )
    log(f"[ok] {subdir}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    # Mostly Basic Python Problems
    save_hf_dataset("google-research-datasets/mbpp", "mbpp")

    # LiveBench 代码子集（HF: livebench/coding）
    save_hf_dataset("livebench/coding", "livebench_coding")

    # LiveCodeBench：新版 datasets 不支持其 loading script，用仓库快照（含 test*.jsonl）
    snap("livecodebench/code_generation_lite", "livecodebench_code_generation_lite")

    # DeepMind Code Contests（较大，约数 GB parquet）
    snap("deepmind/code_contests", "deepmind_code_contests")

    # CodeForces 题库：排除 generated_tests/（体积可达 ~100GB+）；需要完整生成测例可自行快照该目录
    snap(
        "open-r1/codeforces",
        "open-r1_codeforces",
        ignore_patterns=["generated_tests/**"],
    )

    log("全部完成。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("已中断。")
        sys.exit(130)
