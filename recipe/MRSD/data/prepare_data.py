"""
将数学数据集转换为 MRSD 训练所需的 verl parquet 格式。

支持两种数据源：
  - deepmath: zwhe99/DeepMath-103K（Level 7-9）
  - math:     Hendrycks MATH（Level 5，用于开发调试）

输出格式与 verl gsm8k parquet 一致：
  columns: data_source, prompt, ability, reward_model, extra_info

用法示例：
    # DeepMath（需先运行 download_deepmath.py）
    conda run -n verl python recipe/MRSD/data/prepare_data.py \
        --source deepmath \
        --input /data3/yyy/verl/data/deepmath/train_level7to9.parquet \
        --output_dir /data3/yyy/verl/data/mrsd

    # MATH Level 5（本地开发用）
    conda run -n verl python recipe/MRSD/data/prepare_data.py \
        --source math \
        --input /data3/yyy/Self-RePrompt/data/raw/hendrycks_math.json \
        --output_dir /data3/yyy/verl/data/mrsd \
        --level 5
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ──────────────────────────────────────────────────────────────
# 系统提示（与 §2.4 teacher context 模板保持一致）
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step and put your final answer within \\boxed{}."
)


def build_prompt(question: str) -> list[dict]:
    """构造 student 的 chat messages（仅包含问题，不含 hint）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {question}"},
    ]


# ──────────────────────────────────────────────────────────────
# DeepMath-103K 处理
# ──────────────────────────────────────────────────────────────

def load_deepmath(input_path: str) -> list[dict]:
    """读取 DeepMath-103K parquet，转换为统一中间格式。"""
    df = pd.read_parquet(input_path)
    records = []
    for _, row in df.iterrows():
        # difficulty: float，Level 7-9
        records.append(
            {
                "question": row["question"].strip(),
                "answer": str(row["final_answer"]).strip(),
                "difficulty": float(row.get("difficulty", -1)),
                "topic": str(row.get("topic", "")),
                "source": "deepmath",
            }
        )
    return records


# ──────────────────────────────────────────────────────────────
# Hendrycks MATH 处理
# ──────────────────────────────────────────────────────────────

_SIMPLE_NUM_RE = re.compile(
    r"^[+-]?\d+(\.\d+)?$"          # 整数/小数
    r"|^[+-]?\d+/\d+$"             # 分数 a/b
    r"|^\\frac\{[^}]+\}\{[^}]+\}$" # LaTeX 分数 \frac{a}{b}
)


def _is_balanced(s: str) -> bool:
    """检查花括号是否平衡（截断的 LaTeX 答案会不平衡）。"""
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def _is_clean_answer(ans: str) -> bool:
    """判断答案是否完整且适合 rule-based verifier。"""
    ans = ans.strip()
    if not ans:
        return False
    # 花括号必须平衡，否则是截断
    if not _is_balanced(ans):
        return False
    return True


def load_math_hendrycks(input_path: str, level: int = 5) -> list[dict]:
    """读取 Hendrycks MATH JSON，筛选指定 level，转换为统一中间格式。"""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    split = data.get("train", data)
    if isinstance(split, dict) and "train" in split:
        split = split["train"]

    target_level = f"Level {level}"
    records = []
    skipped_trunc = 0
    for item in split:
        if item.get("level", "") != target_level:
            continue
        question = item.get("problem", item.get("question", "")).strip()
        answer = item.get("answer", "").strip()
        if not question or not answer:
            continue
        if not _is_clean_answer(answer):
            skipped_trunc += 1
            continue
        records.append(
            {
                "question": question,
                "answer": answer,
                "difficulty": float(level),
                "topic": item.get("type", item.get("subject", "")),
                "source": f"hendrycks_math_level{level}",
            }
        )
    if skipped_trunc:
        print(f"[prepare_data] 跳过 {skipped_trunc} 条答案不完整的记录")
    return records


# ──────────────────────────────────────────────────────────────
# 转换为 verl parquet 格式
# ──────────────────────────────────────────────────────────────

def to_verl_row(record: dict, idx: int) -> dict[str, Any]:
    """将中间格式记录转换为 verl DataProto 所需的 parquet 行。"""
    return {
        "data_source": record["source"],
        "prompt": build_prompt(record["question"]),
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": record["answer"],
        },
        "extra_info": {
            "index": idx,
            "question": record["question"],
            "answer": record["answer"],
            "difficulty": record["difficulty"],
            "topic": record["topic"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["deepmath", "math"],
        required=True,
        help="数据来源：deepmath（DeepMath-103K）或 math（Hendrycks MATH）",
    )
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument(
        "--output_dir",
        default="/data3/yyy/verl/data/mrsd",
        help="输出目录",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=5,
        help="[仅 math 来源] 筛选的 Level，默认 5",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="测试集比例，默认 0.05",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载数据 ──
    print(f"[prepare_data] 加载数据来源: {args.source}  路径: {args.input}")
    if args.source == "deepmath":
        records = load_deepmath(args.input)
    else:
        records = load_math_hendrycks(args.input, level=args.level)

    print(f"[prepare_data] 加载完成，共 {len(records)} 条")

    # ── 统计 ──
    from collections import Counter

    topic_dist = Counter(r["topic"] for r in records)
    print("[prepare_data] Topic 分布 (top-10):")
    for t, c in topic_dist.most_common(10):
        print(f"  {t}: {c}")

    # ── 转换 ──
    rows = [to_verl_row(r, i) for i, r in enumerate(records)]
    df = pd.DataFrame(rows)

    # ── 划分 train / test ──
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_test = max(1, int(len(df) * args.test_ratio))
    test_df = df.iloc[:n_test].reset_index(drop=True)
    train_df = df.iloc[n_test:].reset_index(drop=True)

    # ── 保存 ──
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train_df.to_parquet(str(train_path), index=False)
    test_df.to_parquet(str(test_path), index=False)

    print(f"[prepare_data] 训练集: {len(train_df)} 条 → {train_path}")
    print(f"[prepare_data] 测试集: {len(test_df)} 条 → {test_path}")

    # 同时保存一份 meta 供诊断脚本使用
    meta = {
        "source": args.source,
        "total": len(records),
        "train": len(train_df),
        "test": len(test_df),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    import json as _json

    with open(output_dir / "meta.json", "w") as f:
        _json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[prepare_data] 元信息已保存到 {output_dir}/meta.json")


if __name__ == "__main__":
    main()
