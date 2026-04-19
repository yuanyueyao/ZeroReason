# Copyright 2026 the verl recipe authors
"""Load plain question strings from verl GSM8K-style RL parquet for A's initial history."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _question_from_row(row: pd.Series) -> str:
    """Prefer ``extra_info['question']``, then ``prompt[0]['content']``."""
    ei = row.get("extra_info")
    if isinstance(ei, dict) and ei.get("question") is not None:
        return str(ei["question"]).strip()
    if isinstance(ei, str):
        # Rare: serialized; skip
        pass
    p = row.get("prompt")
    if isinstance(p, (list, tuple)) and len(p) > 0 and isinstance(p[0], dict):
        c = p[0].get("content")
        if c is not None:
            return str(c).strip()
    q = row.get("question")
    if q is not None and not (isinstance(q, float) and np.isnan(q)):
        return str(q).strip()
    return ""


def load_gsm8k_problem_excerpts_from_parquet(
    path: str,
    n: int,
    *,
    seed: int | None = None,
) -> list[str]:
    """
    Read up to ``n`` question texts from a GSM8K RL parquet (``examples/data_preprocess/gsm8k.py`` schema).

    Args:
        path: Local parquet path (``~`` expanded).
        n: Maximum number of questions to return.
        seed: If ``None``, take the first ``n`` rows in file order. If set, sample ``n`` rows
            without replacement (uniform among all rows), order preserved as sampled.
    """
    n = int(n)
    if n <= 0:
        return []

    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"initial_history_gsm8k_parquet not found: {path}")

    df = pd.read_parquet(path)
    if len(df) == 0:
        return []

    n_eff = min(n, len(df))
    if seed is None:
        sub = df.iloc[:n_eff]
    else:
        rng = np.random.default_rng(int(seed))
        idx = rng.permutation(len(df))[:n_eff]
        sub = df.iloc[idx]

    out: list[str] = []
    for _, row in sub.iterrows():
        t = _question_from_row(row)
        if t:
            out.append(t)
    return out
