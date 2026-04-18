# Copyright 2026 the verl recipe authors
#
# Diversity penalty functions for Model A's code generation.
#
# Each penalty function takes a list of code strings (one GRPO group, length = rollout_n)
# and returns a per-sample penalty in [0, 1].  Higher value means "more similar to group
# peers" → should be subtracted from reward after multiplying by a coefficient.
#
# Usage in fit_competition():
#   penalties = compute_group_diversity_penalty(codes, method="jaccard")
#   reward_tensor_A[i, last_tok] -= coeff * penalties[i]
#
# Selecting method via config:
#   algorithm.diversity_penalty_coeff: 0.1
#   algorithm.diversity_penalty_method: "jaccard"   # one of the keys in _PENALTY_REGISTRY

from __future__ import annotations

import ast as _ast
import difflib
import math
import re
from collections import Counter
from typing import Callable


# ── low-level similarity primitives ──────────────────────────────────────────

def _char_jaccard(a: str, b: str) -> float:
    """Character-set Jaccard similarity.  O(|a|+|b|)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union)


def _ngram_set(s: str, n: int) -> set[str]:
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    """N-gram Jaccard similarity (default n=4 for code)."""
    na, nb = _ngram_set(a, n), _ngram_set(b, n)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def _edit_ratio(a: str, b: str) -> float:
    """difflib SequenceMatcher ratio in [0, 1]: 1 = identical."""
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _ast_node_counter(code: str) -> Counter:
    """Counter of AST node type names; empty Counter on SyntaxError."""
    try:
        tree = _ast.parse(code)
        return Counter(type(node).__name__ for node in _ast.walk(tree))
    except SyntaxError:
        return Counter()


def _ast_multiset_jaccard(a: str, b: str) -> float:
    """Multiset Jaccard over AST node-type counts."""
    ca, cb = _ast_node_counter(a), _ast_node_counter(b)
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    keys = set(ca) | set(cb)
    intersection = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return intersection / union if union > 0 else 0.0


def _token_entropy(code: str) -> float:
    """
    Shannon entropy (nats) of token unigram distribution, normalised by log(vocab_size).
    Returns a diversity score in [0, 1]: 1 = maximally diverse vocabulary.
    """
    tokens = re.findall(r"\w+|[^\w\s]", code)
    if len(tokens) < 2:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
    max_entropy = math.log(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ── per-group penalty functions ───────────────────────────────────────────────
# Contract: list[str] → list[float] in [0,1], higher = more penalised.
# Empty strings (parse-failed samples) receive penalty 0.0 so they are not
# additionally penalised beyond the parse-fail reward already in CompetitionReward.

def penalty_jaccard(codes: list[str], **_) -> list[float]:
    """
    Character-level Jaccard: penalty_i = mean sim(code_i, code_j) for j≠i.

    Fast and language-agnostic; good as a default baseline.
    """
    n = len(codes)
    if n <= 1:
        return [0.0] * n
    result = []
    for i, ci in enumerate(codes):
        if not ci:
            result.append(0.0)
            continue
        sims = [_char_jaccard(ci, codes[j]) for j in range(n) if j != i and codes[j]]
        result.append(sum(sims) / len(sims) if sims else 0.0)
    return result


def penalty_ngram(codes: list[str], n: int = 4, **_) -> list[float]:
    """
    N-gram Jaccard: penalty_i = mean n-gram sim(code_i, code_j) for j≠i.

    More sensitive to local token patterns than character Jaccard.
    Keyword arg ``n`` controls gram length (default 4).
    """
    num = len(codes)
    if num <= 1:
        return [0.0] * num
    result = []
    for i, ci in enumerate(codes):
        if not ci:
            result.append(0.0)
            continue
        sims = [_ngram_jaccard(ci, codes[j], n=n) for j in range(num) if j != i and codes[j]]
        result.append(sum(sims) / len(sims) if sims else 0.0)
    return result


def penalty_edit_distance(codes: list[str], **_) -> list[float]:
    """
    Edit-distance ratio (difflib): penalty_i = mean ratio(code_i, code_j) for j≠i.

    Most precise surface-form measure; slightly slower than Jaccard methods.
    """
    n = len(codes)
    if n <= 1:
        return [0.0] * n
    result = []
    for i, ci in enumerate(codes):
        if not ci:
            result.append(0.0)
            continue
        sims = [_edit_ratio(ci, codes[j]) for j in range(n) if j != i and codes[j]]
        result.append(sum(sims) / len(sims) if sims else 0.0)
    return result


def penalty_ast(codes: list[str], **_) -> list[float]:
    """
    AST multiset-Jaccard: penalty_i = mean ast_sim(code_i, code_j) for j≠i.

    Captures structural/algorithmic similarity regardless of variable naming.
    Falls back gracefully when code cannot be parsed (penalty 0).
    """
    n = len(codes)
    if n <= 1:
        return [0.0] * n
    result = []
    for i, ci in enumerate(codes):
        if not ci:
            result.append(0.0)
            continue
        sims = [_ast_multiset_jaccard(ci, codes[j]) for j in range(n) if j != i and codes[j]]
        result.append(sum(sims) / len(sims) if sims else 0.0)
    return result


def penalty_entropy(codes: list[str], **_) -> list[float]:
    """
    Token-entropy penalty: penalty_i = 1 - token_entropy(code_i).

    A solo measure — does not compare samples to each other, but instead
    penalises low-vocabulary-diversity code.  Combine with a similarity-based
    method via ``penalty_combined`` for best effect.
    """
    return [1.0 - _token_entropy(c) for c in codes]


def penalty_combined(codes: list[str], weights: dict[str, float] | None = None, **_) -> list[float]:
    """
    Weighted combination of all other methods.

    Default weights (if ``weights`` is None):
        jaccard: 0.4, ngram: 0.3, ast: 0.2, entropy: 0.1

    Pass ``weights={"jaccard": 0.5, "ast": 0.5}`` via config
    ``algorithm.diversity_penalty_kwargs.weights``.
    """
    if weights is None:
        weights = {"jaccard": 0.4, "ngram": 0.3, "ast": 0.2, "entropy": 0.1}
    total_w = sum(weights.values())
    if total_w <= 0:
        return [0.0] * len(codes)

    accum = [0.0] * len(codes)
    for method_name, w in weights.items():
        if w <= 0:
            continue
        fn = _PENALTY_REGISTRY.get(method_name)
        if fn is None or fn is penalty_combined:
            raise ValueError(f"penalty_combined: unknown sub-method {method_name!r}")
        sub = fn(codes)
        for i, v in enumerate(sub):
            accum[i] += w * v
    return [v / total_w for v in accum]


# ── registry & public API ─────────────────────────────────────────────────────

_PENALTY_REGISTRY: dict[str, Callable[..., list[float]]] = {
    "jaccard": penalty_jaccard,
    "ngram": penalty_ngram,
    "edit_distance": penalty_edit_distance,
    "ast": penalty_ast,
    "entropy": penalty_entropy,
    "combined": penalty_combined,
}

# Pairwise similarity function for each method (used by compute_memory_similarity).
# "entropy" has no pairwise notion → fall back to char Jaccard.
# "combined" → average of char Jaccard + 4-gram Jaccard.
def _pairwise_sim_combined(a: str, b: str) -> float:
    return (_char_jaccard(a, b) + _ngram_jaccard(a, b)) / 2.0


_PAIRWISE_SIM_REGISTRY: dict[str, Callable[[str, str], float]] = {
    "jaccard": _char_jaccard,
    "ngram": _ngram_jaccard,
    "edit_distance": _edit_ratio,
    "ast": _ast_multiset_jaccard,
    "entropy": _char_jaccard,
    "combined": _pairwise_sim_combined,
}


def compute_memory_similarity(
    codes: list[str],
    past_codes: list[str],
    method: str = "jaccard",
    **kwargs,
) -> list[float]:
    """
    For each code in ``codes``, compute its mean similarity to all non-empty
    codes in ``past_codes`` (the cross-round memory window).

    Returns a list of floats in [0, 1] with the same length as ``codes``.
    If ``past_codes`` is empty, returns ``[1.0] * len(codes)`` so that
    multiplying by this value leaves the within-group penalty unchanged.

    Args:
        codes:      Current-round code strings.
        past_codes: Flat list of historical code strings from the memory window.
        method:     Same method key as used for the group penalty.
        **kwargs:   Forwarded to the pairwise similarity function (e.g. ``n`` for ngram).
    """
    valid_past = [p for p in past_codes if p]
    if not valid_past:
        return [1.0] * len(codes)

    sim_fn = _PAIRWISE_SIM_REGISTRY.get(method, _char_jaccard)
    result: list[float] = []
    for ci in codes:
        if not ci:
            result.append(0.0)
            continue
        sims = [sim_fn(ci, p, **kwargs) if method == "ngram" else sim_fn(ci, p)
                for p in valid_past]
        result.append(sum(sims) / len(sims))
    return result


def compute_group_diversity_penalty(
    codes: list[str],
    method: str = "jaccard",
    **kwargs,
) -> list[float]:
    """
    Compute per-sample diversity penalties for one GRPO group.

    Args:
        codes:   List of ``func_code`` strings, one per rollout sample in the group.
                 Empty strings (parse-failed samples) are allowed and receive 0 penalty.
        method:  One of ``"jaccard"``, ``"ngram"``, ``"edit_distance"``, ``"ast"``,
                 ``"entropy"``, ``"combined"``.
        **kwargs: Forwarded to the selected penalty function
                  (e.g. ``n=3`` for ``"ngram"``; ``weights={...}`` for ``"combined"``).

    Returns:
        List of floats in [0, 1], length == len(codes).
        Subtract ``coeff * penalty[i]`` from sample i's reward.
    """
    if method not in _PENALTY_REGISTRY:
        raise ValueError(
            f"Unknown diversity penalty method: {method!r}. "
            f"Available: {sorted(_PENALTY_REGISTRY)}"
        )
    return _PENALTY_REGISTRY[method](codes, **kwargs)
