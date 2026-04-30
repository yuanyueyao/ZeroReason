"""
MRSD KL Loss 实现。

核心公式（§2.3）：
    L = D_KL( π_teacher(·|x,ŷ,y*) || π_student(·|x) )
      = Σ_t [ log π_teacher(t) - log π_student(t) ]

梯度只通过 π_student（学生侧）传播，π_teacher 做 detach。

Per-token clipping（§6.3）：
    style token（\\n, wait 等）的 KL 值可能比数学 token 高 6-15 倍，
    clip 防止这些 token 主导梯度。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


def compute_mrsd_loss(
    student_logits: torch.Tensor,       # (B, T, V) 学生 context 下的 logits（有梯度）
    teacher_log_probs: torch.Tensor,    # (B, T)    教师 context 下 token 的 log-probs（detach）
    response_ids: torch.Tensor,         # (B, T)    教师生成的 token IDs
    response_mask: torch.Tensor,        # (B, T)    1=response token, 0=prompt token
    kl_clip: float = 10.0,             # per-token KL 截断阈值（§6.3）
    loss_reduction: str = "mean",       # "mean" | "sum" | "token_mean"
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    计算 MRSD KL 损失。

    返回：
        loss:    标量 loss（有梯度）
        metrics: 调试指标字典（均无梯度）
    """
    # ── 计算学生的 per-token log-probs ──────────────────────────────────────
    # student_logits: (B, T, V) → log_softmax → (B, T, V)
    # 取实际 token 位置的 log-prob
    student_log_probs_full = F.log_softmax(student_logits.float(), dim=-1)  # (B, T, V)
    # gather 实际 token 的 log-prob → (B, T)
    student_token_log_probs = student_log_probs_full.gather(
        dim=-1, index=response_ids.unsqueeze(-1)
    ).squeeze(-1)                                                             # (B, T)

    # ── Per-token KL = log π_teacher(t) - log π_student(t) ────────────────
    # teacher_log_probs 已经是 per-token 的 gathered log-probs，直接用
    per_token_kl = teacher_log_probs.detach() - student_token_log_probs      # (B, T)

    # ── Per-token clipping（防止 style token 主导）─────────────────────────
    if kl_clip > 0:
        per_token_kl_clipped = per_token_kl.clamp(max=kl_clip)
    else:
        per_token_kl_clipped = per_token_kl

    # ── 掩码 & 汇总 ──────────────────────────────────────────────────────────
    masked_kl = per_token_kl_clipped * response_mask.float()                 # (B, T)

    if loss_reduction == "mean" or loss_reduction == "token_mean":
        denom = response_mask.float().sum().clamp(min=1.0)
        loss = masked_kl.sum() / denom
    elif loss_reduction == "sum":
        loss = masked_kl.sum()
    else:
        raise ValueError(f"Unknown loss_reduction: {loss_reduction}")

    # ── 诊断指标（detach）────────────────────────────────────────────────────
    with torch.no_grad():
        n_tokens = response_mask.float().sum()
        raw_kl_mean = (per_token_kl * response_mask.float()).sum() / n_tokens.clamp(1)
        clip_frac = ((per_token_kl > kl_clip) * response_mask.float()).sum() / n_tokens.clamp(1)
        student_entropy = -(
            (student_log_probs_full.exp() * student_log_probs_full) * response_mask.float().unsqueeze(-1)
        ).sum() / n_tokens.clamp(1)

    metrics = {
        "mrsd/kl_loss": loss.item(),
        "mrsd/kl_raw_mean": raw_kl_mean.item(),
        "mrsd/kl_clip_frac": clip_frac.item(),
        "mrsd/student_entropy": student_entropy.item(),
        "mrsd/n_response_tokens": n_tokens.item(),
    }

    return loss, metrics


def gather_log_probs(
    logits: torch.Tensor,  # (B, T, V)
    token_ids: torch.Tensor,  # (B, T)
) -> torch.Tensor:
    """
    从 logits 中取出 token_ids 对应位置的 log-prob。
    返回 (B, T)。
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)
    return log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def compute_per_token_kl(
    teacher_logits: torch.Tensor,  # (B, T, V) no_grad
    student_logits: torch.Tensor,  # (B, T, V) with_grad
    token_ids: torch.Tensor,       # (B, T)
    response_mask: torch.Tensor,   # (B, T)
    kl_clip: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    同时从 logits 计算 teacher 和 student 的 per-token KL。

    用于在同一 forward pass 中有效计算两侧 log-probs（避免重复 gather）。
    返回：
        per_token_kl:   (B, T) masked，有梯度（通过 student）
        teacher_logp:   (B, T) detached teacher log-probs，供后续步骤存储
    """
    with torch.no_grad():
        teacher_log_probs = gather_log_probs(teacher_logits, token_ids)   # (B, T)

    student_log_probs = gather_log_probs(student_logits, token_ids)       # (B, T)

    per_token_kl = teacher_log_probs.detach() - student_log_probs         # (B, T)
    if kl_clip > 0:
        per_token_kl = per_token_kl.clamp(max=kl_clip)
    per_token_kl = per_token_kl * response_mask.float()

    return per_token_kl, teacher_log_probs.detach()
