"""
RLSD Self-Distillation Loss。

SD 分支：full-distribution clipped KL (D_clip^KL(p_T || p_S))

    p_T = p_ref(·|x, y*, ŷ_{<n})  ← frozen ref 在特权 context (含 GT) 下的分布
    p_S = p_θ(·|x, ŷ_{<n})        ← student 在无特权 context 下的分布

    D_clip^KL(p_T || p_S) = Σ_v min(p_T(v) · log(p_T(v)/p_S(v)), τ)

梯度只通过 p_S 传播。GRPO 分支直接复用 verl 原生 update_policy。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_sd_loss_chunked(
    stu_full_logits: torch.Tensor,      # (B, seq_stu, V) 模型原始输出 logits
    ref_full_logits: torch.Tensor,      # (B, seq_ref, V) ref 模型原始输出 logits
    T_resp: int,                        # response 长度
    response_mask: torch.Tensor,        # (B, T_resp)
    temperature: float = 1.0,
    kl_clip: float = 10.0,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    内存友好的 full-distribution clipped KL loss。

    核心思路：不一次性切出 (B, T_resp, V) 的完整 logit 张量，
    而是每次只取 chunk_size 个 token 位置做 softmax + KL，
    峰值显存降低 T_resp/chunk_size 倍。

    对于 B=2, chunk_size=128, V=150K：
      每 chunk 峰值 ≈ 2×128×150K×4 bytes × 3 tensors ≈ 440MB（可接受）
    """
    denom = response_mask.float().sum().clamp(min=1.0)
    total_kl = torch.tensor(0.0, device=stu_full_logits.device)

    for t_start in range(0, T_resp, chunk_size):
        t_end = min(t_start + chunk_size, T_resp)
        chunk_mask = response_mask[:, t_start:t_end].float()

        if chunk_mask.sum() == 0:
            continue

        # logits[:, pos, :] 预测 token[pos+1]
        # response tokens 占 input_ids 的最后 T_resp 个位置
        # 预测 response[t] 的 logits 在 logits[:, -(T_resp - t) - 1, :]
        # chunk [t_start, t_end) 对应:
        #   start_idx = seq_len - T_resp - 1 + t_start  (即 -(T_resp + 1 - t_start))
        #   end_idx   = seq_len - T_resp - 1 + t_end    (即 -(T_resp + 1 - t_end))
        seq_stu = stu_full_logits.shape[1]
        seq_ref = ref_full_logits.shape[1]
        s_start = seq_stu - T_resp - 1 + t_start
        s_end = seq_stu - T_resp - 1 + t_end
        r_start = seq_ref - T_resp - 1 + t_start
        r_end = seq_ref - T_resp - 1 + t_end

        stu_chunk = stu_full_logits[:, s_start:s_end, :]
        ref_chunk = ref_full_logits[:, r_start:r_end, :]

        if temperature != 1.0:
            stu_chunk = stu_chunk / temperature
            ref_chunk = ref_chunk / temperature

        # ref distribution (no_grad)
        with torch.no_grad():
            ref_lp = F.log_softmax(ref_chunk.float(), dim=-1)
            ref_p = ref_lp.exp()

        # student distribution (有梯度)
        student_lp = F.log_softmax(stu_chunk.float(), dim=-1)

        # per-vocab KL with clip
        kl = ref_p * (ref_lp - student_lp)  # (B, chunk, V)
        if kl_clip > 0:
            kl = kl.clamp(max=kl_clip)

        per_token_kl = kl.sum(dim=-1)  # (B, chunk)
        total_kl = total_kl + (per_token_kl * chunk_mask).sum()

        del ref_lp, ref_p, student_lp, kl, per_token_kl, stu_chunk, ref_chunk

    loss = total_kl / denom

    with torch.no_grad():
        metrics = {
            "sd/kl_loss": loss.item(),
            "sd/kl_per_token_mean": loss.item(),
            "sd/n_tokens": denom.item(),
        }

    return loss, metrics


