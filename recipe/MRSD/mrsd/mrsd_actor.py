"""
MRSD 自定义 Actor。

继承 verl DataParallelPPOActor，覆写 update_policy 实现：
    L = Σ_t clip(log π_teacher(t) - log π_student(t), max=kl_clip) * response_mask_t

梯度只通过 π_student（学生侧）传播。
teacher_log_prob 通过 DataProto 中的 old_log_probs 字段传入（已 detach）。

约束：仅修改 recipe/MRSD/ 目录。
"""

from __future__ import annotations

import torch
from typing import Dict

from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.dp_actor import DataParallelPPOActor


class MRSDPPOActor(DataParallelPPOActor):
    """
    MRSD-specific Actor。

    与标准 PPO Actor 的区别：
      - update_policy 只计算 per-token KL loss，不使用 advantage / clip ratio
      - old_log_probs 字段解释为 teacher log-probs（而非旧策略 log-probs）
      - 通过 meta_info["mrsd_kl_clip"] 控制 per-token clip 阈值
    """

    def update_policy(self, data: DataProto) -> Dict:
        """MRSD KL Loss 训练步骤。"""
        self.actor_module.train()

        temperature = data.meta_info.get("temperature", 1.0)
        kl_clip: float = float(data.meta_info.get("mrsd_kl_clip", 10.0))

        # ── 提取 batch 字段 ──────────────────────────────────────────────
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",  # 教师 log-probs
            "response_mask",  # response 位置掩码
        ]
        batch = data.select(batch_keys=select_keys).batch

        # ── 分 mini-batch ──────────────────────────────────────────────
        if self.config.use_dynamic_bsz:
            from verl.utils.seqlen_balancing import rearrange_micro_batches
            max_token_len = self.config.ppo_max_token_len_per_gpu * getattr(self, "ulysses_sequence_parallel_size", 1)
            micro_batches, _ = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            micro_batches = batch.split(self.config.ppo_micro_batch_size_per_gpu)

        metrics: Dict = {}
        self.actor_optimizer.zero_grad()

        for micro_batch in micro_batches:
            # 搬到 GPU
            if isinstance(micro_batch, DataProto):
                micro_batch_dict = micro_batch.batch.to(self.device_name)
            elif hasattr(micro_batch, "to"):
                micro_batch_dict = micro_batch.to(self.device_name)
            else:
                micro_batch_dict = {
                    k: (v.to(self.device_name) if isinstance(v, torch.Tensor) else v)
                    for k, v in micro_batch.items()
                }

            # teacher log-probs（已 detach，无梯度）
            teacher_log_prob: torch.Tensor = micro_batch_dict["old_log_probs"].detach()  # (B, T_resp)

            # response mask（在 response 区域内的有效 token）
            responses = micro_batch_dict["responses"]          # (B, T_resp)
            T_resp = responses.shape[1]
            if "response_mask" in micro_batch_dict:
                response_mask = micro_batch_dict["response_mask"].float()[:, :T_resp]
            else:
                attention_mask = micro_batch_dict["attention_mask"]
                response_mask = attention_mask[:, -T_resp:].float()

            # ── Student forward pass ────────────────────────────────────
            _, student_log_prob = self._forward_micro_batch(
                micro_batch=micro_batch_dict,
                temperature=temperature,
                calculate_entropy=False,
            )  # student_log_prob: (B, T_resp)

            # ── Per-token KL loss ───────────────────────────────────────
            #   kl_t = log π_teacher(t) - log π_student(t)
            #   gradient only through student_log_prob
            per_token_kl = teacher_log_prob - student_log_prob   # (B, T_resp)
            if kl_clip > 0:
                per_token_kl = per_token_kl.clamp(max=kl_clip)

            denom = response_mask.sum().clamp(min=1.0)
            kl_loss = (per_token_kl * response_mask).sum() / denom

            # 梯度累积缩放
            if self.config.use_dynamic_bsz:
                n_samples = responses.shape[0]
                loss = kl_loss * (n_samples / self.config.ppo_mini_batch_size)
            else:
                loss = kl_loss / self.gradient_accumulation

            loss.backward()

            # ── 记录诊断指标 ────────────────────────────────────────────
            with torch.no_grad():
                raw_kl = (teacher_log_prob - student_log_prob) * response_mask
                clip_frac = ((teacher_log_prob - student_log_prob) > kl_clip).float() * response_mask
                step_data = {
                    "mrsd/kl_loss": kl_loss.item(),
                    "mrsd/kl_raw_mean": (raw_kl.sum() / denom).item(),
                    "mrsd/kl_clip_frac": (clip_frac.sum() / denom).item(),
                    "mrsd/n_response_tokens": denom.item(),
                }
            append_to_dict(metrics, step_data)

        grad_norm = self._optimizer_step()
        append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
