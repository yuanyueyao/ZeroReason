# Copyright 2026 the verl recipe authors
"""Majority-vote A/B reward plus trainable-judge scores from non_tensor_batch."""

from __future__ import annotations

import torch

from verl import DataProto
from verl.workers.reward_manager import register

from recipe.math_challenger_solver.reward import MathMajorityRewardManager


@register("math_majority_judge_reward")
class MathMajorityJudgeRewardManager(MathMajorityRewardManager):
    """B 的 reward 有两种模式（由 judge_dominant 控制）：

    **judge_dominant=False（默认，向后兼容）**
      base reward 来自多数投票（0/1），judge 在 base≥0.99 时叠加：
        adj = beta * jv - beta_penalty * (1 - jv)
      jv=1→+beta 加分；jv=0→-beta_penalty 减分；beta_penalty=0 时纯加分。

    **judge_dominant=True（推荐，防 reward hacking）**
      judge 直接主导 B 的奖励幅度，多数票仅决定哪些轨迹参与评分：
        reward_B = 2 * jv   （当 B 与多数票一致，jv∈[0,1] → reward∈[0,2]）
      矛盾推理（jv≈0.1）→ reward≈0.2，优秀推理（jv≈0.9）→ reward≈1.8，
      从根源上消除"答案正确但推理乱写"的 reward hacking。
      若 judge 分数尚不可用（jB=None）则退化为普通多数投票 reward=1。

    A 的 reward 两种模式下均为：原始 A_reward + alpha * judge_score_problem_norm。
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        tokenizer_B=None,
        rollout_n: int = 1,
        *,
        alpha: float = 0.1,
        beta: float = 0.1,
        beta_penalty: float = 0.0,
        judge_dominant: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, num_examine, tokenizer_B, rollout_n, **kwargs)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_penalty = float(beta_penalty)
        # judge_dominant=True：judge 分数直接决定 B 的 reward 幅度（范围 [0,2]），
        # 替代原来的 base+adj 叠加方式；beta/beta_penalty 在此模式下无效。
        self.judge_dominant = bool(judge_dominant)

    def __call__(self, data_A: DataProto, data_B: DataProto | None = None, *, return_dict: bool = False):
        out = super().__call__(data_A, data_B, return_dict=True)
        reward_tensor_A = out["reward_tensor_A"]
        reward_tensor_B: torch.Tensor | None = out.get("reward_tensor_B")
        reward_extra = out["reward_extra_info"]

        # ── A 的 judge bonus（两种模式通用）──────────────────────────────────
        jA = data_A.non_tensor_batch.get("judge_score_problem_norm")
        n_a = len(data_A)
        for i in range(n_a):
            bonus = 0.0
            if jA is not None and len(jA) == n_a and self.alpha > 0.0:
                jv = float(jA[i])
                if jv > 0.0:
                    item = data_A[i]
                    plen = item.batch["prompts"].shape[-1]
                    vlen = int(item.batch["attention_mask"][plen:].sum())
                    if vlen >= 1:
                        reward_tensor_A[i, vlen - 1] = reward_tensor_A[i, vlen - 1] + self.alpha * jv
                        bonus = self.alpha * jv
            reward_extra.setdefault("A_judge_bonus", []).append(bonus)

        # ── B 的 judge 处理 ──────────────────────────────────────────────────
        if data_B is None or reward_tensor_B is None:
            # 无 B 数据，nothing to do
            return self._pack_return(out, reward_tensor_A, reward_tensor_B, return_dict)

        len_b = len(data_B)
        jB = data_B.non_tensor_batch.get("judge_score_answer_norm")

        if self.judge_dominant:
            # ── judge_dominant 模式：reward_B = 2 * jv（多数票正确时）──────
            if jB is not None and len(jB) == len_b:
                for j in range(len_b):
                    jv = float(jB[j])
                    item = data_B[j]
                    plen = item.batch["prompts"].shape[-1]
                    vlen = int(item.batch["attention_mask"][plen:].sum())
                    if vlen < 1:
                        reward_extra.setdefault("B_judge_bonus", []).append(0.0)
                        continue
                    base = float(reward_tensor_B[j, vlen - 1].item())
                    if base >= 0.99:
                        # 用 judge 分直接替换 base（1.0），范围 [0, 2]
                        new_r = 2.0 * jv
                        reward_tensor_B[j, vlen - 1] = new_r
                        reward_extra.setdefault("B_judge_bonus", []).append(new_r - base)
                    else:
                        # 多数票错误的轨迹保持原 reward（0 或 -0.5），不受 judge 影响
                        reward_extra.setdefault("B_judge_bonus", []).append(0.0)
            else:
                # judge 分数不可用（例如 has_b=False 的步骤）→ 保持原多数投票 reward
                for j in range(len_b):
                    reward_extra.setdefault("B_judge_bonus", []).append(0.0)

        else:
            # ── 叠加模式（默认）：base + beta*jv - beta_penalty*(1-jv) ───────
            if (self.beta > 0.0 or self.beta_penalty > 0.0) and jB is not None and len(jB) == len_b:
                for j in range(len_b):
                    jv = float(jB[j])
                    item = data_B[j]
                    plen = item.batch["prompts"].shape[-1]
                    vlen = int(item.batch["attention_mask"][plen:].sum())
                    if vlen < 1:
                        reward_extra.setdefault("B_judge_bonus", []).append(0.0)
                        continue
                    base = float(reward_tensor_B[j, vlen - 1].item())
                    if base >= 0.99:
                        adj = self.beta * jv - self.beta_penalty * (1.0 - jv)
                        reward_tensor_B[j, vlen - 1] = base + adj
                        reward_extra.setdefault("B_judge_bonus", []).append(adj)
                    else:
                        reward_extra.setdefault("B_judge_bonus", []).append(0.0)
            else:
                for j in range(len_b):
                    reward_extra.setdefault("B_judge_bonus", []).append(0.0)

        return self._pack_return(out, reward_tensor_A, reward_tensor_B, return_dict)

    @staticmethod
    def _pack_return(out: dict, reward_tensor_A, reward_tensor_B, return_dict: bool):
        if return_dict:
            out["reward_tensor_A"] = reward_tensor_A
            if reward_tensor_B is not None:
                out["reward_tensor_B"] = reward_tensor_B
            out["reward_extra_info"] = dict(out["reward_extra_info"])
            return out
        return reward_tensor_A
