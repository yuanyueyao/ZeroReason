# Copyright 2026 the verl recipe authors
"""Majority-vote A/B reward plus trainable-judge scores from non_tensor_batch."""

from __future__ import annotations

import torch

from verl import DataProto
from verl.workers.reward_manager import register

from recipe.math_challenger_solver.reward import MathMajorityRewardManager


@register("math_majority_judge_reward")
class MathMajorityJudgeRewardManager(MathMajorityRewardManager):
    """
    ``judge_score_problem_norm`` (per A row) and ``judge_score_answer_norm`` (per B row) when set by
    ``MathChallengerJudgeTrainer``; adds ``alpha * norm`` to A, and ``beta * norm`` to B when base token reward is ~1.0.
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
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, num_examine, tokenizer_B, rollout_n, **kwargs)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __call__(self, data_A: DataProto, data_B: DataProto | None = None, *, return_dict: bool = False):
        out = super().__call__(data_A, data_B, return_dict=True)
        reward_tensor_A = out["reward_tensor_A"]
        reward_tensor_B: torch.Tensor | None = out.get("reward_tensor_B")
        reward_extra = out["reward_extra_info"]

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

        if data_B is not None and reward_tensor_B is not None and self.beta > 0.0:
            jB = data_B.non_tensor_batch.get("judge_score_answer_norm")
            len_b = len(data_B)
            if jB is not None and len(jB) == len_b:
                for j in range(len_b):
                    jv = float(jB[j])
                    if jv <= 0.0:
                        reward_extra.setdefault("B_judge_bonus", []).append(0.0)
                        continue
                    item = data_B[j]
                    plen = item.batch["prompts"].shape[-1]
                    vlen = int(item.batch["attention_mask"][plen:].sum())
                    if vlen < 1:
                        reward_extra.setdefault("B_judge_bonus", []).append(0.0)
                        continue
                    base = float(reward_tensor_B[j, vlen - 1].item())
                    if base >= 0.99:
                        reward_tensor_B[j, vlen - 1] = base + self.beta * jv
                        reward_extra.setdefault("B_judge_bonus", []).append(self.beta * jv)
                    else:
                        reward_extra.setdefault("B_judge_bonus", []).append(0.0)
            else:
                for j in range(len_b):
                    reward_extra.setdefault("B_judge_bonus", []).append(0.0)
        else:
            if data_B is not None:
                for j in range(len(data_B)):
                    reward_extra.setdefault("B_judge_bonus", []).append(0.0)

        if return_dict:
            out["reward_tensor_A"] = reward_tensor_A
            if reward_tensor_B is not None:
                out["reward_tensor_B"] = reward_tensor_B
            out["reward_extra_info"] = dict(reward_extra)
            return out
        return reward_tensor_A
