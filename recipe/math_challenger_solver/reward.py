# Copyright 2026 the verl recipe authors
"""Reward for math competition: pseudo-label = majority of B extracts; A from consensus shape."""

from __future__ import annotations

from collections import defaultdict

import torch

from verl import DataProto
from verl.workers.reward_manager import register

from recipe.math_challenger_solver.answer_extractor import AnswerExtractor
from recipe.math_challenger_solver.problem_parse import extract_challenger_problem


def _reward_A_from_B_consensus_fraction(k: int, n: int) -> float:
    """Same bell curve as code recipe: peak when half of B match majority, zero when all or none."""
    if n <= 0:
        return 0.0
    p = k / n
    return max(0.0, 1.0 - 4.0 * (p - 0.5) ** 2)


@register("math_majority_reward")
class MathMajorityRewardManager:
    """
    Expects ``data_B.non_tensor_batch``:

    - ``majority_valid`` (bool array): whether pseudo-label exists for this row's group.
    - ``majority_label`` (object array): normalized key string (same within each GRPO group).

    Solver reward: match majority label; invalid group → 0; parse fail → small negative.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        tokenizer_B=None,
        rollout_n: int = 1,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_B = tokenizer_B if tokenizer_B is not None else tokenizer
        self.num_examine = int(num_examine)
        self.rollout_n = int(kwargs.pop("rollout_n", rollout_n))
        self._extractor = AnswerExtractor()

    def __call__(self, data_A: DataProto, data_B: DataProto | None = None, *, return_dict: bool = False):
        n = self.rollout_n
        reward_tensor_A = torch.zeros_like(data_A.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        parse_arr = data_A.non_tensor_batch.get("parse_ok")
        validate_arr = data_A.non_tensor_batch.get("validate_ok")
        b_group_arr = data_A.non_tensor_batch.get("competition_b_group")
        majority_valid_sample = data_A.non_tensor_batch.get("majority_valid_for_sample")

        b_correct_by_group: dict[int, int] = defaultdict(int)
        num_groups = 0
        reward_tensor_B: torch.Tensor | None = None

        if data_B is None:
            pass
        else:
            rt_b = torch.zeros_like(data_B.batch["responses"], dtype=torch.float32)
            len_b = len(data_B)
            if len_b % n != 0:
                raise ValueError(f"math_majority reward: len(data_B)={len_b} not divisible by rollout_n={n}")

            num_groups = len_b // n
            maj_lab = data_B.non_tensor_batch.get("majority_label")
            maj_ok = data_B.non_tensor_batch.get("majority_valid")
            if maj_lab is None or len(maj_lab) != len_b:
                raise ValueError("data_B.non_tensor_batch['majority_label'] missing or length mismatch")
            if maj_ok is None or len(maj_ok) != len_b:
                raise ValueError("data_B.non_tensor_batch['majority_valid'] missing or length mismatch")

            for j in range(len_b):
                item_b = data_B[j]
                prompt_len_b = item_b.batch["prompts"].shape[-1]
                resp_ids = item_b.batch["responses"]
                vlen = item_b.batch["attention_mask"][prompt_len_b:].sum()
                resp_str = self.tokenizer_B.decode(resp_ids[: int(vlen)], skip_special_tokens=False)

                valid = bool(maj_ok[j])
                label = maj_lab[j]
                if isinstance(label, bytes):
                    label = label.decode()
                label = str(label) if label is not None else ""

                pred = self._extractor.extract(resp_str)

                if not valid:
                    r_b = 0.0
                elif pred is None:
                    r_b = -0.5
                elif pred == label:
                    r_b = 1.0
                    g_idx = j // n
                    b_correct_by_group[g_idx] += 1
                else:
                    r_b = 0.0

                if int(vlen) < 8:
                    r_b = min(r_b, 0.0)

                if self.num_examine > 0 and j < self.num_examine:
                    print("================================================")
                    print("[B response]", resp_str)
                    print("[B extracted]", pred, "[majority_label]", label if valid else "N/A")
                    print("[B majority_valid]", valid, "[B reward]", r_b)
                    print("================================================")

                rt_b[j, int(vlen) - 1] = r_b
                reward_extra_info["B_reward"].append(r_b)

            for g in range(num_groups):
                reward_extra_info["B_correct_count"].append(b_correct_by_group.get(g, 0))

            reward_tensor_B = rt_b

        for i in range(len(data_A)):
            item = data_A[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = item.batch["responses"]
            valid_response_length = item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[: int(valid_response_length)]
            response_str_A = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            if parse_arr is not None and i < len(parse_arr):
                parse_ok = bool(parse_arr[i])
            else:
                parse_ok = extract_challenger_problem(response_str_A)[1]

            if not parse_ok:
                ra = -1.0
                reward_extra_info["A_parse_ok"].append(False)
                reward_extra_info["A_validate_ok"].append(False)
                reward_extra_info["A_reward"].append(ra)
                reward_extra_info["A_b_correct_k"].append(-1)
                reward_extra_info["A_b_group"].append(-1)
            else:
                if validate_arr is not None and i < len(validate_arr):
                    validate_ok = bool(validate_arr[i])
                else:
                    validate_ok = True
                if not validate_ok:
                    ra = -0.5
                    reward_extra_info["A_parse_ok"].append(True)
                    reward_extra_info["A_validate_ok"].append(False)
                    reward_extra_info["A_reward"].append(ra)
                    reward_extra_info["A_b_correct_k"].append(-1)
                    reward_extra_info["A_b_group"].append(-1)
                else:
                    gid = int(b_group_arr[i]) if b_group_arr is not None and i < len(b_group_arr) else -1
                    if gid < 0 or gid >= num_groups:
                        ra = 0.0
                        k = -1
                    else:
                        mv = True
                        if majority_valid_sample is not None and i < len(majority_valid_sample):
                            mv = bool(majority_valid_sample[i])
                        if not mv:
                            ra = 0.0
                            k = 0
                        else:
                            k = b_correct_by_group.get(gid, 0)
                            ra = _reward_A_from_B_consensus_fraction(k, n)
                    reward_extra_info["A_parse_ok"].append(True)
                    reward_extra_info["A_validate_ok"].append(True)
                    reward_extra_info["A_reward"].append(ra)
                    reward_extra_info["A_b_correct_k"].append(k)
                    reward_extra_info["A_b_group"].append(gid)

            reward_tensor_A[i, int(valid_response_length) - 1] = ra

            if self.num_examine > 0 and i < self.num_examine:
                print("================================================")
                print("[A response]", response_str_A)
                print("[A reward]", ra)
                print("================================================")

        out: dict = {
            "reward_tensor_A": reward_tensor_A,
            "reward_extra_info": dict(reward_extra_info),
        }
        if reward_tensor_B is not None:
            out["reward_tensor_B"] = reward_tensor_B
        if return_dict:
            return out
        return reward_tensor_A
