"""
RLSD Ray Trainer。

继承 RayPPOTrainer，覆写 fit() 实现 RLSD 混合训练：
  - SD 分支（dead zone）：student rollout 全部答错 → full-distribution clipped KL(p_ref || p_student)
  - GRPO 分支（mixed rewards）：student rollout 有答对有答错 → 标准 clipped policy gradient

init_workers / _save_checkpoint / _load_checkpoint 全部复用官方实现。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.utils.tracking import Tracking

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from recipe.RLSD.rlsd.dataset import MRSDDataset, MRSDProblem
from recipe.RLSD.rlsd.prompt import build_student_messages, build_teacher_privileged_messages
from recipe.RLSD.rlsd.verifier import is_correct


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _build_gen_batch(tokenizer, messages_list, max_prompt_len):
    """将 messages_list 编码为 DataProto 用于 rollout。"""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc = tokenizer(texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return DataProto.from_single_dict({"input_ids": ids, "attention_mask": mask, "position_ids": pos})


def _build_sd_train_batch(tokenizer, student_msgs, teacher_msgs, responses_text, max_prompt_len, max_resp_len):
    """
    构建 SD 分支训练 batch。

    Student 和 Teacher 拥有不同的 prompt（Teacher 含 GT 特权信息），
    但 response tokens 相同。

    返回的 DataProto 包含：
      - input_ids / attention_mask / position_ids: student 完整序列
      - ref_input_ids / ref_attention_mask / ref_position_ids: teacher 完整序列
      - responses: 共享的 response token ids
      - response_mask: response 区域掩码
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Encode student prompts
    s_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in student_msgs]
    enc_sp = tokenizer(s_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    # Encode teacher prompts (with GT privilege)
    t_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in teacher_msgs]
    enc_tp = tokenizer(t_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    # Encode shared response tokens
    tokenizer.padding_side = "right"
    enc_r = tokenizer(responses_text, return_tensors="pt", max_length=max_resp_len,
                      truncation=True, padding=True, add_special_tokens=False)
    tokenizer.padding_side = "left"

    responses_tensor = enc_r["input_ids"]
    response_mask = enc_r["attention_mask"]

    # Student full sequence
    s_full_ids = torch.cat([enc_sp["input_ids"], enc_r["input_ids"]], dim=1)
    s_full_mask = torch.cat([enc_sp["attention_mask"], enc_r["attention_mask"]], dim=1)
    s_pos = (s_full_mask.cumsum(-1) - 1).clamp(min=0)

    # Teacher full sequence
    t_full_ids = torch.cat([enc_tp["input_ids"], enc_r["input_ids"]], dim=1)
    t_full_mask = torch.cat([enc_tp["attention_mask"], enc_r["attention_mask"]], dim=1)
    t_pos = (t_full_mask.cumsum(-1) - 1).clamp(min=0)

    return DataProto.from_single_dict({
        "input_ids": s_full_ids,
        "attention_mask": s_full_mask,
        "position_ids": s_pos,
        "ref_input_ids": t_full_ids,
        "ref_attention_mask": t_full_mask,
        "ref_position_ids": t_pos,
        "responses": responses_tensor,
        "response_mask": response_mask,
    })


def _build_logprob_batch(tokenizer, messages_list, responses_text, max_prompt_len, max_resp_len):
    """构建仅用于 compute_log_prob 的 batch（单 prompt + response 文本）。"""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc_p = tokenizer(prompt_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    tokenizer.padding_side = "right"
    enc_r = tokenizer(responses_text, return_tensors="pt", max_length=max_resp_len,
                      truncation=True, padding=True, add_special_tokens=False)
    tokenizer.padding_side = "left"

    full_ids = torch.cat([enc_p["input_ids"], enc_r["input_ids"]], dim=1)
    full_mask = torch.cat([enc_p["attention_mask"], enc_r["attention_mask"]], dim=1)
    pos = (full_mask.cumsum(-1) - 1).clamp(min=0)

    return DataProto.from_single_dict({
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "position_ids": pos,
        "responses": enc_r["input_ids"],
    })


def _build_logprob_batch_from_tokens(tokenizer, messages_list, responses_tokens,
                                      response_mask, max_prompt_len):
    """
    构建用于 compute_log_prob / compute_ref_log_prob 的 batch。
    使用原始生成 token ids（不重新编码文本），确保与 old_log_probs 严格对齐。
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc_p = tokenizer(prompt_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    full_ids = torch.cat([enc_p["input_ids"], responses_tokens], dim=1)
    full_mask = torch.cat([enc_p["attention_mask"], response_mask], dim=1)
    pos = (full_mask.cumsum(-1) - 1).clamp(min=0)

    return DataProto.from_single_dict({
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "position_ids": pos,
        "responses": responses_tokens,
    })


def _build_grpo_train_batch(tokenizer, messages_list, responses_tokens, response_mask,
                            rewards, old_log_probs, group_ids, max_prompt_len,
                            ref_log_probs=None):
    """
    构建 GRPO 分支训练 batch。

    使用原始生成的 token IDs（不重新编码文本），确保 old_log_probs 与
    responses token 位置严格对齐，与官方 verl GRPO 一致。

    Args:
        responses_tokens: (B, T_r) 原始生成 token ids（来自 rollout 输出）
        response_mask:    (B, T_r) 对应 attention mask（非 padding 位置为 1）
        rewards:          list[float], 每条 response 的 reward (0/1)
        old_log_probs:    (B, T_r) rollout 时的 per-token log probs
        group_ids:        list[int], 每条 response 归属的 problem index（用于组内归一化）
        ref_log_probs:    (B, T_r) or None，参考模型的 per-token log probs；
                          use_kl_loss=True 时必须提供
    """
    from collections import defaultdict

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc_p = tokenizer(prompt_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    # 直接使用原始 token ids，不再重新编码文本（避免 round-trip tokenization 错位）
    T_r = responses_tokens.shape[1]
    full_ids = torch.cat([enc_p["input_ids"], responses_tokens], dim=1)
    full_mask = torch.cat([enc_p["attention_mask"], response_mask], dim=1)
    pos = (full_mask.cumsum(-1) - 1).clamp(min=0)

    # 按 problem (group) 内部归一化 advantage — 与官方 GRPO 完全一致：
    #   单样本 group: mean=0, std=1 → advantage = raw_reward（保留梯度信号）
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    g2scores = defaultdict(list)
    for i, gid in enumerate(group_ids):
        g2scores[gid].append(rewards_t[i])
    g2mean, g2std = {}, {}
    for gid, scores in g2scores.items():
        st = torch.stack(scores)
        if len(scores) == 1:
            g2mean[gid] = torch.tensor(0.0)   # 与官方一致：单样本不减均值
            g2std[gid] = torch.tensor(1.0)
        else:
            g2mean[gid] = st.mean()
            g2std[gid] = st.std()

    normed = torch.zeros_like(rewards_t)
    for i, gid in enumerate(group_ids):
        normed[i] = (rewards_t[i] - g2mean[gid]) / (g2std[gid] + 1e-6)

    advantages = normed.unsqueeze(1).expand(-1, T_r) * response_mask.float()

    batch_dict = {
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "position_ids": pos,
        "responses": responses_tokens,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
    }
    if ref_log_probs is not None:
        batch_dict["ref_log_prob"] = ref_log_probs

    return DataProto.from_single_dict(batch_dict)


# ══════════════════════════════════════════════════════════════════════
# RLSDTrainer
# ══════════════════════════════════════════════════════════════════════

class MRSDTrainer(RayPPOTrainer):
    """
    RLSD Trainer：继承 RayPPOTrainer，覆写 fit()。

    核心逻辑（per step）：
      1. 从死区数据集中采样 problems
      2. Student rollout（n_samples per problem）
      3. 分流：
         - 若 problem 的所有 rollout reward=0 → SD 分支
         - 若 problem 的 rollout 有 mixed rewards → GRPO 分支
      4. 分别构建 train batch，调用 update_actor
    """

    def __init__(self, *args, mrsd_dataset: Optional[MRSDDataset] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mrsd_dataset = mrsd_dataset

        mrsd_cfg = OmegaConf.select(self.config, "mrsd", default=OmegaConf.create({}))
        self.student_k = int(OmegaConf.select(mrsd_cfg, "student_rollout_per_problem", default=8))
        self.kl_clip = float(OmegaConf.select(mrsd_cfg, "kl_clip", default=10.0))
        self.problems_per_step = int(OmegaConf.select(mrsd_cfg, "problems_per_step", default=8))
        self.graduation_interval = int(OmegaConf.select(mrsd_cfg, "graduation_interval", default=100))
        self.max_prompt_len = int(OmegaConf.select(self.config, "data.max_prompt_length", default=2048))
        self.max_resp_len = int(OmegaConf.select(self.config, "data.max_response_length", default=3072))
        # GRPO-Only 模式：跳过 SD 分支，死区题不产生梯度，仅用 mixed 题做 GRPO
        self.grpo_only = bool(OmegaConf.select(mrsd_cfg, "grpo_only", default=False))

    def init_workers(self):
        """
        覆写 init_workers：使用 role="actor_rollout_ref" 使得 ref model 与 actor 共享 GPU。
        这样 update_actor 时可以在本地计算 ref_logits，无需跨进程传输。
        """
        from verl.trainer.ppo.ray_trainer import RayClassWithInitArgs, create_colocated_worker_cls

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout_ref",  # 关键：加载 ref model
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    # ──────────────────────────────────────────────────────────────────
    # 推理工具
    # ──────────────────────────────────────────────────────────────────

    def _generate(self, messages_list, n_samples):
        """对一批 messages 生成 n_samples 条回复，返回 list[list[str]]。"""
        repeated = [m for m in messages_list for _ in range(n_samples)]
        gen_batch = _build_gen_batch(self.tokenizer, repeated, self.max_prompt_len)
        gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        all_texts = self.tokenizer.batch_decode(
            out.batch["responses"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        n = len(messages_list)
        return [all_texts[i * n_samples:(i + 1) * n_samples] for i in range(n)]

    def _generate_with_logprobs(self, messages_list, n_samples):
        """
        生成回复，返回 (grouped_texts, old_log_probs, all_responses_tokens, all_response_masks)。

        - grouped_texts:        list[list[str]]，按 problem 分组的解码文本
        - old_log_probs:        (total_samples, T) per-token log probs，或 None
        - all_responses_tokens: (total_samples, T) 原始 response token ids
        - all_response_masks:   (total_samples, T) 对应 attention mask（非 padding 为 1）
        """
        repeated = [m for m in messages_list for _ in range(n_samples)]
        gen_batch = _build_gen_batch(self.tokenizer, repeated, self.max_prompt_len)
        gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        all_texts = self.tokenizer.batch_decode(
            out.batch["responses"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # vLLM rollout with calculate_log_probs=True stores "rollout_log_probs"
        old_log_probs = None
        for key in ("rollout_log_probs", "old_log_probs", "log_probs"):
            if key in out.batch:
                old_log_probs = out.batch[key]
                break

        # 保留原始 token ids 和 response mask（用于 GRPO batch 构建，避免重新编码）
        all_responses_tokens = out.batch["responses"]          # (B, T)
        resp_len = all_responses_tokens.shape[1]
        full_attn = out.batch["attention_mask"]                # (B, prompt+resp)
        all_response_masks = full_attn[:, -resp_len:]          # (B, T) response 部分

        n = len(messages_list)
        grouped_texts = [all_texts[i * n_samples:(i + 1) * n_samples] for i in range(n)]
        return grouped_texts, old_log_probs, all_responses_tokens, all_response_masks

    def _compute_log_probs(self, messages_list, responses):
        """在 messages_list[i] 的 context 下计算 responses[i] 的 per-token log-probs。"""
        data = _build_logprob_batch(self.tokenizer, messages_list, responses, self.max_prompt_len, self.max_resp_len)
        data.meta_info["micro_batch_size"] = 4
        data.meta_info["temperature"] = float(
            OmegaConf.select(self.config, "actor_rollout_ref.rollout.temperature", default=1.0)
        )
        data.meta_info["use_dynamic_bsz"] = False
        data.meta_info["max_token_len"] = 8192
        data_padded, pad_size = pad_dataproto_to_divisor(data, self.actor_rollout_wg.world_size)
        out_padded = self.actor_rollout_wg.compute_log_prob(data_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)
        batch = out.batch
        for key in ("old_log_probs", "log_probs", "response_log_probs"):
            if key in batch:
                return batch[key]
        raise KeyError(f"compute_log_prob 返回 keys: {list(batch.keys())}")

    # ──────────────────────────────────────────────────────────────────
    # 单步训练：RLSD 核心
    # ──────────────────────────────────────────────────────────────────

    def _rlsd_step(self, problems: list[MRSDProblem]) -> dict:
        """
        RLSD 单步：
          1. Student rollout
          2. 分类：全错 → SD; mixed → GRPO; 全对 → skip
          3. 分别 update
        """
        metrics = {}
        t0 = time.time()
        rollout_temp = float(
            OmegaConf.select(self.config, "actor_rollout_ref.rollout.temperature", default=1.0)
        )

        # ── Step 1: Student Rollout ──────────────────────────────────
        student_msgs = [build_student_messages(p.question) for p in problems]
        student_resps_grouped, all_old_log_probs, all_resp_tokens, all_resp_masks = \
            self._generate_with_logprobs(student_msgs, n_samples=self.student_k)

        # ── Step 2: 计算 reward + 分类 ──────────────────────────────
        sd_student_msgs, sd_teacher_msgs, sd_resps, sd_lp_indices = [], [], [], []
        grpo_msgs, grpo_resps, grpo_rewards, grpo_lp_indices, grpo_group_ids = [], [], [], [], []

        flat_idx = 0
        for pi, (prob, resps) in enumerate(zip(problems, student_resps_grouped)):
            correctness = [is_correct(r, prob.ground_truth) for r in resps]
            n_correct = sum(correctness)

            if n_correct == 0:
                # Dead zone → SD 分支：teacher 用特权 context (含 GT)
                for ri, r in enumerate(resps):
                    sd_student_msgs.append(build_student_messages(prob.question))
                    sd_teacher_msgs.append(build_teacher_privileged_messages(prob.question, prob.ground_truth))
                    sd_resps.append(r)
                    sd_lp_indices.append(flat_idx + ri)
            elif n_correct < len(resps):
                # Mixed rewards → GRPO 分支
                for ri, (r, c) in enumerate(zip(resps, correctness)):
                    grpo_msgs.append(build_student_messages(prob.question))
                    grpo_resps.append(r)
                    grpo_rewards.append(1.0 if c else 0.0)
                    grpo_lp_indices.append(flat_idx + ri)
                    grpo_group_ids.append(pi)
            # else: 全对 → 已经会做，不训练

            # 更新 dataset 状态
            self.mrsd_dataset.update_problem_stats(
                prob.index,
                new_wrong_trajs=[r for r, c in zip(resps, correctness) if not c],
                n_correct=n_correct,
                n_total=len(resps),
            )
            flat_idx += self.student_k

        metrics["rlsd/n_sd_samples"] = float(len(sd_resps))
        metrics["rlsd/n_grpo_samples"] = float(len(grpo_resps))
        metrics["rlsd/n_problems"] = float(len(problems))

        n_dead = sum(1 for prob, resps in zip(problems, student_resps_grouped)
                     if all(not is_correct(r, prob.ground_truth) for r in resps))
        n_mixed = sum(1 for prob, resps in zip(problems, student_resps_grouped)
                      if 0 < sum(is_correct(r, prob.ground_truth) for r in resps) < len(resps))
        n_solved = len(problems) - n_dead - n_mixed
        metrics["rlsd/n_dead_zone"] = float(n_dead)
        metrics["rlsd/n_mixed"] = float(n_mixed)
        metrics["rlsd/n_all_correct"] = float(n_solved)
        metrics["rlsd/grpo_only_mode"] = float(self.grpo_only)

        # ── Response length 统计（全部 rollout）────────────────────
        resp_lens = all_resp_masks.sum(dim=-1).float()  # (total_samples,)
        metrics["rollout/resp_len_min"]  = resp_lens.min().item()
        metrics["rollout/resp_len_mean"] = resp_lens.mean().item()
        metrics["rollout/resp_len_max"]  = resp_lens.max().item()

        # 各分支的长度均值（便于对比 SD/GRPO rollout 长短差异）
        if grpo_lp_indices:
            grpo_lens = resp_lens[torch.tensor(grpo_lp_indices, dtype=torch.long)]
            metrics["rollout/grpo_resp_len_mean"] = grpo_lens.mean().item()
        if sd_lp_indices and not self.grpo_only:
            sd_lens = resp_lens[torch.tensor(sd_lp_indices, dtype=torch.long)]
            metrics["rollout/sd_resp_len_mean"] = sd_lens.mean().item()

        mode_tag = "grpo-only" if self.grpo_only else "rlsd"
        print(f"  [{mode_tag}] problems={len(problems)}  dead={n_dead}  mixed={n_mixed}  solved={n_solved}")
        sys.stdout.flush()

        # ── Step 3: SD 分支训练（GRPO-Only 模式下跳过）────────────────
        if sd_resps and not self.grpo_only:
            sd_data = _build_sd_train_batch(
                self.tokenizer, sd_student_msgs, sd_teacher_msgs, sd_resps,
                self.max_prompt_len, self.max_resp_len
            )
            sd_data.meta_info["rlsd_mode"] = "sd"
            sd_data.meta_info["temperature"] = rollout_temp
            sd_data.meta_info["kl_clip"] = self.kl_clip
            sd_data.meta_info["global_token_num"] = (
                sd_data.batch["attention_mask"].sum(dim=-1).tolist()
            )
            sd_padded, pad_size = pad_dataproto_to_divisor(sd_data, self.actor_rollout_wg.world_size)
            sd_out_padded = self.actor_rollout_wg.update_actor(sd_padded)
            sd_out = unpad_dataproto(sd_out_padded, pad_size=pad_size)
            sd_metrics = sd_out.meta_info.get("metrics", {})
            for k, v in sd_metrics.items():
                metrics[k] = v

        # ── Step 4: GRPO 分支训练 ────────────────────────────────────
        if grpo_resps:
            lp_indices = torch.tensor(grpo_lp_indices, dtype=torch.long)

            # 取出对应的原始 token ids 和 response mask
            grpo_resp_tokens = all_resp_tokens[lp_indices]   # (B_grpo, T)
            grpo_resp_masks  = all_resp_masks[lp_indices]    # (B_grpo, T)

            # ── 与官方 verl 对齐：用 actor 的 compute_log_prob 重算 old_log_probs
            # 官方注释："we should always recompute old_log_probs when it is HybridEngine"
            # compute_log_prob 在 worker 层硬编码 calculate_entropy=True，
            # 因此 old_log_probs 与 entropys 一次前向同时拿到，无额外开销。
            lp_batch = _build_logprob_batch_from_tokens(
                self.tokenizer, grpo_msgs, grpo_resp_tokens, grpo_resp_masks,
                self.max_prompt_len,
            )
            lp_batch.meta_info["micro_batch_size"] = (
                self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
            )
            lp_batch.meta_info["temperature"] = rollout_temp
            lp_batch.meta_info["use_dynamic_bsz"] = bool(
                self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
            )
            lp_batch.meta_info["max_token_len"] = int(
                self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
            )
            lp_padded, lp_pad_size = pad_dataproto_to_divisor(lp_batch, self.actor_rollout_wg.world_size)
            lp_out_padded = self.actor_rollout_wg.compute_log_prob(lp_padded)
            lp_out = unpad_dataproto(lp_out_padded, pad_size=lp_pad_size)

            grpo_old_lp = lp_out.batch["old_log_probs"]   # (B_grpo, T)

            # entropy：update 前的策略熵，与官方 agg_loss 聚合方式一致
            from verl.trainer.ppo.core_algos import agg_loss as _agg_loss
            entropys = lp_out.batch["entropys"]            # (B_grpo, T)
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = _agg_loss(
                loss_mat=entropys,
                loss_mask=grpo_resp_masks.float(),
                loss_agg_mode=loss_agg_mode,
            )
            metrics["actor/entropy"] = entropy_agg.item()

            # ref_log_probs：use_kl_loss=True 时 dp_actor 需要此字段
            use_kl = OmegaConf.select(self.config, "actor_rollout_ref.actor.use_kl_loss", default=False)
            grpo_ref_lp = None
            if use_kl:
                ref_lp_data = _build_logprob_batch_from_tokens(
                    self.tokenizer, grpo_msgs, grpo_resp_tokens, grpo_resp_masks,
                    self.max_prompt_len,
                )
                ref_lp_data.meta_info["micro_batch_size"] = 4
                ref_lp_data.meta_info["temperature"] = rollout_temp
                ref_lp_data.meta_info["use_dynamic_bsz"] = False
                ref_lp_data.meta_info["max_token_len"] = 8192
                ref_lp_data_padded, pad_size_ref = pad_dataproto_to_divisor(
                    ref_lp_data, self.actor_rollout_wg.world_size
                )
                ref_out_padded = self.actor_rollout_wg.compute_ref_log_prob(ref_lp_data_padded)
                ref_out = unpad_dataproto(ref_out_padded, pad_size=pad_size_ref)
                grpo_ref_lp = ref_out.batch["ref_log_prob"]  # (B_grpo, T)

            grpo_data = _build_grpo_train_batch(
                self.tokenizer, grpo_msgs, grpo_resp_tokens, grpo_resp_masks,
                grpo_rewards, grpo_old_lp, grpo_group_ids,
                self.max_prompt_len,
                ref_log_probs=grpo_ref_lp,
            )
            grpo_data.meta_info["rlsd_mode"] = "grpo"
            grpo_data.meta_info["temperature"] = rollout_temp
            grpo_data.meta_info["global_token_num"] = (
                grpo_data.batch["attention_mask"].sum(dim=-1).tolist()
            )
            grpo_padded, pad_size = pad_dataproto_to_divisor(grpo_data, self.actor_rollout_wg.world_size)
            grpo_out_padded = self.actor_rollout_wg.update_actor(grpo_padded)
            grpo_out = unpad_dataproto(grpo_out_padded, pad_size=pad_size)
            grpo_metrics = grpo_out.meta_info.get("metrics", {})
            for k, v in grpo_metrics.items():
                metrics[k] = v

        metrics["rlsd/step_time_s"] = time.time() - t0
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # 验证（pass@1，仅 data.val_files parquet）
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _eval_val_parquet_path(val_files):
        """从 config.data.val_files 得到单个 parquet 路径；暂不合并多路径。"""
        from omegaconf import ListConfig

        if val_files is None:
            return None
        if isinstance(val_files, (list, tuple, ListConfig)):
            if len(val_files) == 0:
                return None
            if len(val_files) > 1:
                print(
                    f"[eval] WARN: data.val_files 含 {len(val_files)} 个路径，暂不合并 parquet，仅用第一个。"
                )
            val_files = val_files[0]
        path = str(val_files).strip()
        return path if path else None

    def _evaluate(self, step: int, logger: Tracking) -> dict:
        """在 data.val_files 指定的 parquet 上跑 pass@1（greedy），样本追加写入 eval_samples.jsonl。"""
        import json

        val_spec = OmegaConf.select(self.config, "data.val_files", default=None)
        val_path = self._eval_val_parquet_path(val_spec)
        if not val_path:
            return {}

        if not val_path.endswith(".parquet"):
            raise ValueError(
                f"[eval] 当前仅支持 parquet 验证集（data.val_files），收到: {val_path!r}"
            )

        print(f"\n[eval] step={step}  验证集(parquet,data.val_files): {val_path}")
        max_val = int(OmegaConf.select(self.config, "mrsd.val_max_samples", default=64))

        messages_list = []
        ground_truths = []
        questions = []

        df = pd.read_parquet(val_path)
        df = df.head(max_val)
        for _, row in df.iterrows():
            msgs = row["prompt"] if isinstance(row["prompt"], list) else list(row["prompt"])
            gt = row["reward_model"]["ground_truth"]
            q = msgs[-1]["content"] if msgs else ""
            questions.append(q)
            messages_list.append(msgs)
            ground_truths.append(gt)

        gen_batch = _build_gen_batch(self.tokenizer, messages_list, self.max_prompt_len)
        gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        gen_padded.meta_info["do_sample"] = False
        gen_padded.meta_info["temperature"] = 0.0
        gen_padded.meta_info["top_p"] = 1.0
        gen_padded.meta_info["top_k"] = 1
        out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        responses = self.tokenizer.batch_decode(
            out.batch["responses"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        from recipe.RLSD.rlsd.verifier import extract_boxed_answer

        # 样本保存路径：<default_local_dir>/eval_samples.jsonl
        ckpt_dir = Path(self.config.trainer.default_local_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sample_file = ckpt_dir / "eval_samples.jsonl"

        n_correct = 0
        n_examine = min(3, len(responses))
        with open(sample_file, "a") as fout:
            for i, (r, gt, q) in enumerate(zip(responses, ground_truths, questions)):
                correct = is_correct(r, gt)
                extracted = extract_boxed_answer(r) or ""
                if correct:
                    n_correct += 1
                if i < n_examine:
                    status = "+" if correct else "-"
                    print(f"  [{status}] Q{i} gt={gt}  pred={extracted or '(none)'}")
                    sys.stdout.flush()
                fout.write(json.dumps({
                    "step": step,
                    "idx": i,
                    "question": q,
                    "ground_truth": gt,
                    "response": r,
                    "extracted": extracted,
                    "correct": correct,
                }, ensure_ascii=False) + "\n")

        pass1 = n_correct / max(len(ground_truths), 1)
        metrics = {
            "val/pass@1": pass1,
            "val/n_correct": float(n_correct),
            "val/n_total": float(len(ground_truths)),
        }
        logger.log(data=metrics, step=step)
        print(f"[eval] step={step}  pass@1={pass1:.3f}  ({n_correct}/{len(ground_truths)})")
        print(f"[eval] 样本已追加到 {sample_file}\n")
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────────────────────────────

    def fit(self) -> None:
        """RLSD 训练循环。"""
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        total_steps = self.total_training_steps
        save_freq = int(OmegaConf.select(self.config.trainer, "save_freq", default=50))
        test_freq = int(OmegaConf.select(self.config.trainer, "test_freq", default=10))
        ckpt_dir = Path(self.config.trainer.default_local_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.global_steps = 0
        self._load_checkpoint()

        print(f"\n[RLSDTrainer] 开始  total_steps={total_steps}  active={self.mrsd_dataset.n_active}")
        self._evaluate(step=0, logger=logger)

        progress = tqdm(total=total_steps, initial=self.global_steps, desc="RLSD")

        def _scalar(v):
            return float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        while self.global_steps < total_steps:
            if self.mrsd_dataset.n_active == 0:
                print("[RLSDTrainer] 所有死区题目已毕业，提前结束")
                break

            batch = self.mrsd_dataset.sample_batch(self.problems_per_step)
            if not batch:
                continue

            step_metrics = self._rlsd_step(batch)
            self.global_steps += 1
            step_metrics["train/global_step"] = float(self.global_steps)
            step_metrics["train/n_active"] = float(self.mrsd_dataset.n_active)
            step_metrics["train/n_graduated"] = float(self.mrsd_dataset.n_graduated)

            logger.log(data=step_metrics, step=self.global_steps)
            progress.update(1)
            progress.set_postfix({
                "sd": int(_scalar(step_metrics.get("rlsd/n_sd_samples", 0))),
                "grpo": int(_scalar(step_metrics.get("rlsd/n_grpo_samples", 0))),
                "dead": int(_scalar(step_metrics.get("rlsd/n_dead_zone", 0))),
                "active": self.mrsd_dataset.n_active,
            })

            newly = self.mrsd_dataset.maybe_graduate_problems()
            if newly:
                logger.log({"train/newly_graduated": len(newly)}, step=self.global_steps)

            if self.global_steps % test_freq == 0:
                self._evaluate(step=self.global_steps, logger=logger)

            if self.global_steps % save_freq == 0:
                self._save_checkpoint()
                self.mrsd_dataset.save_state(str(ckpt_dir / f"rlsd_dataset_step{self.global_steps}.json"))

        progress.close()
        self._save_checkpoint()
        self.mrsd_dataset.save_state(str(ckpt_dir / f"rlsd_dataset_final.json"))

        stats = self.mrsd_dataset.stats()
        print(f"\n[RLSDTrainer] 完成  毕业: {stats['n_graduated']}/{stats['n_total']}  死区: {stats['n_active']}")
