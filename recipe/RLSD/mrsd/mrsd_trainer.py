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
from recipe.RLSD.mrsd.dataset import MRSDDataset, MRSDProblem
from recipe.RLSD.mrsd.prompt import build_student_messages, build_teacher_privileged_messages
from recipe.RLSD.mrsd.verifier import is_correct


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
    """构建仅用于 compute_log_prob 的 batch（单 prompt + response）。"""
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


def _build_grpo_train_batch(tokenizer, messages_list, responses_text, rewards,
                            old_log_probs, group_ids, max_prompt_len, max_resp_len):
    """
    构建 GRPO 分支训练 batch。

    与官方 verl GRPO 一致：按 group (problem) 内部做 advantage 归一化，
    advantages shape = (B, T_resp)，padding 位置为 0。

    Args:
        rewards: list[float], 每条 response 的 reward (0/1)
        group_ids: list[int], 每条 response 归属的 problem index（用于组内归一化）
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc_p = tokenizer(prompt_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    tokenizer.padding_side = "right"
    enc_r = tokenizer(responses_text, return_tensors="pt", max_length=max_resp_len,
                      truncation=True, padding=True, add_special_tokens=False)
    tokenizer.padding_side = "left"

    T_r = enc_r["input_ids"].shape[1]
    full_ids = torch.cat([enc_p["input_ids"], enc_r["input_ids"]], dim=1)
    full_mask = torch.cat([enc_p["attention_mask"], enc_r["attention_mask"]], dim=1)
    pos = (full_mask.cumsum(-1) - 1).clamp(min=0)
    responses_tensor = enc_r["input_ids"]
    response_mask = enc_r["attention_mask"]  # (B, T_r)

    # 按 problem (group) 内部归一化 advantage — 与官方 GRPO 一致
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    from collections import defaultdict
    g2scores = defaultdict(list)
    for i, gid in enumerate(group_ids):
        g2scores[gid].append(rewards_t[i])
    g2mean, g2std = {}, {}
    for gid, scores in g2scores.items():
        st = torch.stack(scores)
        g2mean[gid] = st.mean()
        g2std[gid] = st.std() if len(scores) > 1 else torch.tensor(1.0)

    normed = torch.zeros_like(rewards_t)
    for i, gid in enumerate(group_ids):
        normed[i] = (rewards_t[i] - g2mean[gid]) / (g2std[gid] + 1e-8)

    # expand to (B, T_resp) 并 mask padding
    advantages = normed.unsqueeze(1).expand(-1, T_r) * response_mask.float()

    # 对齐 old_log_probs 形状
    if old_log_probs.shape[1] > T_r:
        old_log_probs = old_log_probs[:, :T_r]
    elif old_log_probs.shape[1] < T_r:
        B = old_log_probs.shape[0]
        pad = torch.zeros(B, T_r - old_log_probs.shape[1], dtype=old_log_probs.dtype)
        old_log_probs = torch.cat([old_log_probs, pad], dim=1)

    return DataProto.from_single_dict({
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "position_ids": pos,
        "responses": responses_tensor,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
    })


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
        生成回复并返回 (grouped_texts, old_log_probs)。
        old_log_probs: (total_samples, T) 或 None。
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

        n = len(messages_list)
        grouped_texts = [all_texts[i * n_samples:(i + 1) * n_samples] for i in range(n)]
        return grouped_texts, old_log_probs

    def _compute_log_probs(self, messages_list, responses):
        """在 messages_list[i] 的 context 下计算 responses[i] 的 per-token log-probs。"""
        data = _build_logprob_batch(self.tokenizer, messages_list, responses, self.max_prompt_len, self.max_resp_len)
        data.meta_info["micro_batch_size"] = 4
        data.meta_info["temperature"] = 1.0
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

        # ── Step 1: Student Rollout ──────────────────────────────────
        student_msgs = [build_student_messages(p.question) for p in problems]
        student_resps_grouped, all_old_log_probs = self._generate_with_logprobs(
            student_msgs, n_samples=self.student_k
        )

        # ── Step 2: 计算 reward + 分类 ──────────────────────────────
        sd_student_msgs, sd_teacher_msgs, sd_resps = [], [], []
        grpo_msgs, grpo_resps, grpo_rewards, grpo_lp_indices, grpo_group_ids = [], [], [], [], []

        flat_idx = 0
        for pi, (prob, resps) in enumerate(zip(problems, student_resps_grouped)):
            correctness = [is_correct(r, prob.ground_truth) for r in resps]
            n_correct = sum(correctness)

            if n_correct == 0:
                # Dead zone → SD 分支：teacher 用特权 context (含 GT)
                for r in resps:
                    sd_student_msgs.append(build_student_messages(prob.question))
                    sd_teacher_msgs.append(build_teacher_privileged_messages(prob.question, prob.ground_truth))
                    sd_resps.append(r)
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

        print(f"  [rlsd] problems={len(problems)}  dead={n_dead}  mixed={n_mixed}  solved={n_solved}")
        sys.stdout.flush()

        # ── Step 3: SD 分支训练 ──────────────────────────────────────
        if sd_resps:
            sd_data = _build_sd_train_batch(
                self.tokenizer, sd_student_msgs, sd_teacher_msgs, sd_resps,
                self.max_prompt_len, self.max_resp_len
            )
            sd_data.meta_info["rlsd_mode"] = "sd"
            sd_data.meta_info["temperature"] = 1.0
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
        if grpo_resps and all_old_log_probs is not None:
            # 提取对应 index 的 old_log_probs
            lp_indices = torch.tensor(grpo_lp_indices, dtype=torch.long)
            grpo_old_lp = all_old_log_probs[lp_indices]

            grpo_data = _build_grpo_train_batch(
                self.tokenizer, grpo_msgs, grpo_resps, grpo_rewards,
                grpo_old_lp, grpo_group_ids, self.max_prompt_len, self.max_resp_len,
            )
            grpo_data.meta_info["rlsd_mode"] = "grpo"
            grpo_data.meta_info["temperature"] = 1.0
            grpo_data.meta_info["global_token_num"] = (
                grpo_data.batch["attention_mask"].sum(dim=-1).tolist()
            )
            grpo_padded, pad_size = pad_dataproto_to_divisor(grpo_data, self.actor_rollout_wg.world_size)
            grpo_out_padded = self.actor_rollout_wg.update_actor(grpo_padded)
            grpo_out = unpad_dataproto(grpo_out_padded, pad_size=pad_size)
            grpo_metrics = grpo_out.meta_info.get("metrics", {})
            for k, v in grpo_metrics.items():
                metrics[k] = v

        elif grpo_resps and all_old_log_probs is None:
            grpo_old_lp = self._compute_log_probs(grpo_msgs, grpo_resps)
            grpo_data = _build_grpo_train_batch(
                self.tokenizer, grpo_msgs, grpo_resps, grpo_rewards,
                grpo_old_lp, grpo_group_ids, self.max_prompt_len, self.max_resp_len,
            )
            grpo_data.meta_info["rlsd_mode"] = "grpo"
            grpo_data.meta_info["temperature"] = 1.0
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
    # 验证（pass@1 on dead zone set）
    # ──────────────────────────────────────────────────────────────────

    def _evaluate(self, step: int, logger: Tracking) -> dict:
        """在死区问题上跑 pass@1 greedy。"""
        val_path = OmegaConf.select(self.config, "data.mrsd_problems_path", default=None)
        is_jsonl = val_path and val_path.endswith(".jsonl")
        if not val_path:
            val_path = OmegaConf.select(self.config, "data.val_files")
        if not val_path:
            return {}

        print(f"\n[eval] step={step}  验证集: {val_path}")
        max_val = int(OmegaConf.select(self.config, "mrsd.val_max_samples", default=64))

        messages_list = []
        ground_truths = []

        if is_jsonl:
            import json
            with open(val_path) as f:
                for i, line in enumerate(f):
                    if i >= max_val:
                        break
                    r = json.loads(line)
                    messages_list.append(build_student_messages(r["question"]))
                    ground_truths.append(r["ground_truth"])
        else:
            df = pd.read_parquet(val_path)
            df = df.head(max_val)
            for _, row in df.iterrows():
                msgs = row["prompt"] if isinstance(row["prompt"], list) else list(row["prompt"])
                gt = row["reward_model"]["ground_truth"]
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

        n_correct = 0
        n_examine = min(3, len(responses))
        for i, (r, gt) in enumerate(zip(responses, ground_truths)):
            correct = is_correct(r, gt)
            if correct:
                n_correct += 1
            if i < n_examine:
                from recipe.RLSD.mrsd.verifier import extract_boxed_answer
                extracted = extract_boxed_answer(r) or "(none)"
                status = "+" if correct else "-"
                print(f"  [{status}] Q{i} gt={gt}  pred={extracted}")
                sys.stdout.flush()

        pass1 = n_correct / max(len(ground_truths), 1)
        metrics = {
            "val/pass@1": pass1,
            "val/n_correct": float(n_correct),
            "val/n_total": float(len(ground_truths)),
        }
        logger.log(data=metrics, step=step)
        print(f"[eval] step={step}  pass@1={pass1:.3f}  ({n_correct}/{len(ground_truths)})\n")
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
