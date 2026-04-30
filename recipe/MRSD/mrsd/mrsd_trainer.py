"""
MRSD Ray Trainer。

继承 RayPPOTrainer，只覆写 fit()。
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
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.tracking import Tracking

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from recipe.MRSD.mrsd.dataset import MRSDDataset, MRSDProblem
from recipe.MRSD.mrsd.prompt import build_student_messages, build_teacher_context_b
from recipe.MRSD.mrsd.verifier import is_correct


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _build_gen_batch(tokenizer, messages_list, max_prompt_len):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc = tokenizer(texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return DataProto.from_single_dict({"input_ids": ids, "attention_mask": mask, "position_ids": pos})


def _build_logprob_batch(tokenizer, messages_list, responses, max_prompt_len, max_resp_len):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc_p = tokenizer(prompt_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    tokenizer.padding_side = "right"
    enc_r = tokenizer(responses, return_tensors="pt", max_length=max_resp_len, truncation=True, padding=True, add_special_tokens=False)
    tokenizer.padding_side = "left"

    T_p = enc_p["input_ids"].shape[1]
    full_ids = torch.cat([enc_p["input_ids"], enc_r["input_ids"]], dim=1)
    full_mask = torch.cat([enc_p["attention_mask"], enc_r["attention_mask"]], dim=1)
    pos = (full_mask.cumsum(-1) - 1).clamp(min=0)
    responses_tensor = enc_r["input_ids"]

    B = len(messages_list)
    resp_mask = torch.zeros(B, full_ids.shape[1], dtype=torch.long)
    resp_mask[:, T_p:] = enc_r["attention_mask"]

    return DataProto.from_single_dict({
        "input_ids": full_ids, "attention_mask": full_mask, "position_ids": pos,
        "responses": responses_tensor, "response_mask": resp_mask,
    })


def _build_train_batch(tokenizer, student_msgs_list, teacher_resps, teacher_lps, max_prompt_len, max_resp_len, kl_clip):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in student_msgs_list]
    enc_p = tokenizer(prompt_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    tokenizer.padding_side = "right"
    enc_r = tokenizer(teacher_resps, return_tensors="pt", max_length=max_resp_len, truncation=True, padding=True, add_special_tokens=False)
    tokenizer.padding_side = "left"

    T_r = enc_r["input_ids"].shape[1]
    B = len(student_msgs_list)
    full_ids = torch.cat([enc_p["input_ids"], enc_r["input_ids"]], dim=1)
    full_mask = torch.cat([enc_p["attention_mask"], enc_r["attention_mask"]], dim=1)
    pos = (full_mask.cumsum(-1) - 1).clamp(min=0)
    responses_tensor = enc_r["input_ids"]
    response_mask = enc_r["attention_mask"].float()

    # 对齐 teacher_log_probs 到 T_r
    if teacher_lps.shape[1] > T_r:
        teacher_lps = teacher_lps[:, :T_r]
    elif teacher_lps.shape[1] < T_r:
        pad = torch.zeros(B, T_r - teacher_lps.shape[1], dtype=teacher_lps.dtype)
        teacher_lps = torch.cat([teacher_lps, pad], dim=1)

    data = DataProto.from_single_dict({
        "input_ids": full_ids, "attention_mask": full_mask, "position_ids": pos,
        "responses": responses_tensor, "response_mask": response_mask,
        "old_log_probs": teacher_lps.detach(),
        "advantages": response_mask,   # 等权重（MRSDPPOActor 用 old_log_probs 计算 KL）
    })
    data.meta_info["temperature"] = 1.0
    data.meta_info["mrsd_kl_clip"] = kl_clip
    data.meta_info["mrsd_mode"] = True
    return data


# ══════════════════════════════════════════════════════════════════════
# MRSDTrainer
# ══════════════════════════════════════════════════════════════════════

class MRSDTrainer(RayPPOTrainer):
    """
    继承 RayPPOTrainer，只覆写 fit()。
    init_workers / _save_checkpoint / _load_checkpoint 全部使用官方实现。
    """

    def __init__(self, *args, mrsd_dataset: Optional[MRSDDataset] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mrsd_dataset = mrsd_dataset

        mrsd_cfg = OmegaConf.select(self.config, "mrsd", default=OmegaConf.create({}))
        self.student_k = int(OmegaConf.select(mrsd_cfg, "student_rollout_per_problem", default=4))
        self.teacher_k = int(OmegaConf.select(mrsd_cfg, "teacher_rollout_per_error", default=4))
        self.kl_clip = float(OmegaConf.select(mrsd_cfg, "kl_clip", default=10.0))
        self.problems_per_step = int(OmegaConf.select(mrsd_cfg, "problems_per_step", default=8))
        self.graduation_interval = int(OmegaConf.select(mrsd_cfg, "graduation_interval", default=100))
        self.max_prompt_len = int(OmegaConf.select(self.config, "data.max_prompt_length", default=2048))
        self.max_resp_len = int(OmegaConf.select(self.config, "data.max_response_length", default=3072))

    # ──────────────────────────────────────────────────────────────────
    # 推理工具
    # ──────────────────────────────────────────────────────────────────

    def _generate(self, messages_list, n_samples):
        """对一批 messages 生成 n_samples 条回复，返回 list[list[str]]。"""
        # 将每条 prompt 重复 n_samples 次，批量送入 vllm
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

    def _compute_log_probs(self, messages_list, responses):
        """在 messages_list[i] 的 context 下计算 responses[i] 的 per-token log-probs。"""
        data = _build_logprob_batch(self.tokenizer, messages_list, responses, self.max_prompt_len, self.max_resp_len)
        data_padded, pad_size = pad_dataproto_to_divisor(data, self.actor_rollout_wg.world_size)
        out_padded = self.actor_rollout_wg.compute_log_prob(data_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)
        batch = out.batch
        for key in ("old_log_probs", "log_probs", "response_log_probs"):
            if key in batch:
                return batch[key]
        raise KeyError(f"compute_log_prob 返回 keys: {list(batch.keys())}")

    # ──────────────────────────────────────────────────────────────────
    # 单步训练
    # ──────────────────────────────────────────────────────────────────

    def _mrsd_step(self, problems: list[MRSDProblem]) -> dict:
        metrics = {}
        t0 = time.time()

        # Step 1: Student Rollout
        student_msgs = [build_student_messages(p.question) for p in problems]
        student_resps = self._generate(student_msgs, n_samples=self.student_k)
        wrong_per_prob = [
            [r for r in resps if not is_correct(r, p.ground_truth)]
            for p, resps in zip(problems, student_resps)
        ]
        metrics["mrsd/n_with_wrong"] = float(sum(1 for w in wrong_per_prob if w))

        # Step 2: Teacher Rollout
        teacher_msgs_flat, teacher_meta = [], []
        for pi, (prob, wrongs) in enumerate(zip(problems, wrong_per_prob)):
            for wrong in wrongs[:self.student_k]:
                teacher_msgs_flat.append(build_teacher_context_b(prob.question, wrong, prob.ground_truth))
                teacher_meta.append((pi, wrong, prob.ground_truth))

        if not teacher_msgs_flat:
            metrics["mrsd/n_valid_pairs"] = 0.0
            return metrics

        teacher_resps_flat = self._generate(teacher_msgs_flat, n_samples=self.teacher_k)

        # Step 3: 质量过滤
        valid_pairs = []
        for (pi, wrong, gt), t_resps in zip(teacher_meta, teacher_resps_flat):
            prob = problems[pi]
            for t_resp in t_resps:
                if is_correct(t_resp, gt):
                    valid_pairs.append((
                        build_student_messages(prob.question),
                        build_teacher_context_b(prob.question, wrong, gt),
                        t_resp,
                    ))

        metrics["mrsd/n_valid_pairs"] = float(len(valid_pairs))
        if not valid_pairs:
            metrics["mrsd/step_time_s"] = time.time() - t0
            return metrics

        s_msgs = [p[0] for p in valid_pairs]
        t_msgs = [p[1] for p in valid_pairs]
        t_resps = [p[2] for p in valid_pairs]

        # Step 4: Teacher log-probs（no_grad）
        teacher_lps = self._compute_log_probs(t_msgs, t_resps)

        # Step 5: KL Loss Update（student context + teacher tokens）
        train_data = _build_train_batch(
            self.tokenizer, s_msgs, t_resps, teacher_lps,
            self.max_prompt_len, self.max_resp_len, self.kl_clip,
        )
        # update_actor (fsdp_workers.py:620) 需要 global_token_num
        train_data.meta_info["global_token_num"] = (
            train_data.batch["attention_mask"].sum(dim=-1).tolist()
        )
        train_padded, pad_size = pad_dataproto_to_divisor(train_data, self.actor_rollout_wg.world_size)
        actor_out_padded = self.actor_rollout_wg.update_actor(train_padded)
        actor_out = unpad_dataproto(actor_out_padded, pad_size=pad_size)
        metrics.update(actor_out.meta_info.get("metrics", {}))

        # 更新 dataset 状态（毕业判断）
        for pi, (prob, s_resp) in enumerate(zip(problems, student_resps)):
            n_correct = sum(is_correct(r, prob.ground_truth) for r in s_resp)
            self.mrsd_dataset.update_problem_stats(
                prob.index,
                new_wrong_trajs=[r for r in s_resp if not is_correct(r, prob.ground_truth)],
                n_correct=n_correct,
                n_total=len(s_resp),
            )

        metrics["mrsd/step_time_s"] = time.time() - t0
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # 验证（pass@1 on val set）
    # ──────────────────────────────────────────────────────────────────

    def _evaluate(self, step: int, logger: Tracking) -> dict:
        """在 val_files 上跑 pass@1，打印并记录到 wandb。"""
        val_path = OmegaConf.select(self.config, "data.val_files")
        if not val_path:
            return {}

        print(f"\n[eval] step={step}  载入验证集: {val_path}")
        df = pd.read_parquet(val_path)

        # 每次评估最多用 64 题，避免占用过多时间
        max_val = int(OmegaConf.select(self.config, "mrsd.val_max_samples", default=64))
        df = df.head(max_val)

        # 构建 prompt messages
        messages_list = []
        ground_truths = []
        for _, row in df.iterrows():
            # prompt 字段已是 chat messages list
            msgs = row["prompt"] if isinstance(row["prompt"], list) else list(row["prompt"])
            gt = row["reward_model"]["ground_truth"]
            messages_list.append(msgs)
            ground_truths.append(gt)

        # 生成（greedy, n=1）：临时换用 val_kwargs 温度
        gen_batch = _build_gen_batch(self.tokenizer, messages_list, self.max_prompt_len)
        gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        # 使用 greedy（temperature=0）
        gen_padded.meta_info["do_sample"] = False
        gen_padded.meta_info["temperature"] = 0.0
        gen_padded.meta_info["top_p"] = 1.0
        gen_padded.meta_info["top_k"] = 1
        out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        responses = self.tokenizer.batch_decode(
            out.batch["responses"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        n_correct = sum(is_correct(r, gt) for r, gt in zip(responses, ground_truths))
        pass1 = n_correct / len(ground_truths)

        metrics = {
            "val/pass@1": pass1,
            "val/n_correct": float(n_correct),
            "val/n_total": float(len(ground_truths)),
        }
        logger.log(data=metrics, step=step)
        print(f"[eval] step={step}  pass@1={pass1:.3f}  ({n_correct}/{len(ground_truths)})\n")
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # 主训练循环（覆写官方 fit()）
    # ──────────────────────────────────────────────────────────────────

    def fit(self) -> None:
        """MRSD 训练循环，覆写 RayPPOTrainer.fit()。"""
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

        # ── step 0：训练前基线评估 ──────────────────────────────────
        print(f"\n[MRSDTrainer] 开始  total_steps={total_steps}  active={self.mrsd_dataset.n_active}")
        self._evaluate(step=0, logger=logger)

        progress = tqdm(total=total_steps, initial=self.global_steps, desc="MRSD")

        def _scalar(v):
            return float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        while self.global_steps < total_steps:
            if self.mrsd_dataset.n_active == 0:
                print("[MRSDTrainer] ✅ 所有死区题目已毕业，提前结束")
                break

            batch = self.mrsd_dataset.sample_batch(self.problems_per_step)
            if not batch:
                continue

            step_metrics = self._mrsd_step(batch)
            self.global_steps += 1
            step_metrics["train/global_step"] = float(self.global_steps)
            step_metrics["train/n_active"] = float(self.mrsd_dataset.n_active)
            step_metrics["train/n_graduated"] = float(self.mrsd_dataset.n_graduated)

            logger.log(data=step_metrics, step=self.global_steps)
            progress.update(1)
            progress.set_postfix({
                "kl": f"{_scalar(step_metrics.get('mrsd/kl_loss', 0)):.3f}",
                "pairs": int(_scalar(step_metrics.get("mrsd/n_valid_pairs", 0))),
                "active": self.mrsd_dataset.n_active,
            })

            newly = self.mrsd_dataset.maybe_graduate_problems()
            if newly:
                logger.log({"train/newly_graduated": len(newly)}, step=self.global_steps)

            # ── 定期评估 ──────────────────────────────────────────
            if self.global_steps % test_freq == 0:
                self._evaluate(step=self.global_steps, logger=logger)

            if self.global_steps % save_freq == 0:
                self._save_checkpoint()
                self.mrsd_dataset.save_state(str(ckpt_dir / f"mrsd_dataset_step{self.global_steps}.json"))

        progress.close()
        self._save_checkpoint()
        self.mrsd_dataset.save_state(str(ckpt_dir / f"mrsd_dataset_final.json"))

        stats = self.mrsd_dataset.stats()
        print(f"\n[MRSDTrainer] 完成  毕业: {stats['n_graduated']}/{stats['n_total']}  死区: {stats['n_active']}")
