# Copyright 2026 the verl recipe authors
"""Math challenger + judge: all logic in this recipe; does not modify math_challenger_solver."""

from __future__ import annotations

import math
import os
import warnings
from collections import defaultdict, deque
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import postprocess_data
from verl.utils.tracking import Tracking
from tqdm import tqdm

from recipe.math_challenger_solver.history_window import HistoryEntry
from recipe.math_challenger_solver.math_trainer import MathChallengerTrainer
from recipe.math_challenger_solver.problem_parse import extract_challenger_problem, problem_content_ok
from recipe.math_challenger_solver.prompt import (
    SYSTEM_A,
    SYSTEM_A_NO_HISTORY,
    SYSTEM_B,
    build_user_prompt_challenger,
    build_user_prompt_challenger_no_history,
    build_user_prompt_solver,
)
from recipe.math_challenger_solver_judge.judge_prompt import (
    SYSTEM_JUDGE_ANSWER,
    SYSTEM_JUDGE_PROBLEM,
    build_user_judge_answer,
    build_user_judge_problem,
)
from recipe.math_challenger_solver_judge.score_parse import (
    extract_score_tag,
    majority_vote_judge_score,
    normalize_score_to_01,
    rounded_score_bucket,
    score_in_range,
)
from recipe.my_project.diversity import compute_group_diversity_penalty, compute_memory_similarity
from recipe.my_project.ray_trainer import Role, assign_grpo_uids, compute_advantage, compute_response_mask


def _merge_judge_config(base_config):
    from omegaconf import OmegaConf, open_dict
    out = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    j = OmegaConf.select(base_config, "judge", default=None)
    if j is not None and OmegaConf.select(j, "actor_rollout_ref") is not None:
        with open_dict(out):
            out.actor_rollout_ref = OmegaConf.merge(
                OmegaConf.create({}),
                out.actor_rollout_ref,
                j.actor_rollout_ref,
            )
    return out


class MathChallengerJudgeTrainer(MathChallengerTrainer):
    """Same as MathChallengerTrainer plus third pool (judge) and inlined judge steps."""

    def __init__(self, *args, tokenizer_J=None, **kwargs):
        if tokenizer_J is None:
            raise TypeError("MathChallengerJudgeTrainer requires tokenizer_J=...")
        super().__init__(*args, **kwargs)
        # Apply B-specific lr override (declared in actor_rollout_ref_b.actor.optim.lr).
        # Must happen after super().__init__() which constructs self.config_B from the base config.
        _lr_b = OmegaConf.select(self.config, "actor_rollout_ref_b.actor.optim.lr", default=None)
        if _lr_b is not None:
            with open_dict(self.config_B):
                self.config_B.actor_rollout_ref.actor.optim.lr = float(_lr_b)
        self.tokenizer_J = tokenizer_J
        self.config_J = _merge_judge_config(self.config)
        if bool(OmegaConf.select(self.config, "judge.entropy_zero", default=False)):
            with open_dict(self.config_J):
                self.config_J.actor_rollout_ref.actor.entropy_coeff = 0.0
        self._judge_gen_merged = None
        self._judge_row_rewards: list[float] = []
        # 本步 judge 的日志快照（_run_judge_rollouts 写入，_log_rollouts_human 打印后清空）
        self._judge_log_snapshot: dict | None = None

    @staticmethod
    def _log_clip(s: str, max_chars: int) -> str:
        t = s.replace("\n", "↵")
        if len(t) <= max_chars:
            return t
        return t[: max_chars - 1] + "…"

    def _print_judge_state_line(self, step_label: str, *, n_valid_groups: int, bs_a: int, len_b: int) -> None:
        snap = self._judge_log_snapshot
        if not snap:
            return
        print(
            f"\n▶ {step_label}  [本步规模]  A_batch={bs_a}  有效题组={n_valid_groups}  B条数={len_b}  |  "
            f"Judge: jn={snap['j_n']}  合并序列数={snap['tot_judge_seq']}  |  "
            f"r̄={snap['r_all_mean']:.4f} (题面{snap['r_p_mean']:.4f} / 解{snap['r_a_mean']:.4f})"
        )

    def _print_judge_sample_details(self) -> None:
        snap = self._judge_log_snapshot
        if not snap or (not snap.get("problem_groups") and not snap.get("answer_rows")):
            return
        print(
            f"\n{'*' * 20} Judge(J) 输出 (字符上限={snap.get('log_max_chars', 480)}) {'*' * 20}"
        )
        for pg in snap.get("problem_groups", []):
            print(
                f"  [J·题面] group={pg['group']}  v_maj={pg['v_maj']}  mode={pg['mode_int']}  "
                f"→norm_A={pg['norm_to_A']:.3f}  题组B多票有效={pg['maj_B_ok']}"
            )
            for b in pg.get("branches", []):
                sc, p_ok = b["score"], b["parse_ok"]
                print(
                    f"      s={b['s']}  parse={p_ok}  score={sc!r}  r={b['r_line']:.3f} "
                    f"(base {b['r_base']:.2f} +多数 {b['r_maj_match']:.2f}) | {b['raw_reply']}"
                )
        for ar in snap.get("answer_rows", []):
            print(
                f"  [J·解答] B行={ar['B_index']}  题组={ar['group_B']}  v_maj={ar['v_maj']}  mode={ar['mode_int']}  "
                f"→norm_B={ar['norm_to_B']:.3f}  多票有效={ar['maj_B_ok']}"
            )
            for b in ar.get("branches", []):
                sc, p_ok = b["score"], b["parse_ok"]
                print(
                    f"      s={b['s']}  parse={p_ok}  score={sc!r}  r={b['r_line']:.3f} "
                    f"(base {b['r_base']:.2f} +多数 {b['r_maj_match']:.2f}) | {b['raw_reply']}"
                )
        print(f"{'*' * 72}\n")

    def _log_rollouts_human(
        self,
        *,
        gen_batch_output_A: DataProto,
        problem_texts: list[str],
        parse_ok_list: list[bool],
        validated_tuples: list[tuple[str, int]],
        gen_batch_output_B: DataProto | None,
        maj_labels: list[str],
        maj_valid: list[bool],
        rollout_n: int,
        step_label: str,
    ) -> None:
        nvg = len(validated_tuples)
        bs_a = len(problem_texts)
        len_b = len(gen_batch_output_B) if gen_batch_output_B is not None else 0
        self._print_judge_state_line(step_label, n_valid_groups=nvg, bs_a=bs_a, len_b=len_b)
        super()._log_rollouts_human(
            gen_batch_output_A=gen_batch_output_A,
            problem_texts=problem_texts,
            parse_ok_list=parse_ok_list,
            validated_tuples=validated_tuples,
            gen_batch_output_B=gen_batch_output_B,
            maj_labels=maj_labels,
            maj_valid=maj_valid,
            rollout_n=rollout_n,
            step_label=step_label,
        )
        self._print_judge_sample_details()
        self._judge_log_snapshot = None

    def init_workers(self) -> None:
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout_A)
        actor_rollout_cls_A = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout_A],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout_A"] = actor_rollout_cls_A
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout_B)
        actor_rollout_cls_B = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout_B],
            config=self.config_B.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout_B"] = actor_rollout_cls_B
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout_J)
        actor_rollout_cls_J = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout_J],
            config=self.config_J.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout_J"] = actor_rollout_cls_J
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy_A)
        ref_policy_cls_A = RayClassWithInitArgs(
            self.role_worker_mapping[Role.RefPolicy_A], config=self.config.actor_rollout_ref, role="ref"
        )
        self.resource_pool_to_cls[resource_pool]["ref_policy_A"] = ref_policy_cls_A
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy_B)
        ref_policy_cls_B = RayClassWithInitArgs(
            self.role_worker_mapping[Role.RefPolicy_B], config=self.config_B.actor_rollout_ref, role="ref"
        )
        self.resource_pool_to_cls[resource_pool]["ref_policy_B"] = ref_policy_cls_B
        _no_train_j = bool(OmegaConf.select(self.config, "judge.no_train", default=False))
        if not _no_train_j:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy_J)
            ref_policy_cls_J = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy_J], config=self.config_J.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref_policy_J"] = ref_policy_cls_J
        all_wg = {}
        wg_kwargs: dict = {}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        self.ref_policy_wg_A = all_wg["ref_policy_A"]
        self.ref_policy_wg_A.init_model()
        self.ref_policy_wg_B = all_wg["ref_policy_B"]
        self.ref_policy_wg_B.init_model()
        if not _no_train_j:
            self.ref_policy_wg_J = all_wg["ref_policy_J"]
            self.ref_policy_wg_J.init_model()
        else:
            self.ref_policy_wg_J = None
        self.actor_rollout_wg_A = all_wg["actor_rollout_A"]
        self.actor_rollout_wg_A.init_model()
        self.actor_rollout_wg_B = all_wg["actor_rollout_B"]
        self.actor_rollout_wg_B.init_model()
        self.actor_rollout_wg_J = all_wg["actor_rollout_J"]
        self.actor_rollout_wg_J.init_model()
        self.async_rollout_mode = False

    def _save_checkpoint_competition(self) -> None:
        from verl.utils.fs import local_mkdir_safe
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        print(f"save checkpoint (competition + judge): {local_global_step_folder}")
        local_mkdir_safe(local_global_step_folder)
        remove_prev = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        max_k = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_prev else 1
        a_loc = os.path.join(local_global_step_folder, "actor_A")
        b_loc = os.path.join(local_global_step_folder, "actor_B")
        a_rem = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor_A"
        )
        b_rem = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor_B"
        )
        self.actor_rollout_wg_A.save_checkpoint(a_loc, a_rem, self.global_steps, max_ckpt_to_keep=max_k)
        self.actor_rollout_wg_B.save_checkpoint(b_loc, b_rem, self.global_steps, max_ckpt_to_keep=max_k)
        _no_train_j = bool(OmegaConf.select(self.config, "judge.no_train", default=False))
        if not _no_train_j:
            j_loc = os.path.join(local_global_step_folder, "actor_J")
            j_rem = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor_J"
            )
            self.actor_rollout_wg_J.save_checkpoint(j_loc, j_rem, self.global_steps, max_ckpt_to_keep=max_k)
        torch.save(self.train_dataloader.state_dict(), os.path.join(local_global_step_folder, "data.pt"))
        torch.save(self._question_history.state_dict(), os.path.join(local_global_step_folder, "a_question_history.pt"))
        torch.save(
            [e.problem_excerpt for e in self._problem_history._q],
            os.path.join(local_global_step_folder, "math_problem_history.pt"),
        )
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        with open(os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint_competition(self) -> None:
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
        import torch
        if self.config.trainer.resume_mode == "disable":
            return
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("hdfs not implemented")
        ck = self.config.trainer.default_local_dir
        if not os.path.isabs(ck):
            ck = os.path.join(os.getcwd(), ck)
        if self.config.trainer.resume_mode == "auto":
            gsf = find_latest_ckpt_path(ck)
            if gsf is None:
                print("judge: no checkpoint, train from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str)
            assert "global_step_" in self.config.trainer.resume_from_path
            gsf = self.config.trainer.resume_from_path
            if not os.path.isabs(gsf):
                gsf = os.path.join(os.getcwd(), gsf)
        else:
            raise ValueError(self.config.trainer.resume_mode)
        self.global_steps = int(gsf.split("global_step_")[-1])
        dl = self.config.trainer.del_local_ckpt_after_load
        self.actor_rollout_wg_A.load_checkpoint(os.path.join(gsf, "actor_A"), del_local_after_load=dl)
        self.actor_rollout_wg_B.load_checkpoint(os.path.join(gsf, "actor_B"), del_local_after_load=dl)
        if not bool(OmegaConf.select(self.config, "judge.no_train", default=False)):
            jp = os.path.join(gsf, "actor_J")
            if os.path.isdir(jp):
                self.actor_rollout_wg_J.load_checkpoint(jp, del_local_after_load=dl)
        dlp = os.path.join(gsf, "data.pt")
        if os.path.exists(dlp):
            self.train_dataloader.load_state_dict(torch.load(dlp, weights_only=False))
        qhp = os.path.join(gsf, "a_question_history.pt")
        if os.path.exists(qhp):
            self._question_history.load_state_dict(torch.load(qhp, weights_only=False))
        php = os.path.join(gsf, "math_problem_history.pt")
        if os.path.exists(php):
            exs = torch.load(php, weights_only=False)
            if isinstance(exs, list):
                while len(self._problem_history._q) > 0:
                    self._problem_history._q.popleft()
                for t in exs:
                    self._problem_history.append(HistoryEntry(problem_excerpt=str(t)))

    @staticmethod
    def _judge_row_r(parse_ok: bool, in_range: bool, w_f: float, w_r: float) -> float:
        return w_f * (1.0 if parse_ok else 0.0) + w_r * (1.0 if in_range else 0.0)

    def _encode_judge_chats(self, system: str, user_texts: list[str], gts: list[str] | None = None) -> DataProto:
        gts = gts or [""] * len(user_texts)
        ids, masks, pids = [], [], []
        for u, gt in zip(user_texts, gts):
            raw = self.tokenizer_J.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": u}],
                add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            mi = self.tokenizer_J(raw, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = postprocess_data(
                input_ids=mi["input_ids"],
                attention_mask=mi["attention_mask"],
                max_length=self.config.data.get("max_prompt_length", 2048),
                pad_token_id=self.tokenizer_J.pad_token_id,
                left_pad=True, truncation="left",
            )
            pids.append(compute_position_id_with_mask(attention_mask))
            ids.append(input_ids)
            masks.append(attention_mask)
        gen = DataProto.from_single_dict({
            "input_ids": torch.cat(ids, dim=0),
            "attention_mask": torch.cat(masks, dim=0),
            "position_ids": torch.cat(pids, dim=0),
            "gt_output": np.array(gts, dtype=object),
        })
        gen.meta_info = {
            "eos_token_id": self.tokenizer_J.eos_token_id,
            "pad_token_id": self.tokenizer_J.pad_token_id,
            "recompute_log_prob": False,
        }
        return gen

    def _run_judge_rollouts(
        self,
        gen_batch_output_B: DataProto,
        validated_tuples: list,
        maj_valid: list[bool],
        rollout_n: int,
        bs_a: int,
        a_b_group: np.ndarray,
        timing_raw: dict,
    ) -> np.ndarray:
        jcfg = OmegaConf.select(self.config, "math_challenger_judge", default={}) or {}
        smin, smax = float(jcfg.get("score_min", 1.0)), float(jcfg.get("score_max", 10.0))
        w_f, w_r = float(jcfg.get("judge_format_weight", 0.5)), float(jcfg.get("judge_range_weight", 0.5))
        w_m = float(jcfg.get("judge_majority_match_weight", 0.2))
        log_n_prob = int(jcfg.get("log_judge_problem_groups", 2))
        log_n_ans = int(jcfg.get("log_judge_answer_rows", 2))
        log_max_c = int(jcfg.get("log_judge_chars_max", 480))
        j_n = int(OmegaConf.select(self.config, "judge.actor_rollout_ref.rollout.n", default=4) or 4)
        j_n = max(1, j_n)
        num_groups = len(validated_tuples)
        len_b = len(gen_batch_output_B)
        dp_j = self.actor_rollout_wg_J.world_size
        log_prob_snap: list[dict] = []
        log_ans_snap: list[dict] = []
        user_probs = [build_user_judge_problem(validated_tuples[g][0]) for g in range(num_groups)]
        gen_p = self._encode_judge_chats(SYSTEM_JUDGE_PROBLEM, user_probs)
        divisor_p = math.lcm(dp_j, j_n) if j_n > 1 else dp_j
        with marked_timer("time/judge/generate_problem", timing_raw, color="magenta"):
            gpp, pz = pad_dataproto_to_divisor(gen_p, divisor_p)
            out_p = self.actor_rollout_wg_J.generate_sequences(gpp)
            out_p.batch["response_mask"] = compute_response_mask(out_p)
            if pz:
                out_p = unpad_dataproto(out_p, pz * j_n)
        assert len(out_p) == num_groups * j_n, f"judge problem: {len(out_p)} != {num_groups}*{j_n}"
        problem_norm_by_g: list[float] = []
        r_p: list[float] = []
        for g in range(num_groups):
            vals, oks = [], []
            dec_p: list[str] = []
            p_rows: list = []
            for s in range(j_n):
                it = out_p[g * j_n + s]
                pl, resp = it.batch["prompts"].shape[-1], it.batch["responses"]
                vl = int(it.batch["attention_mask"][pl:].sum())
                text = self.tokenizer_J.decode(resp[:vl], skip_special_tokens=False)
                dec_p.append(text)
                pr = extract_score_tag(text)
                inr = score_in_range(pr.value, score_min=smin, score_max=smax, parse_ok=pr.parse_ok)
                p_rows.append((self._judge_row_r(pr.parse_ok, inr, w_f, w_r), pr))
                vals.append(pr.value)
                oks.append(pr.parse_ok)
            v_maj, mode_int = majority_vote_judge_score(
                vals, oks, score_min=smin, score_max=smax
            )
            gr_r: list[float] = []
            for s in range(j_n):
                base, pr = p_rows[s]
                bonus = 0.0
                if w_m > 0.0 and mode_int is not None and j_n > 1:
                    bi = rounded_score_bucket(
                        pr.value, parse_ok=pr.parse_ok, score_min=smin, score_max=smax
                    )
                    if bi is not None and bi == mode_int:
                        bonus = w_m
                gr_r.append(base + bonus)
            r_p.extend(gr_r)
            parse_maj = v_maj is not None
            pn = (
                normalize_score_to_01(v_maj, score_min=smin, score_max=smax, parse_ok=parse_maj)
                if v_maj is not None
                else 0.0
            )
            if not bool(maj_valid[g * rollout_n]):
                pn = 0.0
            problem_norm_by_g.append(pn)
            if log_n_prob > 0 and g < log_n_prob:
                br: list[dict] = []
                for s in range(j_n):
                    base, pr = p_rows[s]
                    bonus = float(gr_r[s] - base)
                    br.append(
                        {
                            "s": s,
                            "raw_reply": self._log_clip(dec_p[s], log_max_c),
                            "score": pr.value,
                            "parse_ok": pr.parse_ok,
                            "r_line": float(gr_r[s]),
                            "r_base": float(base),
                            "r_maj_match": bonus,
                        }
                    )
                log_prob_snap.append(
                    {
                        "kind": "problem",
                        "group": g,
                        "v_maj": v_maj,
                        "mode_int": mode_int,
                        "norm_to_A": float(pn),
                        "maj_B_ok": bool(maj_valid[g * rollout_n]),
                        "branches": br,
                    }
                )
        prob_norm = np.zeros(bs_a, dtype=np.float32)
        for i in range(bs_a):
            g = int(a_b_group[i]) if i < len(a_b_group) else -1
            if 0 <= g < num_groups:
                prob_norm[i] = float(problem_norm_by_g[g])
        user_ans: list[str] = []
        for j in range(len_b):
            g = j // rollout_n
            pt = validated_tuples[g][0]
            b_it = gen_batch_output_B[j]
            bpl = b_it.batch["prompts"].shape[-1]
            bvl = int(b_it.batch["attention_mask"][bpl:].sum())
            st = self.tokenizer_B.decode(b_it.batch["responses"][:bvl], skip_special_tokens=False)
            user_ans.append(build_user_judge_answer(pt, st))
        gen_a = self._encode_judge_chats(SYSTEM_JUDGE_ANSWER, user_ans)
        divisor_a = math.lcm(dp_j, j_n) if j_n > 1 else dp_j
        with marked_timer("time/judge/generate_answer", timing_raw, color="magenta"):
            gpa, az = pad_dataproto_to_divisor(gen_a, divisor_a)
            out_a = self.actor_rollout_wg_J.generate_sequences(gpa)
            out_a.batch["response_mask"] = compute_response_mask(out_a)
            if az:
                out_a = unpad_dataproto(out_a, az * j_n)
        assert len(out_a) == len_b * j_n, f"judge answer: {len(out_a)} != {len_b}*{j_n}"
        ans = np.zeros(len_b, dtype=np.float32)
        r_a: list[float] = []
        for j in range(len_b):
            vals, oks = [], []
            dec_a: list[str] = []
            ab_rows: list = []
            for s in range(j_n):
                it = out_a[j * j_n + s]
                pl, resp = it.batch["prompts"].shape[-1], it.batch["responses"]
                vl = int(it.batch["attention_mask"][pl:].sum())
                text = self.tokenizer_J.decode(resp[:vl], skip_special_tokens=False)
                dec_a.append(text)
                pr = extract_score_tag(text)
                inr = score_in_range(pr.value, score_min=smin, score_max=smax, parse_ok=pr.parse_ok)
                ab_rows.append((self._judge_row_r(pr.parse_ok, inr, w_f, w_r), pr))
                vals.append(pr.value)
                oks.append(pr.parse_ok)
            v_maj, mode_int = majority_vote_judge_score(
                vals, oks, score_min=smin, score_max=smax
            )
            gr_a: list[float] = []
            for s in range(j_n):
                base, pr = ab_rows[s]
                bonus = 0.0
                if w_m > 0.0 and mode_int is not None and j_n > 1:
                    bi = rounded_score_bucket(
                        pr.value, parse_ok=pr.parse_ok, score_min=smin, score_max=smax
                    )
                    if bi is not None and bi == mode_int:
                        bonus = w_m
                gr_a.append(base + bonus)
            r_a.extend(gr_a)
            parse_maj = v_maj is not None
            an = (
                normalize_score_to_01(v_maj, score_min=smin, score_max=smax, parse_ok=parse_maj)
                if v_maj is not None
                else 0.0
            )
            if not bool(maj_valid[(j // rollout_n) * rollout_n]):
                an = 0.0
            ans[j] = an
            if log_n_ans > 0 and j < log_n_ans:
                br2: list[dict] = []
                for s in range(j_n):
                    base, pr = ab_rows[s]
                    br2.append(
                        {
                            "s": s,
                            "raw_reply": self._log_clip(dec_a[s], log_max_c),
                            "score": pr.value,
                            "parse_ok": pr.parse_ok,
                            "r_line": float(gr_a[s]),
                            "r_base": float(base),
                            "r_maj_match": float(gr_a[s] - base),
                        }
                    )
                log_ans_snap.append(
                    {
                        "kind": "answer",
                        "B_index": j,
                        "group_B": j // rollout_n,
                        "v_maj": v_maj,
                        "mode_int": mode_int,
                        "norm_to_B": float(an),
                        "maj_B_ok": bool(maj_valid[(j // rollout_n) * rollout_n]),
                        "branches": br2,
                    }
                )
        gen_batch_output_B.non_tensor_batch["judge_score_answer_norm"] = ans
        self._judge_row_rewards = r_p + r_a
        self._judge_gen_merged = DataProto.concat([out_p, out_a])
        tot_j = (num_groups + len_b) * j_n
        self._judge_log_snapshot = {
            "j_n": j_n,
            "num_problem_groups": num_groups,
            "len_B": len_b,
            "tot_judge_seq": tot_j,
            "r_p_mean": float(np.mean(r_p)) if r_p else 0.0,
            "r_a_mean": float(np.mean(r_a)) if r_a else 0.0,
            "r_all_mean": float(np.mean(self._judge_row_rewards)) if self._judge_row_rewards else 0.0,
            "smin": smin,
            "smax": smax,
            "problem_groups": log_prob_snap,
            "answer_rows": log_ans_snap,
            "log_max_chars": log_max_c,
        }
        return prob_norm

    def _maybe_update_judge(self, has_b: bool, timing_raw: dict) -> None:
        if bool(OmegaConf.select(self.config, "judge.no_train", default=False)):
            self._judge_gen_merged = None
            self._judge_row_rewards = []
            return
        if not has_b or self._judge_gen_merged is None or not self._judge_row_rewards:
            return
        n_j = len(self._judge_gen_merged)
        if n_j != len(self._judge_row_rewards):
            return
        gen_j, rj = self._judge_gen_merged, self._judge_row_rewards
        rt = torch.zeros_like(gen_j.batch["responses"], dtype=torch.float32)
        for i in range(n_j):
            it = gen_j[i]
            pl = it.batch["prompts"].shape[-1]
            vl = int(it.batch["attention_mask"][pl:].sum())
            if vl > 0:
                rt[i, vl - 1] = float(rj[i])
        dp_j = self.actor_rollout_wg_J.world_size
        j_grpo_n = int(OmegaConf.select(self.config, "judge.actor_rollout_ref.rollout.n", default=4) or 4)
        j_grpo_n = max(1, j_grpo_n)
        pad_div = math.lcm(dp_j, j_grpo_n) if j_grpo_n > 1 else dp_j
        gen_j_pad, pad_j = pad_dataproto_to_divisor(gen_j, pad_div)
        if pad_j:
            # pad_dataproto 在末尾用已有行块复制补齐；对 DP 补行不施加 PPO 信号，避免对重复行更新。
            rt_pad = torch.zeros_like(gen_j_pad.batch["responses"], dtype=torch.float32)
            rt_pad[:n_j] = rt
            rt = rt_pad
        with marked_timer("time/old_log_prob_J", timing_raw, color="purple"):
            batch_J = gen_j_pad.union(self.actor_rollout_wg_J.compute_log_prob(gen_j_pad))
        assert self.ref_policy_wg_J is not None, "ref_policy_wg_J should not be None when judge.no_train=False"
        with marked_timer("time/ref_judge", timing_raw, color="orange"):
            batch_J = batch_J.union(self.ref_policy_wg_J.compute_ref_log_prob(batch_J))
        # 不在此 unpad：len 须保持为 dp_j 的倍数，供 compute_log_prob 与 update_actor 的 chunk 一致（否则 unpad 后 63%2 在 update 再失败）。
        # 同一题面/同一 B 的 j_grpo_n 条 judge 样本共享 GRPO 组
        jn = j_grpo_n
        if len(batch_J) % jn != 0:
            raise ValueError(
                f"judge PPO: batch {len(batch_J)} not divisible by judge.rollout.n={jn} "
                f"(use lcm padding or fix concat order)."
            )
        batch_J.batch["token_level_scores"] = rt
        batch_J.batch["token_level_rewards"] = rt
        assign_grpo_uids(batch_J, jn)
        with marked_timer("time/adv_judge", timing_raw, color="red"):
            batch_J = compute_advantage(
                batch_J,
                adv_estimator=self.config.algorithm.adv_estimator,
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            )
        with marked_timer("time/update_judge", timing_raw, color="orange"):
            batch_J.meta_info["global_token_num"] = torch.sum(batch_J.batch["attention_mask"], dim=-1).tolist()
            self.actor_rollout_wg_J.update_actor(batch_J)
        self._judge_gen_merged = None
        self._judge_row_rewards = []


    def fit_competition(self):
        """Train A/B with majority-vote pseudo labels and optional problem-text diversity penalty."""
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.total_training_steps = self.config.trainer.get("total_training_steps", 600)
        self._load_checkpoint_competition()

        metrics: dict = {}
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="MathChallengerJudge")

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._my_validate()
            assert val_metrics, f"{val_metrics=}"
            print(f"Initial B validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        dp_size_a = self.actor_rollout_wg_A.world_size
        dp_size_b = self.actor_rollout_wg_B.world_size
        _ = self.actor_rollout_wg_J.world_size

        while self.global_steps < self.total_training_steps:
            if rollout_n == 1 and not self._warned_rollout_n_one:
                warnings.warn(
                    "math_challenger_solver: rollout.n=1 — majority vote degenerates to a single vote.",
                    UserWarning,
                    stacklevel=1,
                )
                self._warned_rollout_n_one = True

            _sn = self.global_steps + 1
            _st = self.total_training_steps
            print(
                f"\n========== MathChallenger+Judge  step {_sn}/{_st}  |  "
                f"A/B rollouts={rollout_n}  pool_A={dp_size_a}g  pool_B={dp_size_b}g  pool_J={self.actor_rollout_wg_J.world_size}g "
                f"==========\n"
            )
            timing_raw: dict = {}

            use_problem_history = bool(
                OmegaConf.select(self.config, "math_challenger.use_problem_history", default=True)
            )
            if use_problem_history:
                history_block = self._problem_history.format_for_user_prompt()
                user_a = build_user_prompt_challenger(history_block)
                system_a = SYSTEM_A
            else:
                user_a = build_user_prompt_challenger_no_history()
                system_a = SYSTEM_A_NO_HISTORY
            raw_prompt_A = self.tokenizer_A.apply_chat_template(
                [{"role": "system", "content": system_a}, {"role": "user", "content": user_a}],
                add_generation_prompt=True,
                tokenize=False,
            )
            model_inputs_A = self.tokenizer_A(raw_prompt_A, return_tensors="pt", add_special_tokens=False)
            input_ids_A = model_inputs_A.pop("input_ids")
            attention_mask_A = model_inputs_A.pop("attention_mask")
            input_ids_A, attention_mask_A = postprocess_data(
                input_ids=input_ids_A,
                attention_mask=attention_mask_A,
                max_length=self.config.data.get("max_prompt_length", 2048),
                pad_token_id=self.tokenizer_A.pad_token_id,
                left_pad=True,
                truncation="left",
            )
            position_ids_A = compute_position_id_with_mask(attention_mask_A)
            gen_batch_A = DataProto.from_single_dict(
                {"input_ids": input_ids_A, "attention_mask": attention_mask_A, "position_ids": position_ids_A}
            )
            gen_batch_A.meta_info = {
                "eos_token_id": self.tokenizer_A.eos_token_id,
                "pad_token_id": self.tokenizer_A.pad_token_id,
                "recompute_log_prob": False,
            }
            gen_batch_padded_A, pad_size_A = pad_dataproto_to_divisor(gen_batch_A, dp_size_a)

            with marked_timer("time/step", timing_raw, color="red"):
                with marked_timer("time/generate_sequences_A", timing_raw, color="blue"):
                    gen_batch_output_A = self.actor_rollout_wg_A.generate_sequences(gen_batch_padded_A)
                    gen_batch_output_A.batch["response_mask"] = compute_response_mask(gen_batch_output_A)
                    if pad_size_A:
                        gen_batch_output_A = unpad_dataproto(gen_batch_output_A, pad_size_A * rollout_n)
                    texts_a = self.tokenizer_A.batch_decode(gen_batch_output_A.batch["responses"], skip_special_tokens=True)
                    metrics["A/response_preview"] = texts_a

                with marked_timer("time/parse_problem", timing_raw, color="green"):
                    bs_a = len(gen_batch_output_A)
                    problem_texts: list[str] = []
                    parse_ok_list: list[bool] = []
                    for idx in range(bs_a):
                        data_item = gen_batch_output_A[idx]
                        prompt_ids = data_item.batch["prompts"]
                        prompt_length = prompt_ids.shape[-1]
                        response_ids = data_item.batch["responses"]
                        vlen = data_item.batch["attention_mask"][prompt_length:].sum()
                        resp_str = self.tokenizer_A.decode(response_ids[: int(vlen)], skip_special_tokens=False)
                        body, ok = extract_challenger_problem(resp_str)
                        problem_texts.append(body)
                        parse_ok_list.append(ok)

                    gen_batch_output_A.non_tensor_batch["parse_ok"] = np.array(parse_ok_list, dtype=bool)
                    validate_ok_row = [
                        bool(parse_ok_list[i] and problem_texts[i].strip() and problem_content_ok(problem_texts[i]))
                        for i in range(bs_a)
                    ]
                    gen_batch_output_A.non_tensor_batch["validate_ok"] = np.array(validate_ok_row, dtype=bool)

                    validated_tuples: list[tuple[str, int]] = []
                    for idx in range(bs_a):
                        if not validate_ok_row[idx]:
                            continue
                        validated_tuples.append((problem_texts[idx], idx))

                    a_b_group = np.full(bs_a, -1, dtype=np.int64)
                    for g, (_p, a_idx) in enumerate(validated_tuples):
                        a_b_group[a_idx] = g
                    gen_batch_output_A.non_tensor_batch["competition_b_group"] = a_b_group

                    metrics["A/problem_parse_ok_count"] = int(np.sum(parse_ok_list))
                    metrics["A/problem_validate_ok_count"] = int(np.sum(validate_ok_row))
                    _meta_rej = sum(
                        1
                        for i in range(bs_a)
                        if parse_ok_list[i] and problem_texts[i].strip() and not problem_content_ok(problem_texts[i])
                    )
                    metrics["A/problem_meta_rejected"] = float(_meta_rej)
                    has_b = len(validated_tuples) > 0
                    maj_labels: list[str] = []
                    maj_valid: list[bool] = []

                with marked_timer("time/prepare_data_B", timing_raw, color="green"):
                    tensors_B: dict = defaultdict(list)
                    non_tensors_B: dict = defaultdict(list)
                    for prob, _a_idx in validated_tuples:
                        user_b = build_user_prompt_solver(prob)
                        prompt_B = self.tokenizer_B.apply_chat_template(
                            [{"role": "system", "content": SYSTEM_B}, {"role": "user", "content": user_b}],
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        model_inputs_B = self.tokenizer_B(prompt_B, return_tensors="pt", add_special_tokens=False)
                        input_ids_B = model_inputs_B.pop("input_ids")
                        attention_mask_B = model_inputs_B.pop("attention_mask")
                        input_ids_B, attention_mask_B = postprocess_data(
                            input_ids=input_ids_B,
                            attention_mask=attention_mask_B,
                            max_length=self.config.data.get("max_prompt_length", 2048),
                            pad_token_id=self.tokenizer_B.pad_token_id,
                            left_pad=True,
                            truncation="left",
                        )
                        position_ids_B = compute_position_id_with_mask(attention_mask_B)
                        tensors_B["input_ids"].append(input_ids_B)
                        tensors_B["attention_mask"].append(attention_mask_B)
                        tensors_B["position_ids"].append(position_ids_B)
                        non_tensors_B["prompt"].append(prompt_B)
                        non_tensors_B["gt_output"].append("")

                    gen_batch_output_B = None
                    if has_b:
                        gen_batch_B = DataProto.from_single_dict(
                            {
                                "input_ids": torch.cat(tensors_B["input_ids"], dim=0),
                                "attention_mask": torch.cat(tensors_B["attention_mask"], dim=0),
                                "position_ids": torch.cat(tensors_B["position_ids"], dim=0),
                                "prompt": np.array(non_tensors_B["prompt"], dtype=object),
                                "gt_output": np.array(non_tensors_B["gt_output"], dtype=object),
                            }
                        )
                        gen_batch_B.meta_info = {
                            "eos_token_id": self.tokenizer_B.eos_token_id,
                            "pad_token_id": self.tokenizer_B.pad_token_id,
                            "recompute_log_prob": False,
                        }
                        gen_batch_padded_B, pad_size_B = pad_dataproto_to_divisor(gen_batch_B, dp_size_b)
                        with marked_timer("time/generate_sequences_B", timing_raw, color="blue"):
                            gen_batch_output_B = self.actor_rollout_wg_B.generate_sequences(gen_batch_padded_B)
                            gen_batch_output_B.batch["response_mask"] = compute_response_mask(gen_batch_output_B)
                            if pad_size_B:
                                gen_batch_output_B = unpad_dataproto(gen_batch_output_B, pad_size_B * rollout_n)
                            texts_b = self.tokenizer_B.batch_decode(gen_batch_output_B.batch["responses"], skip_special_tokens=True)
                            metrics["B/response_preview"] = texts_b
                            metrics["B/response_count"] = len(texts_b)

                        with marked_timer("time/majority_vote", timing_raw, color="cyan"):
                            len_b = len(gen_batch_output_B)
                            num_groups_b = len_b // rollout_n
                            assert len_b == num_groups_b * rollout_n
                            maj_labels = []
                            maj_valid = []
                            distinct_counts: list[float] = []
                            for g in range(num_groups_b):
                                preds: list[Optional[str]] = []
                                for k in range(rollout_n):
                                    j = g * rollout_n + k
                                    item_b = gen_batch_output_B[j]
                                    plen = item_b.batch["prompts"].shape[-1]
                                    resp_ids = item_b.batch["responses"]
                                    vl = item_b.batch["attention_mask"][plen:].sum()
                                    resp_str = self.tokenizer_B.decode(resp_ids[: int(vl)], skip_special_tokens=False)
                                    preds.append(self._answer_extractor.extract(resp_str))
                                vote = self._majority_labeler.label(preds)
                                ok_g = vote.label is not None
                                lab = str(vote.label) if ok_g and vote.label is not None else ""
                                for _ in range(rollout_n):
                                    maj_labels.append(lab)
                                    maj_valid.append(ok_g)
                                distinct_counts.append(float(len({p for p in preds if p is not None})))
                            if distinct_counts:
                                metrics["math/vote_distinct_mean"] = float(np.mean(distinct_counts))
                            gen_batch_output_B.non_tensor_batch["majority_label"] = np.array(maj_labels, dtype=object)
                            gen_batch_output_B.non_tensor_batch["majority_valid"] = np.array(maj_valid, dtype=bool)

                        mv_for_sample = np.zeros(bs_a, dtype=bool)
                        for a_i in range(bs_a):
                            gid = int(a_b_group[a_i])
                            if 0 <= gid < num_groups_b:
                                mv_for_sample[a_i] = bool(maj_valid[gid * rollout_n])
                        gen_batch_output_A.non_tensor_batch["majority_valid_for_sample"] = mv_for_sample

                        # History: only when enabled; only successful majority (skip all-fail groups)
                        if use_problem_history:
                            for g in range(num_groups_b):
                                if not maj_valid[g * rollout_n]:
                                    continue
                                prob_g, _a_idx_g = validated_tuples[g]
                                self._problem_history.append(HistoryEntry(problem_excerpt=prob_g))
                        metrics["fit_competition/skip_B_step"] = 0.0
                    else:
                        metrics["fit_competition/skip_B_step"] = 1.0
                        gen_batch_output_A.non_tensor_batch["majority_valid_for_sample"] = np.zeros(bs_a, dtype=bool)

                if not has_b or not validated_tuples:
                    gen_batch_output_A.non_tensor_batch["judge_score_problem_norm"] = np.zeros(bs_a, dtype=np.float32)
                    self._judge_gen_merged = None
                    self._judge_row_rewards = []
                else:
                    prob_norm = self._run_judge_rollouts(
                        gen_batch_output_B,
                        validated_tuples,
                        maj_valid,
                        rollout_n,
                        bs_a,
                        a_b_group,
                        timing_raw,
                    )
                    gen_batch_output_A.non_tensor_batch["judge_score_problem_norm"] = prob_norm

                _step_tag = f"global_step={self.global_steps + 1}/{self.total_training_steps}"
                if has_b:
                    self._log_rollouts_human(
                        gen_batch_output_A=gen_batch_output_A,
                        problem_texts=problem_texts,
                        parse_ok_list=parse_ok_list,
                        validated_tuples=validated_tuples,
                        gen_batch_output_B=gen_batch_output_B,
                        maj_labels=maj_labels,
                        maj_valid=maj_valid,
                        rollout_n=rollout_n,
                        step_label=_step_tag,
                    )
                else:
                    self._log_rollouts_human(
                        gen_batch_output_A=gen_batch_output_A,
                        problem_texts=problem_texts,
                        parse_ok_list=parse_ok_list,
                        validated_tuples=[],
                        gen_batch_output_B=None,
                        maj_labels=[],
                        maj_valid=[],
                        rollout_n=rollout_n,
                        step_label=_step_tag,
                    )

                with marked_timer("time/reward", timing_raw, color="green"):
                    reward_AB = self.reward_fn(
                        gen_batch_output_A,
                        gen_batch_output_B if has_b else None,
                        return_dict=True,
                    )
                    reward_tensor_A = reward_AB["reward_tensor_A"]
                    reward_tensor_B = reward_AB.get("reward_tensor_B")
                    extra_merged = reward_AB["reward_extra_info"]
                    metrics.update({f"AB/{k}": v for k, v in reduce_metrics(deepcopy(extra_merged)).items()})

                # Diversity penalty on problem text (optional)
                _div_coeff = float(self.config.algorithm.get("diversity_penalty_coeff", 0.0))
                if _div_coeff > 0.0:
                    with marked_timer("time/diversity_penalty", timing_raw, color="cyan"):
                        _div_method = str(self.config.algorithm.get("diversity_penalty_method", "jaccard"))
                        _div_kwargs = dict(self.config.algorithm.get("diversity_penalty_kwargs") or {})
                        _rollout_n = self.config.actor_rollout_ref.rollout.n
                        _div_num_examine: int = getattr(self.reward_fn, "num_examine", 0)

                        problem_list = [problem_texts[i] if parse_ok_list[i] else "" for i in range(bs_a)]
                        _prompt_len_A = gen_batch_output_A.batch["prompts"].shape[-1]
                        _resp_lengths_A = gen_batch_output_A.batch["attention_mask"][:, _prompt_len_A:].sum(dim=-1).long()

                        _past_problems: list[str] = [p for rnd in self._diversity_problem_memory for p in rnd]
                        _mem_sim_all = compute_memory_similarity(problem_list, _past_problems, method=_div_method, **_div_kwargs)

                        _all_penalties: list[float] = []
                        for _g_idx, _g_start in enumerate(range(0, bs_a, _rollout_n)):
                            _g_end = min(_g_start + _rollout_n, bs_a)
                            _group_probs = problem_list[_g_start:_g_end]
                            _group_pen = compute_group_diversity_penalty(_group_probs, method=_div_method, **_div_kwargs)
                            _mem_sim_slice = _mem_sim_all[_g_start:_g_end]
                            _combined_pen = [gp * ms for gp, ms in zip(_group_pen, _mem_sim_slice)]
                            _all_penalties.extend(_combined_pen)

                        for _i, (_pen, _vlen) in enumerate(zip(_all_penalties, _resp_lengths_A)):
                            if _vlen > 0:
                                reward_tensor_A[_i, int(_vlen) - 1] -= _div_coeff * _pen

                        metrics["A/diversity_penalty/mean"] = float(np.mean(_all_penalties)) if _all_penalties else 0.0
                    self._diversity_problem_memory.append(list(problem_list))
                else:
                    self._diversity_problem_memory.append([problem_texts[i] if parse_ok_list[i] else "" for i in range(bs_a)])

                batch_B = None
                with marked_timer("time/old_log_prob", timing_raw, color="purple"):
                    old_log_prob_A = self.actor_rollout_wg_A.compute_log_prob(gen_batch_output_A)
                    batch_A = gen_batch_output_A.union(old_log_prob_A)
                    if has_b:
                        assert gen_batch_output_B is not None
                        old_log_prob_B = self.actor_rollout_wg_B.compute_log_prob(gen_batch_output_B)
                        batch_B = gen_batch_output_B.union(old_log_prob_B)

                with marked_timer("time/ref", timing_raw, color="orange"):
                    ref_log_prob_A = self.ref_policy_wg_A.compute_ref_log_prob(batch_A)
                    batch_A = batch_A.union(ref_log_prob_A)
                    if has_b:
                        ref_log_prob_B = self.ref_policy_wg_B.compute_ref_log_prob(batch_B)
                        batch_B = batch_B.union(ref_log_prob_B)

                with marked_timer("time/adv", timing_raw, color="red"):
                    batch_A.batch["token_level_scores"] = reward_tensor_A
                    batch_A.batch["token_level_rewards"] = reward_tensor_A
                    assign_grpo_uids(batch_A, rollout_n)
                    batch_A = compute_advantage(
                        batch_A,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    )
                    if has_b:
                        assert reward_tensor_B is not None
                        batch_B.batch["token_level_scores"] = reward_tensor_B
                        batch_B.batch["token_level_rewards"] = reward_tensor_B
                        assign_grpo_uids(batch_B, rollout_n)
                        batch_B = compute_advantage(
                            batch_B,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        )

                with marked_timer("time/update_actor", timing_raw, color="orange"):
                    batch_A.meta_info["global_token_num"] = torch.sum(batch_A.batch["attention_mask"], dim=-1).tolist()
                    self.actor_rollout_wg_A.update_actor(batch_A)
                    if has_b:
                        batch_B.meta_info["global_token_num"] = torch.sum(batch_B.batch["attention_mask"], dim=-1).tolist()
                        self.actor_rollout_wg_B.update_actor(batch_B)
                    self._maybe_update_judge(has_b, timing_raw)

                metrics.update(timing_raw)
                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1
                progress_bar.update(1)

                gsm8k_every = self._gsm8k_b_eval_interval()
                if gsm8k_every > 0 and self.global_steps % gsm8k_every == 0:
                    timing_gsm8k: dict = {}
                    with marked_timer("time/gsm8k_b_validate", timing_gsm8k, color="cyan"):
                        gsm8k_metrics = self._my_validate()
                    if gsm8k_metrics:
                        log_payload = dict(gsm8k_metrics)
                        log_payload.update({f"timing_s/{k}": v for k, v in timing_gsm8k.items()})
                        logger.log(data=log_payload, step=self.global_steps)

                save_freq = self.config.trainer.get("save_freq", -1)
                do_save = save_freq > 0 and (self.global_steps % save_freq == 0 or self.global_steps >= self.total_training_steps)
                if do_save:
                    with marked_timer("time/save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint_competition()

        progress_bar.close()