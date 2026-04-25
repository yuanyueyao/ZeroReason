# Copyright 2026 the verl recipe authors
"""Math challenger trainer: subclasses ``MyTrainer`` and overrides ``fit_competition``."""

from __future__ import annotations

import warnings
from collections import defaultdict, deque
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import postprocess_data
from verl.utils.tracking import Tracking

from recipe.math_challenger_solver.answer_extractor import AnswerExtractor
from recipe.math_challenger_solver.history_window import HistoryEntry, ProblemHistoryWindow
from recipe.math_challenger_solver.majority_vote import MajorityVoteLabeler
from recipe.math_challenger_solver.problem_parse import extract_challenger_problem, problem_content_ok
from recipe.math_challenger_solver.prompt import (
    SYSTEM_A,
    SYSTEM_A_NO_HISTORY,
    SYSTEM_B,
    build_user_prompt_challenger,
    build_user_prompt_challenger_no_history,
    build_user_prompt_solver,
)
from recipe.my_project.diversity import compute_group_diversity_penalty, compute_memory_similarity
from recipe.my_project.ray_trainer import (
    MyTrainer,
    assign_grpo_uids,
    compute_advantage,
    compute_response_mask,
)


def _clip(s: str, max_chars: int) -> str:
    s = s.replace("\r\n", "\n").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


class MathChallengerTrainer(MyTrainer):
    """
    Same resource layout as ``MyTrainer`` (pools A/B), but:

    - A emits only `` ```problem`` … `` ``` ``.
    - B has no fixed answer format; pseudo-label = majority of extracted answers.
    - A's user prompt includes a rolling history (no majority value exposed).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        win = int(OmegaConf.select(self.config, "math_challenger.history_window_size", default=10))
        self._problem_history = ProblemHistoryWindow(maxlen=win)
        self._seed_initial_problem_history_from_config()
        self._warned_rollout_n_one = False
        self._answer_extractor = AnswerExtractor()
        self._majority_labeler = MajorityVoteLabeler()
        # Reuse parent's diversity deque name; store **problem strings** per round for memory similarity
        self._diversity_problem_memory: deque[list[str]] = deque(maxlen=int(self.config.algorithm.get("diversity_memory_window", 10)))

    def _seed_initial_problem_history_from_config(self) -> None:
        raw = OmegaConf.select(self.config, "math_challenger.initial_history_gsm8k_parquet", default=None)
        if raw is None:
            return
        path = str(raw).strip()
        if not path or path.lower() in ("null", "none", "~"):
            return
        win = int(OmegaConf.select(self.config, "math_challenger.history_window_size", default=10))
        if win <= 0:
            return
        n_req = int(OmegaConf.select(self.config, "math_challenger.initial_history_num_problems", default=0))
        n_eff = min(max(n_req, 0), win)
        if n_eff <= 0:
            return
        seed_cfg = OmegaConf.select(self.config, "math_challenger.initial_history_seed", default=None)
        if seed_cfg is None or (isinstance(seed_cfg, str) and not str(seed_cfg).strip()):
            seed_i = None
        else:
            seed_i = int(seed_cfg)

        from recipe.math_challenger_solver.gsm8k_history_seed import load_gsm8k_problem_excerpts_from_parquet

        texts = load_gsm8k_problem_excerpts_from_parquet(path, n=n_eff, seed=seed_i)
        for t in texts:
            self._problem_history.append(HistoryEntry(problem_excerpt=t))
        print(
            f"[math_challenger] Seeded A problem history with {len(texts)} GSM8K question(s) from {path} "
            f"(window_max={win}, requested={n_req}, seed={seed_i!s})."
        )

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
        """Console: 出题正文、解题回复与多数伪标签（仅日志；伪标签仍不写入给 A 的历史文案）。"""
        n_groups_log = int(OmegaConf.select(self.config, "math_challenger.log_groups_per_step", default=2))
        n_a_log = int(OmegaConf.select(self.config, "math_challenger.log_A_samples_per_step", default=4))
        if n_groups_log <= 0 and n_a_log <= 0:
            return

        print(f"\n{'#' * 20} {step_label} — 样本日志 {'#' * 20}")

        bs_a = len(problem_texts)
        if n_a_log > 0:
            for i in range(min(n_a_log, bs_a)):
                print(f"\n--- [出题者 A] 样本 i={i} | parse_ok={parse_ok_list[i]} ---")
                if parse_ok_list[i] and problem_texts[i].strip():
                    print(_clip(problem_texts[i], 4000))
                else:
                    plen = gen_batch_output_A[i].batch["prompts"].shape[-1]
                    resp_ids = gen_batch_output_A[i].batch["responses"]
                    vlen = int(gen_batch_output_A[i].batch["attention_mask"][plen:].sum())
                    raw = self.tokenizer_A.decode(resp_ids[:vlen], skip_special_tokens=False)
                    print(_clip(raw, 2000))

        if n_groups_log > 0 and validated_tuples and gen_batch_output_B is not None:
            num_groups_b = len(validated_tuples)
            for g in range(min(n_groups_log, num_groups_b)):
                prob_g, a_idx = validated_tuples[g]
                j0 = g * rollout_n
                lab = maj_labels[j0] if j0 < len(maj_labels) else ""
                ok_mv = maj_valid[j0] if j0 < len(maj_valid) else False
                print(f"\n--- [解题者 B] 题组 g={g} (A 样本索引 {a_idx}) ---")
                print(f"[题目]\n{_clip(prob_g, 4000)}")
                print(f"[多数投票伪标签] {'(无效/全失败)' if not ok_mv else lab}")
                for k in range(rollout_n):
                    j = j0 + k
                    item_b = gen_batch_output_B[j]
                    plen = item_b.batch["prompts"].shape[-1]
                    resp_ids = item_b.batch["responses"]
                    vl = int(item_b.batch["attention_mask"][plen:].sum())
                    resp_str = self.tokenizer_B.decode(resp_ids[:vl], skip_special_tokens=False)
                    pred = self._answer_extractor.extract(resp_str)
                    print(f"  · 分支 k={k} | extract={pred!r} | 回复摘录:\n{_clip(resp_str, 800)}")
        print(f"\n{'#' * 60}\n")

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
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="MathChallenger")

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

        while self.global_steps < self.total_training_steps:
            if rollout_n == 1 and not self._warned_rollout_n_one:
                warnings.warn(
                    "math_challenger_solver: rollout.n=1 — majority vote degenerates to a single vote.",
                    UserWarning,
                    stacklevel=1,
                )
                self._warned_rollout_n_one = True

            print(f"\n========== Math challenger step {self.global_steps + 1} of {self.total_training_steps} ==========\n")
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
                        gen_batch_output_A = unpad_dataproto(gen_batch_output_A, pad_size_A)
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
                                gen_batch_output_B = unpad_dataproto(gen_batch_output_B, pad_size_B)
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
