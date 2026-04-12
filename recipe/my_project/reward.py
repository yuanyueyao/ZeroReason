# Copyright 2026 the verl recipe authors
#
# Custom RewardManager: register via @register and import this module before load_reward_manager.

from __future__ import annotations

import ast
import multiprocessing
import queue
import re
from ast import literal_eval
from collections import defaultdict

import torch

from recipe.my_project import code_validate_exec as _code_validate_exec

from verl import DataProto
from verl.workers.reward_manager import register


def _count_words(text: str) -> int:
    return len([t for t in re.split(r"\s+", text.strip()) if t])


def _score_word_closeness(n_words: int, target: int) -> float:
    """Scalar reward in [0, 1]: 1 when n_words == target, decays with normalized distance."""
    denom = max(target, 1)
    err = abs(n_words - target)
    return max(0.0, 1.0 - min(err / denom, 1.0))


@register("word_count")
class WordCountRewardManager:
    """
    Rule-based reward: encourages responses whose **word count** is close to a target.

    Target per sample (in order):

    1. ``non_tensor_batch["reward_model"]["target_word_count"]`` if set (explicit; avoids
       confusing GSM8K ``ground_truth`` answers with a word target).
    2. Else ``target_words`` from ``reward_model.reward_kwargs`` (see Hydra config).

    Implements the same ``__call__(data, return_dict)`` contract as ``NaiveRewardManager``.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        target_words: int = 80,
        **kwargs,
    ) -> None:
        # compute_score is passed by load_reward_manager; this manager does not use it.
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.target_words = int(kwargs.pop("target_words", target_words))
        self._unused_compute_score = compute_score

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)
        already_printed: dict[str, int] = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            rm = data_item.non_tensor_batch.get("reward_model")
            target = self.target_words
            if isinstance(rm, dict) and rm.get("target_word_count") is not None:
                try:
                    target = int(str(rm["target_word_count"]).strip())
                except (TypeError, ValueError):
                    pass

            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "word_count")

            n_words = _count_words(response_str)
            reward = _score_word_closeness(n_words, target)

            reward_tensor[i, valid_response_length - 1] = reward
            reward_extra_info["word_count"].append(n_words)
            reward_extra_info["target_words"].append(target)
            reward_extra_info["score"].append(reward)

            ds_key = str(data_source)
            if ds_key not in already_printed:
                already_printed[ds_key] = 0
            if already_printed[ds_key] < self.num_examine:
                already_printed[ds_key] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[target_words]", target, "[word_count]", n_words, "[reward]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor


_PYTHON_FENCE_BLOCK_RE = re.compile(
    r"```python\s*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_INPUT_FENCE_BLOCK_RE = re.compile(
    r"```input\s*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_code_func_and_input(response: str) -> tuple[str, str, bool]:
    """
    Parse model output aligned with the ``fit_competition`` prompt in ``ray_trainer.py``:

    1. **Exactly one** fenced block `` ```python `` … `` ``` `` — imports / helpers / ``def f(...)``.
    2. **Exactly one** fenced block `` ```input `` … `` ``` `` — comma-separated argument expressions
       for ``f(...)``, not the answer.

    Same strict contract as ``extract_output_fence``: if either fence appears zero times or more than
    once, parsing fails and returns ``("", "", False)`` (no second-``python`` fallback).
    """
    py_matches = _PYTHON_FENCE_BLOCK_RE.findall(response)
    in_matches = _INPUT_FENCE_BLOCK_RE.findall(response)
    if len(py_matches) != 1 or len(in_matches) != 1:
        return "", "", False
    func_code = py_matches[0].strip()
    input_code = in_matches[0].strip()
    if not func_code or not input_code:
        return "", "", False
    return func_code, input_code, True

def extract_code_func_and_input_from_data(data: DataProto, tokenizer) -> tuple[list[str], list[str], list[bool]]:
    """
    Extract the code and input from the data.
    """
    func_code_list = []
    input_code_list = []
    parse_ok_list = []
    for i in range(len(data)):
        data_item = data[i]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
        func_code, input_code, parse_ok = extract_code_func_and_input(response_str)
        func_code_list.append(func_code)
        input_code_list.append(input_code)
        parse_ok_list.append(parse_ok)
    return func_code_list, input_code_list, parse_ok_list

def validate_python_code_task(
    func_code: str,
    input_code: str,
    *,
    timeout_sec: float = 1.0,
    use_subprocess: bool = True,
) -> tuple[bool, str | None, str | None]:
    """
    Returns (success, error_message, captured_output).

    Subprocess path uses ``code_validate_exec._subprocess_entry`` so the child only imports a
    **stdlib-only** module (no torch). If the worker target lived in ``reward.py``, each spawn
    would import torch and often exceed a 1s timeout even for trivial code.
    """
    if not func_code.strip() or not input_code.strip():
        return False, "empty func_code or input_code", None
    try:
        compile(func_code, "<competition_func>", "exec")
    except SyntaxError as e:
        return False, f"syntax: {e}", None

    if not use_subprocess:
        return _code_validate_exec.validate_exec_and_call_with_capture(func_code, input_code)

    ctx = multiprocessing.get_context("spawn")
    result_q: multiprocessing.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_code_validate_exec._subprocess_entry,
        args=(result_q, func_code, input_code),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)
        return False, f"timeout ({timeout_sec}s)", None

    try:
        return result_q.get_nowait()
    except queue.Empty:
        code = proc.exitcode
        if code is not None and code != 0:
            return False, f"worker exit {code}", None
        return False, "worker produced no result", None


_OUTPUT_FENCE_BLOCK_RE = re.compile(
    r"```output\s*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_output_fence(response: str) -> tuple[str | None, bool]:
    """
    Parse `` ```output`` … `` ``` `` blocks.

    Returns ``(inner, has_any_fence, n_blocks)``.
    ``inner`` is set **only** when ``n_blocks == 1`` (exactly one output fence); otherwise
    ``inner`` is ``None`` so callers can penalize duplicate or missing blocks.
    """
    matches = _OUTPUT_FENCE_BLOCK_RE.findall(response)
    n = len(matches)
    if n == 0:
        return None, False
    if n > 1:
        return None, False
    return matches[0].strip(), True


def _eval_python_expr_syntax_ok(s: str) -> bool:
    """Whether ``s`` is a syntactically valid Python **eval** expression (covers literals, operators, etc.)."""
    s = s.strip()
    if not s:
        return False
    try:
        ast.parse(s, mode="eval")
        return True
    except SyntaxError:
        return False


def _outputs_semantically_equal(gt: str | None, pred: str) -> bool:
    """Compare model output string to reference (stdout capture), allowing strip / literal_eval equality."""
    if gt is None:
        return False
    ga, pb = gt.strip(), pred.strip()
    if ga == pb:
        return True
    try:
        return literal_eval(ga) == literal_eval(pb)
    except (ValueError, SyntaxError, TypeError):
        # TypeError: e.g. set literal with unhashable elements `{[1]}` parses but cannot be built
        return False


def _reward_A_from_B_correct_fraction(k: int, n: int) -> float:
    """
    validate 成功后：按 B 答对条数 k / n，在 **正确率 = 0.5** 时取最大值 1.0，两端 (全错/全对) 为 0。
    r = 1 - 4 * (k/n - 0.5)^2
    """
    if n <= 0:
        return 0.0
    p = k / n
    return max(0.0, 1.0 - 4.0 * (p - 0.5) ** 2)


@register("competition_reward")
class CompetitionRewardManager:
    """
    双模型 competition：``__call__(data_A, data_B, *, return_dict)``。
    校验集等单 batch 调用 ``__call__(data, *, return_dict)`` 时返回全零占位。
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
        self.num_examine = num_examine
        self.rollout_n = int(kwargs.pop("rollout_n", rollout_n))
        self.validation_timeout_sec = float(kwargs.pop("validation_timeout_sec", 1.0))
        self.validation_use_subprocess = bool(kwargs.pop("validation_use_subprocess", True))

    def __call__(self, data_A: DataProto, data_B: DataProto | None = None, *, return_dict: bool = False):
        n = self.rollout_n
        reward_tensor_A = torch.zeros_like(data_A.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        parse_arr = data_A.non_tensor_batch.get("parse_ok")
        validate_arr = data_A.non_tensor_batch.get("validate_ok")
        b_group_arr = data_A.non_tensor_batch.get("competition_b_group")

        # ---------- B：逐条得分；并统计每个 b_group 的答对条数 k ----------
        b_correct_by_group: dict[int, int] = defaultdict(int)
        num_groups = 0
        reward_tensor_B: torch.Tensor | None = None

        if data_B is None:
            # 本步无 B 任务：跳过 B 侧；A 侧 validate_ok 且第三档时 ra=0（gid 无效或 num_groups=0）
            pass
        else:
            rt_b = torch.zeros_like(data_B.batch["responses"], dtype=torch.float32)
            len_b = len(data_B)
            if len_b % n != 0:
                raise ValueError(
                    f"competition reward: len(data_B)={len_b} not divisible by rollout_n={n}"
                )
            num_groups = len_b // n
            gt_arr = data_B.non_tensor_batch.get("gt_output")
            if gt_arr is None or len(gt_arr) != len_b:
                raise ValueError("data_B.non_tensor_batch['gt_output'] missing or length mismatch")

            for j in range(len_b):
                item_b = data_B[j]
                prompt_len_b = item_b.batch["prompts"].shape[-1]
                resp_ids = item_b.batch["responses"]
                vlen = item_b.batch["attention_mask"][prompt_len_b:].sum()
                resp_str = self.tokenizer_B.decode(resp_ids[: int(vlen)], skip_special_tokens=False)

                inner, has_fence = extract_output_fence(resp_str)
                g_idx = j // n
                gt = gt_arr[j]

                if not has_fence:
                    r_b = -1.0
                elif inner is None or not _eval_python_expr_syntax_ok(inner):
                    r_b = -0.5
                elif _outputs_semantically_equal(gt if isinstance(gt, str) else str(gt), inner):
                    r_b = 1.0
                    b_correct_by_group[g_idx] += 1
                else:
                    r_b = 0.0

                if self.num_examine > 0 and j < self.num_examine:
                    print("================================================")
                    print("[B response]", resp_str)
                    print("[B inner]", inner)
                    print("[B gt]", gt)
                    print("[B has_fence]", has_fence)
                    print("[B reward]", r_b)
                    print("================================================")

                rt_b[j, int(vlen) - 1] = r_b
                reward_extra_info["B_reward"].append(r_b)
                reward_extra_info["B_has_output_fence"].append(has_fence)

            for g in range(num_groups):
                reward_extra_info["B_correct_count"].append(b_correct_by_group.get(g, 0))

            reward_tensor_B = rt_b

        # ---------- A ----------
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
                parse_ok = extract_code_func_and_input(response_str_A)[2]

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
                    fc, ic, _ = extract_code_func_and_input(response_str_A)
                    validate_ok = bool(
                        validate_python_code_task(
                            fc,
                            ic,
                            timeout_sec=self.validation_timeout_sec,
                            use_subprocess=self.validation_use_subprocess,
                        )[0]
                    )
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
                        k = 0
                        ra = 0.0
                    else:
                        k = b_correct_by_group.get(gid, 0)
                        ra = _reward_A_from_B_correct_fraction(k, n)
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