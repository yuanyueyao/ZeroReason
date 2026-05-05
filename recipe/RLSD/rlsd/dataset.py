"""
MRSD 动态数据集。

核心功能：
  1. 管理 Type-B 问题池（死区题目）
  2. 追踪哪些题已"毕业"（pass@k > 0 → 切换到标准 GRPO）
  3. 每次 __getitem__ 返回 (question, ground_truth, difficulty, topic) 供 trainer 采样轨迹
  4. 支持从 pass@k 结果文件初始化（diagnostic 结果）
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from recipe.RLSD.rlsd.prompt import question_from_verl_prompt


class MRSDProblem:
    """代表一道 Type-B（搜索死区）问题的状态。"""

    __slots__ = (
        "index",
        "question",
        "ground_truth",
        "difficulty",
        "topic",
        "wrong_trajs",         # 诊断阶段收集的错误轨迹（初始 seed，后续动态更新）
        "n_correct_at_train",  # 训练过程中累计答对次数（用于毕业判断）
        "n_total_at_train",    # 训练过程中累计总采样次数
        "graduated",           # 是否已毕业（切换到 GRPO）
    )

    def __init__(
        self,
        index: int,
        question: str,
        ground_truth: str,
        difficulty: float = 5.0,
        topic: str = "",
        wrong_trajs: Optional[list[str]] = None,
    ):
        self.index = index
        self.question = question
        self.ground_truth = ground_truth
        self.difficulty = difficulty
        self.topic = topic
        self.wrong_trajs = wrong_trajs or []
        self.n_correct_at_train = 0
        self.n_total_at_train = 0
        self.graduated = False

    def update_stats(self, n_correct: int, n_total: int) -> None:
        self.n_correct_at_train += n_correct
        self.n_total_at_train += n_total

    @property
    def pass_at_1_estimate(self) -> float:
        if self.n_total_at_train == 0:
            return 0.0
        return self.n_correct_at_train / self.n_total_at_train

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "difficulty": self.difficulty,
            "topic": self.topic,
            "wrong_trajs": self.wrong_trajs[:2],  # 只序列化前2条
            "n_correct_at_train": self.n_correct_at_train,
            "n_total_at_train": self.n_total_at_train,
            "graduated": self.graduated,
        }


class MRSDDataset:
    """
    MRSD 动态数据集。

    Trainer 在每个 step：
      1. 调用 `sample_batch(n)` 取 n 道题
      2. 用 vllm 采样 student rollout（会出现错误答案）
      3. 构造 teacher context，再次用 vllm 采样
      4. 过滤后调用 `update_problem_stats(...)` 更新状态
      5. 每 `graduation_interval` steps 调用 `maybe_graduate_problems()` 清理死区
    """

    def __init__(
        self,
        problems: list[MRSDProblem],
        seed: int = 42,
        graduation_pass_at_k: int = 4,           # 毕业标准：pass@k（k 次中至少 1 次正确）
        graduation_interval: int = 100,            # 每隔多少 steps 重新评估毕业
        graduation_threshold: float = 0.0,         # pass_at_1 > threshold 则毕业（0=有1次对即毕业）
    ):
        self.problems = {p.index: p for p in problems}
        self.active_indices = [p.index for p in problems if not p.graduated]
        self.graduated_indices: list[int] = []
        self._rng = random.Random(seed)
        self.graduation_pass_at_k = graduation_pass_at_k
        self.graduation_interval = graduation_interval
        self.graduation_threshold = graduation_threshold
        self._step = 0

    # ──────────────────────────────────────────────────────────────────
    # 类方法：从诊断结果文件构建数据集
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_pass_at_k_results(
        cls,
        pass_at_k_jsonl: str,
        type_b_only: bool = True,
        **kwargs,
    ) -> "MRSDDataset":
        """
        从 run_pass_at_k.py 生成的 jsonl 构建数据集。
        type_b_only=True 时只载入 is_dead_zone=True 的题目（全部是 Type-B 候选）。
        """
        problems = []
        with open(pass_at_k_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if type_b_only and not rec.get("is_dead_zone", False):
                    continue
                problems.append(
                    MRSDProblem(
                        index=rec["index"],
                        question=rec["question"],
                        ground_truth=rec["ground_truth"],
                        difficulty=float(rec.get("difficulty", 5.0)),
                        topic=rec.get("topic", ""),
                        wrong_trajs=rec.get("wrong_trajs", []),
                    )
                )
        print(f"[MRSDDataset] 从 {pass_at_k_jsonl} 加载 {len(problems)} 道 Type-B 题目")
        return cls(problems=problems, **kwargs)

    @classmethod
    def from_context_ab_results(
        cls,
        type_b_jsonl: str,
        **kwargs,
    ) -> "MRSDDataset":
        """
        从 run_context_ab_test.py 生成的 type_b_problems.jsonl 构建数据集。
        包含已验证的 Type-B 问题，以及初始教师轨迹。
        """
        problems = []
        with open(type_b_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                problems.append(
                    MRSDProblem(
                        index=rec["index"],
                        question=rec["question"],
                        ground_truth=rec["ground_truth"],
                        difficulty=float(rec.get("difficulty", 5.0)),
                        topic=rec.get("topic", ""),
                        wrong_trajs=rec.get("wrong_trajs", []),
                    )
                )
        print(f"[MRSDDataset] 从 {type_b_jsonl} 加载 {len(problems)} 道 Type-B 题目（含错误轨迹）")
        return cls(problems=problems, **kwargs)

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
        **kwargs,
    ) -> "MRSDDataset":
        """
        从标准 verl parquet 格式加载（不含错误轨迹）。
        用于简单测试或从头构建数据集。
        """
        df = pd.read_parquet(parquet_path)
        problems = []
        for i, row in df.iterrows():
            prompt = row["prompt"]
            question = question_from_verl_prompt(
                prompt if isinstance(prompt, list) else list(prompt)
            )
            problems.append(
                MRSDProblem(
                    index=int(i),
                    question=question,
                    ground_truth=row["reward_model"]["ground_truth"],
                    difficulty=float(row["extra_info"].get("difficulty", 5.0)),
                    topic=str(row["extra_info"].get("topic", "")),
                )
            )
        print(f"[MRSDDataset] 从 {parquet_path} 加载 {len(problems)} 道题目")
        return cls(problems=problems, **kwargs)

    # ──────────────────────────────────────────────────────────────────
    # 访问接口
    # ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.active_indices)

    def sample_batch(self, n: int, replace: bool = False) -> list[MRSDProblem]:
        """随机抽取 n 道活跃题目（不放回采样，若 n > len 则 clamp）。"""
        active = self.active_indices
        if len(active) == 0:
            return []
        n = min(n, len(active))
        if replace:
            indices = self._rng.choices(active, k=n)
        else:
            indices = self._rng.sample(active, n)
        return [self.problems[i] for i in indices]

    def get_problem(self, index: int) -> Optional[MRSDProblem]:
        return self.problems.get(index)

    @property
    def n_active(self) -> int:
        return len(self.active_indices)

    @property
    def n_graduated(self) -> int:
        return len(self.graduated_indices)

    # ──────────────────────────────────────────────────────────────────
    # 状态更新
    # ──────────────────────────────────────────────────────────────────

    def update_problem_stats(
        self,
        index: int,
        new_wrong_trajs: Optional[list[str]] = None,
        n_correct: int = 0,
        n_total: int = 0,
    ) -> None:
        """
        训练后更新题目状态：
          - 追加新的错误轨迹（on-policy）
          - 更新正确率统计
        """
        prob = self.problems.get(index)
        if prob is None:
            return
        if new_wrong_trajs:
            # 保留最新的 4 条错误轨迹（on-policy，丢弃旧的 off-policy 轨迹）
            prob.wrong_trajs = (new_wrong_trajs + prob.wrong_trajs)[:4]
        prob.update_stats(n_correct, n_total)

    def maybe_graduate_problems(self, force: bool = False) -> list[int]:
        """
        每 graduation_interval steps 检查是否有题目应该毕业。
        毕业标准：在训练过程中 n_correct_at_train > 0（即不再是死区）。

        返回：本次新毕业的题目 index 列表。
        """
        self._step += 1
        if not force and (self._step % self.graduation_interval != 0):
            return []

        newly_graduated = []
        new_active = []
        for idx in self.active_indices:
            prob = self.problems[idx]
            # 毕业判断：累计正确次数 > 0
            if prob.n_correct_at_train > 0 and prob.n_total_at_train >= self.graduation_pass_at_k:
                prob.graduated = True
                self.graduated_indices.append(idx)
                newly_graduated.append(idx)
            else:
                new_active.append(idx)

        self.active_indices = new_active

        if newly_graduated:
            print(
                f"[MRSDDataset] step={self._step}  "
                f"新毕业 {len(newly_graduated)} 道题 → "
                f"活跃: {len(self.active_indices)}  已毕业: {len(self.graduated_indices)}"
            )

        return newly_graduated

    # ──────────────────────────────────────────────────────────────────
    # 序列化（checkpoint 用）
    # ──────────────────────────────────────────────────────────────────

    def save_state(self, path: str) -> None:
        state = {
            "active_indices": self.active_indices,
            "graduated_indices": self.graduated_indices,
            "step": self._step,
            "problems": {str(k): v.to_dict() for k, v in self.problems.items()},
        }
        with open(path, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self, path: str) -> None:
        with open(path) as f:
            state = json.load(f)
        self.active_indices = state["active_indices"]
        self.graduated_indices = state["graduated_indices"]
        self._step = state["step"]
        # 恢复统计数据
        for idx_str, pdata in state["problems"].items():
            idx = int(idx_str)
            if idx in self.problems:
                p = self.problems[idx]
                p.n_correct_at_train = pdata.get("n_correct_at_train", 0)
                p.n_total_at_train = pdata.get("n_total_at_train", 0)
                p.graduated = pdata.get("graduated", False)
                if pdata.get("wrong_trajs"):
                    p.wrong_trajs = pdata["wrong_trajs"]

    # ──────────────────────────────────────────────────────────────────
    # 调试
    # ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        from collections import Counter
        topic_dead = Counter(
            self.problems[i].topic for i in self.active_indices
        )
        return {
            "n_total": len(self.problems),
            "n_active": self.n_active,
            "n_graduated": self.n_graduated,
            "topic_distribution": dict(topic_dead.most_common(10)),
        }
