# math_challenger_solver

本目录实现一条 **数学对抗式** 训练流程（对标 `recipe/my_project` 的代码竞赛形态）：**出题模型 A** 只生成题目，**解题模型 B** 多次作答；**不依赖 A 给出标准答案**——每道题的 **ground truth（伪标签）** 仅由 **B 的多数投票** 产生。实现代码位于本目录；**不修改** `verl` 主库文件。训练器通过 **继承** `recipe.my_project.ray_trainer.MyTrainer` 复用双池与 checkpoint 逻辑；验证集打分复用 `recipe.my_project.val_b_compute_score`（只读引用）。

---

## 目标

| 角色 | 职责 |
|------|------|
| **A（出题）** | 以 **`**Problem:**`**（或 `Problem:`）引出题面；解析优先取该段，其次兼容旧版唯一 ` ```problem` **fence**；**不**输出参考答案。 |
| **B（解题）** | 针对题目做推理；**不约定**输出版式，由通用逻辑从完整生成文本中 **抽取数字答案**。 |

训练信号：对同一道题的 `rollout.n` 条 B 回复抽取答案后 **多数投票** 得到该步 **伪标签** \(\hat{y}\)，用于对 B（及 A）构造奖励。该伪标签 **不会** 以数值形式写入给 A 的历史描述中。

---

## 与 `my_project`（代码竞赛）的差异

| 项目 | `my_project` | 本 recipe |
|------|----------------|-------------------|
| A 的输出 | ` ```python` + ` ```input`，真值由解释器执行得到 | **`**Problem:**` 段**（或 legacy ` ```problem` fence），**无**执行器真值 |
| B 的参考答案 | 与 `gt_output`（执行结果）比较 | 与 **多数票** 得到的 \(\hat{y}\) 比较 |
| B 的输出格式 | 约定 ` ```output` fence | **不约定**；通用 **extract answer（数字）** |

---

## 多数投票与伪标签

1. 对同一题、同一 GRPO 组内的 `n` 条 B 回复，分别 `extract` 得到答案字符串（失败记为「无」）。
2. 在 **至少有一条有效抽取** 的前提下，对有效答案做计数，**众数** 作为 \(\hat{y}\)（训练用 ground truth）。
3. **平票**：`MajorityVoteLabeler` 在并列最高票时取 **字典序最小** 的规范化键（见 `majority_vote.py`）。
4. **`rollout.n = 1`**：**允许**；此时「多数」退化为唯一一票，训练上仍自洽；启动或运行时 **仅警告**，不强制报错。

---

## 全部抽取失败时（「全失败」）

若某道题在本步 **没有任何一条** B 回复能抽出有效数字答案：

- **跳过**该题：本步 **不** 写入多数伪标签（`majority_valid=False`），B 侧奖励为 0；A 侧对应样本 **不** 因共识获得正奖励（`majority_valid_for_sample=False` 时 A 的共识项为 0）。
- **历史窗口**：该题 **不** 写入 `ProblemHistoryWindow`（避免在尚无有效多数伪标签时把该轮题面记入「已出题历史」）。

---

## A 侧：历史题目窗口（仅题面，无 B 侧信息）

- 维护长度上限为 `W` 的 **历史窗口**（配置项，如 5～20）。
- **每步**在 B 投票完成后，对 **本轮多数投票有效** 的题目，将其 **题面摘录** **append**（超出则丢弃最旧）；与「全失败」组仍 **不** 写入（与上节一致）。
- 构造下一轮 **A 的 user prompt** 时，窗口中 **仅** 列出既往题目的短摘录（`prior_problem`），**不** 包含：B 解析成功数、答案种类数、共识强弱、\(\hat{y}\) 等任何 **B 的作答或投票统计**。

---

## 答案抽取

- `answer_extractor.AnswerExtractor`：依次尝试 `\boxed{...}`、`Answer:` 行、分数 `a/b`、**最后一个** 类浮点数字串，并经 `normalize_answer_key` 规范化后参与投票。

---

## 代码结构（本目录）

| 文件 | 职责 |
|------|------|
| `answer_extractor.py` | `AnswerExtractor`、`normalize_answer_key` |
| `majority_vote.py` | `MajorityVoteLabeler`、`MajorityResult` |
| `history_window.py` | `ProblemHistoryWindow`、`HistoryEntry`（仅存题面摘录；无 B 侧统计、无 \(\hat{y}\)） |
| `gsm8k_history_seed.py` | 从 GSM8K parquet 读入题面字符串，供初始历史窗口 |
| `problem_parse.py` | 优先解析 `**Problem:**` …；否则唯一 ` ```problem` fence（兼容旧行为） |
| `prompt.py` | A/B 的 system 与 user 模板 |
| `reward.py` | `@register("math_majority_reward")` 的 `MathMajorityRewardManager` |
| `math_trainer.py` | `MathChallengerTrainer`：覆盖 `fit_competition()` |
| `main_ppo.py` | Hydra 入口、`TaskRunner` |
| `config/ppo_trainer.yaml` | 自 `my_project` 拷贝并增加根键 `math_challenger` |
| `run.sh` | 示例启动 |

伪标签与 `majority_valid` 在 **trainer** 中写入 `DataProto.non_tensor_batch`，`MathMajorityRewardManager` 对 B 再解码并 **再次** `extract` 以计算奖励与 `k`（与 README「避免重复解析」略有取舍，优先保证与训练解码一致）。

---

## 配置与启动

```bash
# 在仓库根目录；需双池 GPU（与 my_project 相同）
bash recipe/math_challenger_solver/run.sh
```

或直接：

```bash
python3 -m recipe.math_challenger_solver.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet \
  actor_rollout_ref.model.path=/path/to/model \
  trainer.n_gpus_per_node=1 trainer.nnodes=1
```

- `math_challenger.history_window_size`：A 侧历史条数上限（默认 `10`，见 yaml）。
- **GSM8K 初始历史**：`math_challenger.initial_history_gsm8k_parquet` 指向 verl 格式的 GSM8K RL parquet 时，在训练开始前用其中若干道 **仅题面**（`extra_info.question` / `prompt[0].content`）预填充 `ProblemHistoryWindow`，条数 ≤ `min(initial_history_num_problems, history_window_size)`。`initial_history_seed` 为 `null` 时按文件顺序取前 N 条；为整数时用该种子 **无放回随机** 抽 N 道。`run.sh` 默认从 `.../data/gsm8k/train.parquet` 预填；关闭：`GSM8K_INITIAL_HISTORY=none`。
- `math_challenger.log_groups_per_step` / `log_A_samples_per_step`：每步在控制台打印的 **B 题组数**与 **A 样本数**（默认 `2` / `4`）。日志中的「多数投票伪标签」仅供人类调试，**不会**写入给 A 的历史文案。
- `reward_model.num_examine`：奖励侧额外打印的样本条数（默认 `2`，见 yaml）；设为 `0` 可关闭。

若启动报错 `Please set at least one of 'actor_rollout_ref.actor.micro_batch_size' or '...micro_batch_size_per_gpu'`，请在命令行或 yaml 中为 **actor / ref / rollout** 设置 `ppo_micro_batch_size_per_gpu`（或 `log_prob_micro_batch_size_per_gpu`），`run.sh` 已包含一组与 `recipe/my_project/run.sh` 对齐的默认值。

### 初始化阶段 CUDA OOM（vLLM sampler warmup）

日志若出现 `warming up sampler with 1024 dummy requests` 或 `Please try lowering max_num_seqs or gpu_memory_utilization`：训练为 **hybrid_engine**，同一张卡上同时驻留 **FSDP Actor/Ref** 与 **vLLM**，默认 `max_num_seqs=1024` 时 vLLM 预热会占满显存。处理方式：

1. **优先** 降低 `actor_rollout_ref.rollout.max_num_seqs`（`run.sh` 已默认 `256`）。
2. 同时略降 `actor_rollout_ref.rollout.gpu_memory_utilization`（`run.sh` 与 `my_project` 一致为 `0.4`，可按机器再调低）。
3. 必要时再降 `max_num_batched_tokens`（`run.sh` 已默认 `4096`）。

仍 OOM 时可继续减小 `max_num_seqs`（如 `128`）、或降低 `rollout.n`、或减小 `data.max_prompt_length` / `max_response_length`。


---

## 约束

- **仅** 在 `recipe/math_challenger_solver/` 内新增或修改文件；不修改仓库其余路径，除非项目层面另有约定。

---

## License

与仓库根目录 **Apache 2.0** 及 `Notice.txt` 保持一致。
