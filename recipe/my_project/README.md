# my_project：双模型 Competition + GRPO（重点训练 B 的推理）

本 recipe 在 [verl](https://github.com/volcengine/verl) 上实现一条 **代码竞赛式** RL 流程：模型 **A** 负责出题（生成可执行的 `def f` 与 ```input```），用 **Python 解释器** 执行 `f` 得到 **ground-truth 输出**；模型 **B** 在同一套题目上做 **逐步推理**，并在 ```output``` 中给出预测。训练使用 **GRPO**（`algorithm.adv_estimator=grpo`），对 A、B 各自构造优势并更新策略，其中 **B 的奖励直接与「是否猜中解释器得到的真值」挂钩**，从而强化 **B 的推理与对齐真值能力**。

---

## 目标与数据流

| 阶段 | 说明 |
|------|------|
| **A：生成** | 按固定 system/user 模板采样，输出两段 fenced block：` ```python`（仅含 `def f`）与 ` ```input`（传给 `f` 的参数表达式）。 |
| **校验与真值** | 对解析成功的样本，在子进程里 `exec` 代码、`eval` 输入，调用 `f(*args)`，将 **返回值** 写入 `gt_output`（见 `code_validate_exec.py`）。 |
| **B：条件生成** | 将 A 的代码与 input 填入 B 的 prompt 模板，B 生成自然语言推理，并需在 ` ```output` 中给出可被解析的表达式。 |
| **奖励** | `CompetitionRewardManager`（`reward.py`）比较 B 的 output 与 `gt_output`（含 `literal_eval` 等价）；并对 A 按 B 组内答对比例等规则给分（详见该文件内注释与 `adv_estimator=grpo` 的 uid 设置）。 |

可选：配置中仍可使用 **GSM8K parquet** 等数据字段；主训练环 `fit_competition` 以 **A 的固定出题模板** 为主循环，**周期性**用 `val_reward_fn_gsm8k_b` 在验证集上评测 **B**（见 `ray_trainer.py` 中 `_my_validate` / `gsm8k_b_eval_freq`）。验证集可同时包含 **GSM8K** 与 **MBPP**（`data.val_files` 传两个 parquet）：`main_ppo.py` 里 `NaiveRewardManager` 使用 **`val_b_compute_score`**——`openai/gsm8k` 走规则匹配，`google-research-datasets/mbpp` 走 **`mbpp_exec`**（子进程执行 `test_setup_code` + 模型代码 + `test_list` / `challenge_test_list` 的 assert）。MBPP 数据由 `examples/data_preprocess/mbpp.py` 生成，且 **`extra_info.mbpp`** 与 `reward_model.mbpp` 对齐供评测使用。

---

## 目录结构（核心）

| 路径 | 作用 |
|------|------|
| `main_ppo.py` | Hydra 入口：`python -m recipe.my_project.main_ppo`，初始化 Ray 与 `MyTrainer`。 |
| `ray_trainer.py` | `MyTrainer.fit_competition()`：A→校验→B→reward→GRPO→optimizer；资源池 **pool_A / pool_B** 各需 **每节点 `trainer.n_gpus_per_node` 张 GPU**。 |
| `reward.py` | `@register("competition_reward")`：`CompetitionRewardManager`，双 batch `__call__(data_A, data_B, return_dict=True)`。 |
| `code_validate_exec.py` | 无 torch 依赖：子进程中执行用户代码并捕获 `f(...)` 返回值，供真值与日志。 |
| `prompt.py` | A 侧长 prompt 文案（如 `prompt_A_Qwen3_Base`）。 |
| `val_b_compute_score.py` / `mbpp_exec.py` | B 的验证打分：GSM8K + MBPP 执行评测。 |
| `config/ppo_trainer.yaml` | Hydra 默认配置；可通过命令行或 `run.sh` 覆盖。 |
| `run.sh` | 示例启动脚本（GPU、模型路径、GRPO、rollout `n`、checkpoint 等）。 |

---

## 环境与启动

1. 按仓库根目录说明安装 verl 与依赖（含 Ray、PyTorch、可选 vLLM），例如根目录 `README_verl.md` 或 `docs/` 下文档。
2. 准备与 `data.train_files` / `data.val_files` 一致的 **parquet**（字段需与 `RLHFDataset` 兼容；GSM8K 可用 `examples/data_preprocess/gsm8k.py` 生成）。
3. 在仓库根目录执行（示例）：

```bash
bash recipe/my_project/run.sh
```

或直接：

```bash
python3 -m recipe.my_project.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  actor_rollout_ref.model.path=/path/to/your/model \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1
```

**GPU 数量**：`resource_pool_spec` 为 **A 池 + B 池** 各 `n_gpus_per_node × nnodes`，通常至少需要 **2×** 单池卡数（例如每池 1 卡、共 2 卡）。请与 `CUDA_VISIBLE_DEVICES` 一致。

**显存**：同卡共置 **FSDP Actor + Ref + vLLM** 时较紧，可适当降低 `actor_rollout_ref.rollout.gpu_memory_utilization`、或开启 ref/param offload 等（见 `run.sh` 与上游 `ppo_trainer` 说明）。

---

## 配置要点

- **算法**：`algorithm.adv_estimator=grpo`，且需为同一 prompt 的 `rollout.n` 条样本设置 uid（trainer 内已处理）。
- **Rollout**：`actor_rollout_ref.rollout.n>1` 时，A/B 的 batch 行数与 GRPO 分组需一致（见 `reward.py` 与 `ray_trainer`）。
- **奖励**：`reward_model.reward_manager=competition_reward`（在 `main_ppo` 中通过 `CompetitionRewardManager` 注册与加载）。

更细的字段说明见 `config/ppo_trainer.yaml` 及 verl 上游 `verl/trainer/config/ppo_trainer.yaml`。

---

## 与 verl 主仓库的关系

本目录是 **recipe**：复用 `verl` 的 `ActorRolloutRefWorker`、`RayWorkerGroup`、`core_algos`（GRPO）等，仅在 **Role 扩展、双池调度、`fit_competition` 与 `CompetitionRewardManager`** 上定制。升级 verl 时请注意合并冲突，尤其是 `ResourcePoolManager` 与 `fsdp_workers` 行为变更。

---

## License

与仓库根目录的 **Apache 2.0** 及 `Notice.txt` 保持一致；recipe 内新增文件著作权见各文件头注释。
