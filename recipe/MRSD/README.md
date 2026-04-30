# Multi-Round Self-Refinement Distillation (MRSD)

## 研究方案：在 pass@64=0 场景下无 COT 注释的 On-Policy 自蒸馏

---

## 1. 问题动机

DeepSeek-R1-Zero 之后，RLVR（以 GRPO 为代表）已成为提升模型推理能力的主流方法。其核心机制是：对同一问题采样多条轨迹，根据 reward 差异计算 advantage，反向传播更新模型。

**关键瓶颈**：当所有采样轨迹 reward 一致时，advantage 为零，梯度消失，模型无法更新。对于竞赛级数学题，模型在 pass@64 = 0 的题目上完全陷入这一死区——GRPO 在此场景下完全失效。

现有解法的局限：


| 方法               | 局限                                                                        |
| ---------------- | ------------------------------------------------------------------------- |
| OPSD             | 只给 final answer 作为 privileged context，对 pass@64=0 的题，conditioned 生成质量未经验证 |
| ReGFT            | 需要人工注释的参考 COT，额外标注成本高                                                     |
| Cog-DRIFT        | 改写题目格式，引入额外依赖，改变了原始任务分布                                                   |
| SFT on human COT | 完全 off-policy，covariate shift，catastrophic forgetting 风险高                 |


**本方法的核心约束**：

- ✅ 只使用 ground-truth final answer（无完整 COT）
- ✅ 无外部更强 teacher 模型
- ✅ On-policy（训练数据来自当前模型自身采样）
- ✅ 专门针对 pass@64=0 的死区问题

---

## 2. 核心方法：Multi-Round Self-Refinement Distillation（MRSD）

### 2.1 直觉

人类学生在做错一道题后，如果被告知正确答案，会对比自己的错误尝试，找到关键偏差，从而修正推理过程。MRSD 将这一直觉形式化为 on-policy self-distillation：

- **Teacher**：同一模型，context 包含 `(问题 + 自己的错误尝试 + 正确答案)` → 生成修正后的推理轨迹
- **Student**：同一模型，context 只有 `(问题)` → 在 inference 时的真实状态

**与 OPSD 的关键差异**：OPSD 的 teacher context 只有 `(问题 + answer)`；MRSD 的 teacher context 额外包含了 `(自己的错误轨迹)`，使得 teacher 能做"对比修正"，生成的轨迹更贴近 student 的实际推理空间，而非单纯从答案逆向推理。

### 2.2 训练流程

```
对每道 pass@64=0 的题目，执行以下步骤：

Step 1 [Student Rollout]
  从 π_student(· | 问题) 采样 k 条轨迹
  → 全部错误（pass@64=0 的定义）
  → 保留这些错误轨迹 {ŷ_1, ..., ŷ_k}

Step 2 [Teacher Rollout]
  对每条错误轨迹 ŷ_i，构造 teacher context：
    context_i = [问题] + [My previous attempt: ŷ_i] + [The correct answer is: y*] + [Let me reconsider:]
  从 π_teacher(· | context_i) 采样修正轨迹 ŷ'_i
  → teacher 和 student 是同一个模型，只是 context 不同

Step 3 [质量过滤]
  对 ŷ'_i 进行验证（rule-based verifier 验证最终答案）
  只保留最终答案正确的修正轨迹
  → 若过滤后为空，该题跳过（不参与本轮训练）

Step 4 [On-Policy Distillation]
  对保留的 (ŷ_i, ŷ'_i) 对，计算 per-token KL loss：
    L = -E_{ŷ ~ π_student} [ Σ_t log π_teacher(ŷ'_t | context_i, ŷ'_{<t}) ]
  梯度只通过 student 的 logits 传播

Step 5 [迭代]
  更新模型参数后，重新对所有 pass@64=0 的题采样
  动态追踪哪些题已从死区"毕业"（pass@64 > 0）
  已毕业的题切换到标准 GRPO 训练
```

### 2.3 Loss 形式

$$\mathcal{L}_{\text{MRSD}} = \mathbb{E}_{\hat{y} \sim \pi_\theta(\cdot|x)} \left[ D_{\text{KL}} \left( \pi_\theta(\cdot | x, \hat{y}, y^*) \;\|\; \pi_\theta(\cdot | x) \right) \right]$$

其中：

- $x$：问题
- $\hat{y}$：student 采样的错误轨迹（on-policy）
- $y^*$：ground-truth final answer
- $\pi_\theta(\cdot | x, \hat{y}, y^*)$：teacher 分布（同一模型，不同 context）
- 梯度仅通过右侧 $\pi_\theta(\cdot|x)$ 传播

### 2.4 Teacher Context 模板

```
[SYSTEM] You are a mathematical reasoning assistant.

[USER]
Problem: {问题}

My previous attempt (which was incorrect):
{错误轨迹 ŷ}

I was told the correct answer is: {y*}

Now let me carefully reconsider the problem and provide a correct step-by-step solution:
```

---

## 3. 前置诊断实验（必须先做）

在正式训练之前，需要验证核心假设是否成立。

### 3.1 假设验证实验

**目标**：确认 Qwen3-4B-instruct 在 conditioned context 下是否能生成合理轨迹

**步骤**：

1. 从数据集中随机抽取 30-50 道 pass@64=0 的题目
2. 分别测试两种 teacher context：
  - Context A（OPSD 风格）：`(问题 + answer)`
  - Context B（MRSD 风格）：`(问题 + 错误轨迹 + answer)`
3. 对每种 context 采样 16 条轨迹，用 verifier 检查最终答案正确率
4. **判断标准**：若 Context B 的正确率 > Context A，且 > 10%，则假设成立，方法可行

**预期结论**：

- 若大量题目 conditioned 正确率 = 0 → 说明 4B 模型知识盲区，需要换更大模型或更换数据集
- 若大量题目 conditioned 正确率 > 0 → 说明死区主要是搜索问题，MRSD 有效

### 3.2 题目分层

根据诊断结果，将 pass@64=0 的题目分为：


| 类型           | 定义                        | 处理方式      |
| ------------ | ------------------------- | --------- |
| Type-A（知识盲区） | conditioned 后仍然 pass@16=0 | 跳过，不参与训练  |
| Type-B（搜索死区） | conditioned 后 pass@16 > 0 | MRSD 训练目标 |


记录 Type-B 占总 pass@64=0 题目的比例，这本身是一个重要发现。

---

## 4. 数据集选择

### 推荐数据集

**主训练集**：DeepMath-103K（难度 Level 7-9 子集）

- 大规模、严格去污染
- 有 verifiable final answer（整数/数值）
- 专为 RLVR 场景设计

**主评测集**：OlymMATH-HARD

- 奥林匹克级别，o3-mini 仅 31.2%
- 数值答案可自动验证
- 去污染设计

**OOD 评测**：AIME 2025 + Beyond-AIME

- AIME 2025 污染风险低
- 与主流论文 baseline 对齐

### 数据筛选流程

```bash
# 1. 在 Qwen3-4B-instruct 上对训练集跑 pass@64
# 2. 筛选 pass@64=0 的题目子集（预计 Level 7-9 中占比较高）
# 3. 对筛选出的子集做 §3.1 的诊断实验
# 4. 保留 Type-B 题目作为 MRSD 训练集
```

---

## 5. 实验设计

### 5.1 Baseline 对比


| Baseline             | 描述                                  |
| -------------------- | ----------------------------------- |
| **GRPO（原始）**         | 在 pass@64=0 题上直接跑，验证零梯度现象           |
| **OPSD**             | 只给 answer，不给错误轨迹（对照实验）              |
| **SFT on human COT** | 使用数据集中的人工 COT，完全 off-policy（性能上界参考） |
| **MRSD（本方法）**        | 错误轨迹 + answer → teacher context     |


### 5.2 核心指标

**主指标（Coverage Gain）**：

- pass@64=0 的题目中，训练后有多少变为 pass@64>0
- 这是衡量"真实学习"还是"reranking"的关键指标

**辅助指标**：

- pass@1、pass@8 在评测集上的变化
- 训练 token efficiency（达到同等 pass@1 所需 tokens）
- pass@k 曲线（k=1,2,4,8,16,32,64）——验证是否扩展了 coverage 而非压缩了 diversity

**Anti-regression 指标**：

- 在原来 pass@64>0 的题目上，训练后是否出现性能下降
- 用于验证不引入 catastrophic forgetting

### 5.3 消融实验


| 消融                         | 目的                    |
| -------------------------- | --------------------- |
| 去掉错误轨迹（→ OPSD）             | 验证错误轨迹作为 context 的贡献  |
| 不做质量过滤（Step 3）             | 验证过滤的必要性              |
| 用 forward KL 替代 reverse KL | loss 形式的影响            |
| 不同 teacher context 模板      | prompt 设计的敏感性         |
| 不同 k（teacher 采样数量）         | teacher rollout 数量的影响 |


---

## 6. 实现细节

### 6.1 模型配置

```python
model = "Qwen/Qwen3-4B-Instruct"
# Teacher 和 Student 共享同一份权重
# Teacher forward pass 不计算梯度（torch.no_grad()）
# Student forward pass 计算梯度
```

### 6.2 训练超参数（初始建议）

```yaml
learning_rate: 1e-5
batch_size: 8  # 每道题
student_rollout_per_problem: 4  # k=4 条错误轨迹
teacher_rollout_per_error: 4    # 每条错误轨迹采样 4 条修正
max_new_tokens_student: 2048
max_new_tokens_teacher: 3072    # teacher 需要更长思考
kl_type: reverse_kl             # 参考 GKD 建议
kl_clip: true                   # 参考 OPSD 最新版，对 style token 做 point-wise 截断
training_steps: 500             # 先跑 500 steps 看趋势
eval_every: 50
```

### 6.3 关键工程细节

**per-token KL clipping**（参考 OPSD 最新 code release）：

- style token（如 `wait`、`think`、`\n`）的 KL 值可能比数学 token 高 6-15 倍
- 建议对 KL 值做 per-token clip，避免 style token 主导训练信号

**Teacher context 长度控制**：

- 错误轨迹可能很长，建议截断至最多 1024 tokens
- 截断策略：保留前 512 tokens + 后 512 tokens（保留开头的方向性错误和结尾的答案错误）

**动态课程**：

- 每 100 steps 重新评估哪些题已从 pass@64=0 "毕业"
- 毕业的题切换为标准 GRPO（利用 pass@k>0 的 reward 信号继续强化）

---

## 7. 预期贡献与 Novelty

### 核心 Novelty

1. **首个系统研究 pass@64=0 死区中"知识盲区"vs"搜索死区"比例的工作**
  - 诊断实验本身是 contribution
2. **错误轨迹作为 teacher context 的 on-policy 自蒸馏**
  - 不同于 OPSD（只给 answer）
  - 不同于 ReGFT（需要人工 COT）
  - 填补了"利用错误轨迹信息做自蒸馏"的空白
3. **与 majority voting 训练的对比**
  - 可以验证：MRSD 是否真正扩展了 pass@k ceiling，而不是 reranking

### 预期故事线

```
GRPO 在 pass@64=0 场景下完全失效
  ↓
诊断：死区分为知识盲区和搜索死区
  ↓
MRSD 针对搜索死区：利用错误轨迹+answer 做 on-policy 自蒸馏
  ↓
训练后 coverage gain：X% 的死区题目变为可解
  ↓
pass@k 曲线证明是真实学习（ceiling 提升），而非 reranking
  ↓
动态课程切换 GRPO 进一步强化已解锁的题目
```

---

## 8. 潜在风险与应对


| 风险                       | 可能性 | 应对                                 |
| ------------------------ | --- | ---------------------------------- |
| 4B 模型 conditioned 生成质量极差 | 中   | 先做诊断实验；备选换 7B 模型                   |
| Teacher 轨迹过滤后所剩无几        | 中   | 放宽过滤条件；增加 teacher 采样数              |
| KL 训练不稳定                 | 低-中 | 使用 per-token clip；降低 learning rate |
| 训练后其他题目性能下降              | 中   | 加入 KL penalty 约束与原始模型的距离           |
| Reviewer 质疑与 OPSD 的差异    | 中   | 消融实验明确量化错误轨迹的贡献                    |


---

## 9. 参考文献

- **GRPO / DeepSeek-R1-Zero**：Shao et al., 2024
- **OPSD**：Zhao et al., 2026 (arXiv:2601.18734)
- **GKD**：Agarwal et al., 2024 (arXiv:2306.13649)
- **ReGFT**：Wu et al., 2026 (arXiv:2603.01223)
- **Cog-DRIFT**：arXiv:2604.04767
- **Limit of RLVR**：Yue et al., 2025
- **SDFT**：Shenfeld et al., 2026 (arXiv:2601.19897)
- **DeepMath-103K**：He et al., 2025
- **OlymMATH**：Sun et al., 2025



---

## 10. 环境设置
采用 conda 环境：verl。

只可以修改recipe/MRSD目录下的文件，其他目录下的文件不要修改。

模型在/data3/yyy/models/Qwen3-4B-Base目录下。
数据在/data3/yyy/verl/data目录下。