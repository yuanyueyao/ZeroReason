# RLSD: Bridging the Dead Zone of Verifiable Reinforcement Learning via On-Policy Self-Distillation

## 研究方案：RLVR + 自蒸馏的统一框架——解决零梯度死区

---

## 1. 问题动机

DeepSeek-R1-Zero 之后，RLVR（以 GRPO 为代表）已成为提升模型推理能力的主流方法。其核心机制是：对同一问题采样多条轨迹，根据 reward 差异计算 advantage，反向传播更新模型。

**关键瓶颈**：当组内所有采样轨迹 reward 一致（全对或全错）时，advantage 为零，梯度消失，模型无法更新。对于难题，模型 pass@1=0，在这些题目上 GRPO 产生零梯度——训练完全失效。

现有解法的局限：

| 方法               | 局限                                                                        |
| ---------------- | ------------------------------------------------------------------------- |
| OPSD             | 只做蒸馏，不利用 reward 信号；对已有 reward 的题目浪费了 RL 信号                               |
| ReGFT            | 需要人工注释的参考 COT，额外标注成本高                                                     |
| Cog-DRIFT        | 改写题目格式，引入额外依赖，改变了原始任务分布                                                   |
| SFT on human COT | 完全 off-policy，covariate shift，catastrophic forgetting 风险高                 |

**本方法（RLSD）的核心思路**：

- 有 reward signal 的题（组内有对有错）→ 正常做 **RLVR**（GRPO），利用 reward 差异优化
- 无 reward signal 的题（组内全错）→ 做 **On-Policy Self-Distillation**，用 frozen reference model 提供学习信号

两者统一在同一训练循环中，**无额外外部依赖**：

- ✅ 只使用 ground-truth final answer（无完整 COT）
- ✅ 无外部更强 teacher 模型（reference = 训练前初始 checkpoint）
- ✅ On-policy（训练数据来自当前模型自身采样）
- ✅ 对死区题提供非零梯度；对非死区题保持标准 RLVR 效果

---

## 2. 核心方法：RLSD（RL + Self-Distillation）

### 2.1 核心思路

训练数据为筛选出的 **死区题**（pass@1=0）。每步按 rollout 结果分流：

```
对死区题 x 采样 k 条轨迹后：

┌─ 组内有答对 ──→ GRPO：reward 差异产生 advantage，正常 RL 更新
│
└─ 组内全错 ───→ Self-Distillation：frozen reference 在 on-policy 轨迹上算 KL，提供非零梯度
```

随着训练推进，越来越多题从"全错"变为"有对有错"，自动从 SD 切换到 GRPO——无需手动调节。

### 2.2 Self-Distillation 分支详细流程

```
对组内全错的题目 x（ground truth 为 y*）：

Step 1 [Student Rollout]（已在 RLVR 采样时完成）
  π_S(·|x) 采样 k 条轨迹 ŷ_1...ŷ_k（均为错误答案）

Step 2 [Forward Pass — Teacher 拥有特权信息]
  对每条轨迹 ŷ_i：
  · Student context: (x, ŷ_{<n})          ← 仅有问题 + 已生成 token
  · Teacher context: (x, y*, ŷ_{<n})      ← 问题 + GT答案 + 已生成 token（特权信息）

  Teacher 拿到正确答案 y* 后，在每个 token 位置产生"知道答案时应该怎么续写"的分布
  Student 只看到问题，需要学习这种分布

Step 3 [Clipped Full-Distribution KL]
  对每个 token 位置 n：
    p_T(·) = p_ref(·|x, y*, ŷ_{<n})    ← no_grad，frozen θ_0
    p_S(·) = p_θ(·|x, ŷ_{<n})          ← 有梯度，当前模型
    D_clip(n) = Σ_{v∈V} min(p_T(v) · log(p_T(v)/p_S(v)), τ)
  对序列取均值得到该轨迹的 loss

Step 4 [梯度更新]
  梯度仅通过 π_S 传播；π_T 始终 frozen
```

**为什么 Teacher 需要特权信息**：Teacher（frozen ref）如果只看到和 student 相同的 context，它对死区题的分布同样差（因为它也答不对这些题）。给予 GT 答案后，teacher 在"知道答案"的条件下能产生更有信息量的 token 分布——即使它自身无法独立解出这道题。

### 2.3 Loss

每步 loss 取决于该题 rollout 的结果：

**情况 A：组内有对有错 → GRPO Loss**

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{\hat{y}\sim\pi_\theta} \left[ A(\hat{y}) \cdot \log\pi_\theta(\hat{y}|x) \right]$$

标准 GRPO advantage 计算（组内 reward 归一化）。

**情况 B：组内全错 → Self-Distillation Loss**

$$\mathcal{L}_{\text{SD}}(\theta) = \frac{1}{|\hat{y}|}\sum_{n=1}^{|\hat{y}|} D_{\text{clip}}^{\text{KL}}\Big(p_T^{(n)} \parallel p_S^{(n)}\Big)$$

$$D_{\text{clip}}^{\text{KL}}(p_T \parallel p_S) = \sum_{v\in\mathcal{V}} \min\left( p_T(v)\log\frac{p_T(v)}{p_S(v)},\ \tau \right)$$

- $p_T^{(n)} = p_{\theta_0}(\cdot|x, y^*, \hat{y}_{<n})$：frozen reference model 在 **特权 context**（含 GT）下的分布（`no_grad`）
- $p_S^{(n)} = p_\theta(\cdot|x, \hat{y}_{<n})$：当前 student 在 **无特权 context** 下的分布（有梯度）
- $\tau$：per-token KL clip 阈值（防止 style token 主导梯度）

### 2.4 为什么 Self-Distillation 在死区仍有效

GRPO 在死区上零梯度 = 完全放弃这些题。SD 提供了一个**非零但温和**的学习信号：

1. **防止策略坍缩**：RL 在其它题上的梯度可能让模型在死区题上 diversity 下降（重复同一种错误模式），SD 将分布拉回 reference 的多样性
2. **保持推理基础**：reference 的 token 分布隐含了通用推理偏好，SD 防止这些能力被 RL 侵蚀
3. **隐式课程**：随着训练推进，多样性恢复 → 某些死区题偶然产生正确答案 → 自动切换到 GRPO → 用 RL 信号进一步强化

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
| **GRPO（原始）**         | 标准 RLVR；在死区题上零梯度，验证问题存在             |
| **GRPO + KL penalty**| GRPO 加 reference KL 约束，但死区题梯度仍为零     |
| **Pure SD（OPSD）**    | 所有题都只做自蒸馏，不利用 reward signal           |
| **RLSD（本方法）**        | 有 signal → GRPO；无 signal → SD       |


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
| 去掉 KL clip（τ=∞）            | 验证 per-token clip 的必要性 |
| 用 gathered KL 替代 full-distribution KL | 验证 full-distribution 的贡献 |
| 不同 student rollout 数 k     | on-policy 数据量的影响      |
| 不同 kl_clip τ 值             | clip 阈值的敏感性           |
| OPSD 单独 vs OPSD + GRPO 联合 | 与 RL 信号的互补性           |


---

## 6. 实现细节

### 6.1 模型配置

```python
student_model = "Qwen/Qwen2.5-3B-Instruct"   # 训练中不断更新
teacher_model = "Qwen/Qwen2.5-3B-Instruct"   # frozen reference（初始权重，不更新）
# Teacher forward pass：no_grad，返回 logits 用于计算 full-distribution KL
# Student forward pass：有梯度，loss 通过 student logits 回传
```

### 6.2 训练超参数（初始建议）

```yaml
learning_rate: 1e-5
batch_size: 8                    # problems_per_step
student_rollout_per_problem: 4   # 每道题采 4 条 on-policy 轨迹
max_new_tokens: 8192             # student 生成上限
kl_clip: 10.0                   # per-token KL clip τ
training_steps: 500
eval_every: 10
```

### 6.3 关键工程细节

**Full-distribution clipped KL**：

- 对词表全分布计算 KL（而非只在 gathered token 上）
- per-token clip：`min(p_T(v) * log(p_T(v)/p_S(v)), τ)` 后对 vocab 求和
- style token（如 `\n`、`wait`）的 KL 值可能比数学 token 高 6-15 倍，clip 防止它们主导梯度

**Teacher = frozen reference model + 特权信息**：

- Teacher 使用与 Student 不同的 prompt：在问题之后注入 ground truth 答案作为特权信息
- Teacher prompt: `"Problem: {question}\n\nThe correct answer is: {GT}\n\nNow generate a solution:"`
- Student prompt: `"Problem: {question}"`
- 两者在相同的 response token 序列上做 forward，KL 在 response 位置逐 token 计算
- reference model 在整个训练过程中权重不更新

**动态课程**：

- 每 100 steps 重新评估哪些题已从 pass@1=0 "毕业"
- 毕业的题可切换为标准 GRPO（利用 pass@k>0 的 reward 信号继续强化）

---

## 7. 预期贡献与 Novelty

### 核心 Novelty

1. **统一框架**：首个将 RLVR 和 on-policy self-distillation 按 reward signal 自动分流的方法
   - 有 signal → RL；无 signal → SD；同一循环内无缝切换
2. **零额外依赖**：不需要外部 teacher、不需要人工 COT、不需要改题目格式
   - reference = 训练前初始 checkpoint，训练过程中 frozen
3. **解决零梯度**：在 GRPO 完全失效的场景下仍有非零学习信号
   - 且不干扰 GRPO 在有 signal 题上的正常效果

### 预期故事线

```
GRPO 在组内全错的题上产生零梯度，训练停滞
  ↓
观察：这些题占比可能 20-40%，严重拖累整体训练效率
  ↓
RLSD：对这些题改用 frozen reference 的 on-policy self-distillation
  ↓
训练后 coverage gain：原先 pass@1=0 的题有 X% 变为可解
  ↓
pass@k 曲线证明是真实推理能力提升（ceiling 提升），而非 reranking
  ↓
隐式课程：死区题逐渐产生 reward signal → 自动切换到 RLVR → 进一步强化
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