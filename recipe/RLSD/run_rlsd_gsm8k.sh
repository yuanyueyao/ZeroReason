#!/bin/bash
# GSM8K + 完整 RLSD（SD + GRPO 混合；与 run_grpo_only.sh 同一套 actor 超参骨架）
# 用法：bash recipe/RLSD/run_rlsd_gsm8k.sh [额外 hydra overrides]
#
# - data：gsm8k train/test parquet；mrsd_problems_path=null → 从 train.parquet 建 MRSD 题池
# - mrsd.grpo_only=false → 死区走 SD，mixed 走 GRPO
# - actor：clip / use_kl_loss / kl_loss_coef / entropy_coeff 等与对照一致
# - 其余常用超参在下方 python … 参数里写明，可直接改数字

set -euo pipefail

# ── 环境设置 ────────────────────────────────────────────────────────────────
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_COMPILE_DISABLE=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=WARN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV=verl

# ── 默认参数 ────────────────────────────────────────────────────────────────
MODEL_PATH=/data3/yyy/models/Qwen2.5-3B-Instruct
DATA_DIR=/data3/yyy/verl/data/gsm8k
CKPT_DIR=/data3/yyy/verl/checkpoints/rlsd_gsm8k

# ── 日志目录 ────────────────────────────────────────────────────────────────
LOG_DIR="${VERL_ROOT}/logs/rlsd"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "========================================================"
echo "GSM8K 完整 RLSD（SD + GRPO）"
echo "  模型: ${MODEL_PATH}"
echo "  数据: ${DATA_DIR}"
echo "  检查点: ${CKPT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "========================================================"

# ── 启动训练 ────────────────────────────────────────────────────────────────
cd "${VERL_ROOT}"

conda run -n ${CONDA_ENV} --no-capture-output \
    python recipe/RLSD/main_rlsd.py \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.rollout.top_p=0.9 \
        data.train_files="${DATA_DIR}/train.parquet" \
        data.val_files="${DATA_DIR}/test.parquet" \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.project_name=rlsd \
        trainer.experiment_name="rlsd-gsm8k-qwen25-3b-${TIMESTAMP}" \
        trainer.total_training_steps=500 \
        trainer.save_freq=50 \
        trainer.test_freq=10 \
        trainer.resume_mode=auto \
        mrsd.student_rollout_per_problem=8 \
        mrsd.problems_per_step=8 \
        mrsd.grpo_only=false \
        "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "训练完成，日志已保存到 ${LOG_FILE}"
