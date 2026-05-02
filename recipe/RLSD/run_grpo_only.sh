#!/bin/bash
# GRPO-Only 对比实验（与 RLSD 做对照）
# 同数据集、同模型、同超参，唯一区别：跳过 SD 分支，死区题不产生梯度
# 用法：bash recipe/RLSD/run_grpo_only.sh [额外 hydra overrides]

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

# ── 默认参数（与 run_rlsd.sh 保持一致，便于对比）────────────────────────────
MODEL_PATH=/data3/yyy/models/Qwen2.5-3B-Instruct
DATA_DIR=/data3/yyy/verl/data/rlsd
CKPT_DIR=/data3/yyy/verl/checkpoints/grpo-only

PROBLEMS_PATH="/data3/yyy/verl/data/rlsd/pass_at_k_pass1_resp8192_20260501_095948_dead_zone.jsonl"

if [ ! -f "${PROBLEMS_PATH}" ]; then
    echo "[ERROR] 问题文件不存在: ${PROBLEMS_PATH}"
    exit 1
fi

# ── 日志目录 ────────────────────────────────────────────────────────────────
LOG_DIR="${VERL_ROOT}/logs/grpo-only"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "========================================================"
echo "GRPO-Only 对比训练配置"
echo "  模型:     ${MODEL_PATH}"
echo "  数据:     ${PROBLEMS_PATH}"
echo "  检查点:   ${CKPT_DIR}"
echo "  日志:     ${LOG_FILE}"
echo "  说明:     SD 分支已禁用，死区题跳过（grpo_only=true）"
echo "========================================================"

# ── 启动训练 ────────────────────────────────────────────────────────────────
cd "${VERL_ROOT}"

conda run -n ${CONDA_ENV} --no-capture-output \
    python recipe/RLSD/main_rlsd.py \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        data.mrsd_problems_path="${PROBLEMS_PATH}" \
        data.train_files="${DATA_DIR}/train_level45.parquet" \
        data.val_files="${DATA_DIR}/test.parquet" \
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.project_name=grpo-only \
        trainer.experiment_name="grpo-only-qwen25-3b-${TIMESTAMP}" \
        trainer.total_training_steps=500 \
        trainer.save_freq=50 \
        trainer.test_freq=10 \
        mrsd.problems_per_step=8 \
        mrsd.student_rollout_per_problem=8 \
        mrsd.grpo_only=true \
        "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "训练完成，日志已保存到 ${LOG_FILE}"
