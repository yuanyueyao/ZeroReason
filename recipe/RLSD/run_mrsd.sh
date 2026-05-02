#!/bin/bash
# RLSD 训练启动脚本（8×A800）
# 用法：bash recipe/RLSD/run_mrsd.sh [额外 hydra overrides]

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
DATA_DIR=/data3/yyy/verl/data/mrsd
CKPT_DIR=/data3/yyy/verl/checkpoints/rlsd

MRSD_PROBLEMS_PATH="/data3/yyy/verl/data/mrsd/pass_at_k_pass1_resp8192_20260501_095948_dead_zone.jsonl"

# 问题文件选择逻辑
if [ -n "${MRSD_PROBLEMS_PATH:-}" ]; then
    if [ ! -f "${MRSD_PROBLEMS_PATH}" ]; then
        echo "[ERROR] MRSD_PROBLEMS_PATH 文件不存在: ${MRSD_PROBLEMS_PATH}"
        exit 1
    fi
    PROBLEMS_PATH="${MRSD_PROBLEMS_PATH}"
    echo "[INFO] 使用 MRSD_PROBLEMS_PATH: ${PROBLEMS_PATH}"
elif [ -f "${DATA_DIR}/dead_zone_verified.jsonl" ]; then
    PROBLEMS_PATH="${DATA_DIR}/dead_zone_verified.jsonl"
    echo "[INFO] 使用 math_verify 验证后的真死区: ${PROBLEMS_PATH}"
elif [ -f "${DATA_DIR}/dead_zone_phase_a.jsonl" ]; then
    PROBLEMS_PATH="${DATA_DIR}/dead_zone_phase_a.jsonl"
elif [ -f "${DATA_DIR}/dead_zone_problems.jsonl" ]; then
    PROBLEMS_PATH="${DATA_DIR}/dead_zone_problems.jsonl"
elif [ -f "${DATA_DIR}/pass_at_k_results.jsonl" ]; then
    PROBLEMS_PATH="${DATA_DIR}/pass_at_k_results.jsonl"
    echo "[WARNING] 回退到 pass_at_k_results.jsonl"
else
    echo "[ERROR] 找不到问题文件，请先运行诊断实验"
    exit 1
fi

# ── 日志目录 ────────────────────────────────────────────────────────────────
LOG_DIR="${VERL_ROOT}/logs/rlsd"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "========================================================"
echo "RLSD 训练配置"
echo "  模型: ${MODEL_PATH}"
echo "  数据: ${PROBLEMS_PATH}"
echo "  检查点: ${CKPT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "========================================================"

# ── 启动训练 ────────────────────────────────────────────────────────────────
cd "${VERL_ROOT}"

conda run -n ${CONDA_ENV} --no-capture-output \
    python recipe/RLSD/main_mrsd.py \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        data.mrsd_problems_path="${PROBLEMS_PATH}" \
        data.train_files="${DATA_DIR}/train_level45.parquet" \
        data.val_files="${DATA_DIR}/test.parquet" \
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.project_name=rlsd \
        trainer.experiment_name="rlsd-qwen25-3b-${TIMESTAMP}" \
        trainer.total_training_steps=500 \
        trainer.save_freq=50 \
        trainer.test_freq=10 \
        mrsd.problems_per_step=8 \
        mrsd.student_rollout_per_problem=8 \
        mrsd.kl_clip=1.0 \
        "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "训练完成，日志已保存到 ${LOG_FILE}"
