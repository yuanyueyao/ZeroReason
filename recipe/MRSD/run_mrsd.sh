#!/bin/bash
# MRSD 训练启动脚本（8×A800）
# 用法：bash recipe/MRSD/run_mrsd.sh [额外 hydra overrides]

set -euo pipefail

# ── 环境设置 ────────────────────────────────────────────────────────────────
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_COMPILE_DISABLE=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=WARN

# 项目根目录（脚本所在位置往上两级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# conda 环境
CONDA_ENV=verl

# ── 默认参数 ────────────────────────────────────────────────────────────────
MODEL_PATH=/data3/yyy/models/Qwen3-4B-Instruct-2507
DATA_DIR=/data3/yyy/verl/data/mrsd
CKPT_DIR=/data3/yyy/verl/checkpoints/mrsd

# 问题文件优先级：
#   1. type_b_problems.jsonl   （Context A/B 测试后精选的 Type-B 问题，最优）
#   2. dead_zone_problems.jsonl （pass@k 找到的全部死区，含 wrong_trajs，可直接使用）
#   3. pass_at_k_results.jsonl  （含全部题目，dataset 会自动过滤 is_dead_zone=True）
if [ -f "${DATA_DIR}/type_b_problems.jsonl" ]; then
    PROBLEMS_PATH="${DATA_DIR}/type_b_problems.jsonl"
    echo "[INFO] 使用 Context A/B 精选 Type-B 问题: ${PROBLEMS_PATH}"
elif [ -f "${DATA_DIR}/dead_zone_problems.jsonl" ]; then
    PROBLEMS_PATH="${DATA_DIR}/dead_zone_problems.jsonl"
    echo "[INFO] 使用 pass@k 死区问题（包含 wrong_trajs）: ${PROBLEMS_PATH}"
elif [ -f "${DATA_DIR}/pass_at_k_results.jsonl" ]; then
    PROBLEMS_PATH="${DATA_DIR}/pass_at_k_results.jsonl"
    echo "[WARNING] 回退到 pass_at_k_results.jsonl（将自动过滤 is_dead_zone=True）"
else
    echo "[ERROR] 找不到问题文件，请先运行 Phase 2 诊断实验"
    exit 1
fi

# ── 日志目录 ────────────────────────────────────────────────────────────────
LOG_DIR="${VERL_ROOT}/logs/mrsd"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "========================================================"
echo "MRSD 训练配置"
echo "  模型: ${MODEL_PATH}"
echo "  数据: ${PROBLEMS_PATH}"
echo "  检查点: ${CKPT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "========================================================"

# ── 启动训练 ────────────────────────────────────────────────────────────────
cd "${VERL_ROOT}"

conda run -n ${CONDA_ENV} --no-capture-output \
    python recipe/MRSD/main_mrsd.py \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        data.mrsd_problems_path="${PROBLEMS_PATH}" \
        data.train_files="${DATA_DIR}/train_level45.parquet" \
        data.val_files="${DATA_DIR}/test.parquet" \
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.project_name=mrsd \
        trainer.experiment_name="mrsd-qwen3-4b-${TIMESTAMP}" \
        trainer.total_training_steps=500 \
        trainer.save_freq=50 \
        mrsd.problems_per_step=8 \
        mrsd.student_rollout_per_problem=4 \
        mrsd.teacher_rollout_per_error=4 \
        mrsd.kl_clip=10.0 \
        "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "训练完成，日志已保存到 ${LOG_FILE}"
