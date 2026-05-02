#!/usr/bin/env bash
# 从零重新跑 pass@1（不读旧 jsonl），再筛出 pass_at_1=0。
# 用法：
#   bash recipe/RLSD/diagnostic/rescreen_pass_at_1_zero.sh
# 可选环境变量：
#   MODEL_PATH  DATA_PARQUET  N_GPUS  BATCH_SIZE  MAX_PROBLEMS  OUT_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${VERL_ROOT}"

MODEL_PATH="${MODEL_PATH:-/data3/yyy/models/Qwen3-4B-Instruct-2507}"
DATA_PARQUET="${DATA_PARQUET:-${VERL_ROOT}/data/mrsd/train_level45.parquet}"
OUT_DIR="${OUT_DIR:-${VERL_ROOT}/data/mrsd}"
N_GPUS="${N_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PASS_JSONL="${OUT_DIR}/pass_at_k_pass1_${TIMESTAMP}.jsonl"
ZERO_JSONL="${OUT_DIR}/pass_at_1_zero_${TIMESTAMP}.jsonl"

EXTRA=()
if [[ -n "${MAX_PROBLEMS:-}" ]]; then
  EXTRA+=(--max_problems "${MAX_PROBLEMS}")
fi

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "重新筛选 pass@1=0（全新推理 + 全新输出）"
echo "  parquet: ${DATA_PARQUET}"
echo "  model:   ${MODEL_PATH}"
echo "  pass@1 结果: ${PASS_JSONL}"
echo "  pass@1=0 子集: ${ZERO_JSONL}"
echo "=========================================="

conda run -n verl --no-capture-output python recipe/RLSD/diagnostic/run_pass_at_k.py \
  --data "${DATA_PARQUET}" \
  --model "${MODEL_PATH}" \
  --output "${PASS_JSONL}" \
  --n_samples 1 \
  --n_gpus "${N_GPUS}" \
  --batch_size "${BATCH_SIZE}" \
  "${EXTRA[@]}"

conda run -n verl --no-capture-output python recipe/RLSD/diagnostic/filter_pass_at_1_zero.py \
  --input "${PASS_JSONL}" \
  --output "${ZERO_JSONL}"

echo ""
echo "完成。"
echo "  全量 pass@1: ${PASS_JSONL}"
echo "  pass@1=0:    ${ZERO_JSONL}"
