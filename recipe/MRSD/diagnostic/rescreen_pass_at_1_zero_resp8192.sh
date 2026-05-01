#!/usr/bin/env bash
# pass@1 + 筛 pass_at_1=0；单次生成上限 8192 tokens（与 data.max_response_length=8192 对齐）。
#
# 用法：
#   bash recipe/MRSD/diagnostic/rescreen_pass_at_1_zero_resp8192.sh
#
# 可选环境变量：
#   MODEL_PATH  DATA_PARQUET  OUT_DIR  N_GPUS  BATCH_SIZE  MAX_PROBLEMS
#   MAX_NEW_TOKENS（默认 8192）  MAX_MODEL_LEN（不设则由 Python 自动：≥ max_new_tokens+2048）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${VERL_ROOT}"

MODEL_PATH="${MODEL_PATH:-/data3/yyy/models/Qwen3-4B-Instruct-2507}"
DATA_PARQUET="${DATA_PARQUET:-${VERL_ROOT}/data/mrsd/train_level45.parquet}"
OUT_DIR="${OUT_DIR:-${VERL_ROOT}/data/mrsd}"
N_GPUS="${N_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PASS_JSONL="${OUT_DIR}/pass_at_k_pass1_resp8192_${TIMESTAMP}.jsonl"
ZERO_JSONL="${OUT_DIR}/pass_at_1_zero_resp8192_${TIMESTAMP}.jsonl"

TOKEN_ARGS=(--max_new_tokens "${MAX_NEW_TOKENS}")
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
  TOKEN_ARGS+=(--max_model_len "${MAX_MODEL_LEN}")
fi

EXTRA=()
if [[ -n "${MAX_PROBLEMS:-}" ]]; then
  EXTRA+=(--max_problems "${MAX_PROBLEMS}")
fi

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "pass@1 + pass_at_1=0（max_new_tokens=${MAX_NEW_TOKENS}）"
echo "  parquet: ${DATA_PARQUET}"
echo "  model:   ${MODEL_PATH}"
echo "  全量:    ${PASS_JSONL}"
echo "  =0 子集: ${ZERO_JSONL}"
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
  echo "  max_model_len: ${MAX_MODEL_LEN}（手动）"
else
  echo "  max_model_len: 自动 ≥ ${MAX_NEW_TOKENS}+2048"
fi
echo "=========================================="

conda run -n verl --no-capture-output python recipe/MRSD/diagnostic/run_pass_at_k.py \
  --data "${DATA_PARQUET}" \
  --model "${MODEL_PATH}" \
  --output "${PASS_JSONL}" \
  --n_samples 1 \
  --n_gpus "${N_GPUS}" \
  --batch_size "${BATCH_SIZE}" \
  "${TOKEN_ARGS[@]}" \
  "${EXTRA[@]}"

conda run -n verl --no-capture-output python recipe/MRSD/diagnostic/filter_pass_at_1_zero.py \
  --input "${PASS_JSONL}" \
  --output "${ZERO_JSONL}"

echo ""
echo "完成。"
echo "  全量 pass@1: ${PASS_JSONL}"
echo "  pass@1=0:    ${ZERO_JSONL}"
