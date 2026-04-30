#!/bin/bash
# ══════════════════════════════════════════════════════════
# MRSD §3 诊断实验 一键运行脚本
#
# 用法：
#   bash recipe/MRSD/diagnostic/run_diagnostic.sh
#
# 可选环境变量：
#   N_SAMPLES=64       每道题采样次数（pass@k 阶段）
#   N_PROBLEMS=50      Context A/B 测试题目数
#   N_CTX_SAMPLES=16   每种 context 采样次数
#   N_GPUS=8           使用 GPU 数量
#   DATA=...           输入数据 parquet 路径
#   MODEL=...          模型路径
#   OUT_DIR=...        输出目录
# ══════════════════════════════════════════════════════════
set -e

MODEL="${MODEL:-/data3/yyy/models/Qwen3-4B-Instruct-2507}"
DATA="${DATA:-/data3/yyy/verl/data/mrsd/train_level45.parquet}"
OUT_DIR="${OUT_DIR:-/data3/yyy/verl/data/mrsd}"
N_SAMPLES="${N_SAMPLES:-64}"
N_PROBLEMS="${N_PROBLEMS:-50}"
N_CTX_SAMPLES="${N_CTX_SAMPLES:-16}"
N_GPUS="${N_GPUS:-8}"

CONDA_ENV="verl"
CONDA_RUN="conda run -n $CONDA_ENV --no-capture-output"

echo "══════════════════════════════════════════════════════════"
echo " MRSD 诊断实验"
echo "══════════════════════════════════════════════════════════"
echo " 模型:        $MODEL"
echo " 数据:        $DATA"
echo " 输出目录:    $OUT_DIR"
echo " 采样次数:    $N_SAMPLES"
echo " GPU 数量:    $N_GPUS"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Step 1: pass@k 采样 ──
echo "[Step 1/3] pass@${N_SAMPLES} 采样..."
$CONDA_RUN python recipe/MRSD/diagnostic/run_pass_at_k.py \
    --data "$DATA" \
    --model "$MODEL" \
    --output "$OUT_DIR/pass_at_k_results.jsonl" \
    --n_samples "$N_SAMPLES" \
    --n_gpus "$N_GPUS" \
    --resume

echo ""
echo "[Step 1/3] 完成 ✓"
echo ""

# ── Step 2: Context A/B 测试 ──
echo "[Step 2/3] Context A/B 对比测试..."
$CONDA_RUN python recipe/MRSD/diagnostic/run_context_ab_test.py \
    --dead_zone "$OUT_DIR/dead_zone_problems.jsonl" \
    --model "$MODEL" \
    --output_dir "$OUT_DIR/diagnostic" \
    --n_problems "$N_PROBLEMS" \
    --n_samples_per_context "$N_CTX_SAMPLES" \
    --n_gpus "$N_GPUS"

echo ""
echo "[Step 2/3] 完成 ✓"
echo ""

# ── Step 3: 分类报告 ──
echo "[Step 3/3] 生成分类报告..."
$CONDA_RUN python recipe/MRSD/diagnostic/classify_problems.py \
    --pass_at_k "$OUT_DIR/pass_at_k_results.jsonl" \
    --context_ab "$OUT_DIR/diagnostic/context_ab_results.jsonl" \
    --output_dir "$OUT_DIR/diagnostic"

echo ""
echo "[Step 3/3] 完成 ✓"
echo ""
echo "══════════════════════════════════════════════════════════"
echo " 诊断实验结束，查看报告："
echo "  cat $OUT_DIR/diagnostic/diagnostic_report.txt"
echo "══════════════════════════════════════════════════════════"
