#!/usr/bin/env bash
# GSM8K 评测：仅跑验证，将全部题目的模型输出与规则分数写入
#   ${OUT_DIR}/0.jsonl
# 需已准备 openai/gsm8k 风格的 test.parquet（见 examples/data_preprocess/gsm8k.py）。
# 在仓库根目录执行：
#   bash recipe/find_wrong_qa/eval_gsm8k.sh
#
# 环境变量（可选）：
#   CUDA_VISIBLE_DEVICES  默认 0
#   MODEL_PATH            默认 /data3/yyy/models/Qwen3-4B-Base
#   DATA_VAL              默认 /data3/yyy/verl/data/gsm8k/test.parquet
#   OUT_DIR               默认 recipe/find_wrong_qa/outputs/gsm8k_eval
#   N_GPUS_PER_NODE       默认 1
# 其余可追加 hydra 覆盖：  bash recipe/find_wrong_qa/eval_gsm8k.sh data.max_response_length=512

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

# 直接执行 recipe/.../main_ppo.py 时 sys.path 不含仓库根，未 pip install -e 时会报 No module named 'verl'
export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# vLLM + FlashInfer 编译链：与 math_challenger 脚本一致
if [[ -d /usr/local/cuda/lib64 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -e "${CONDA_PREFIX}/lib/libcudart.so" ]]; then
  export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

DATA_VAL="${DATA_VAL:-/data3/yyy/verl/data/gsm8k/test.parquet}"
MODEL_PATH="${MODEL_PATH:-/data3/yyy/models/Qwen3-4B-Base}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/gsm8k_eval}"
mkdir -p "${OUT_DIR}"
# Hydra 落盘用绝对路径，避免工作目录变化导致找不到目录
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"

VAL_NUM_EXAMINE="${VAL_NUM_EXAMINE:-0}"

python3 "${VERL_ROOT}/recipe/train_gsm8k/main_ppo.py" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  +reward_model.train_num_examine=0 \
  +reward_model.val_num_examine="${VAL_NUM_EXAMINE}" \
  data.train_files="${DATA_VAL}" \
  data.val_files="${DATA_VAL}" \
  data.shuffle=False \
  data.train_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger="[console]" \
  trainer.project_name=find_wrong_qa \
  trainer.experiment_name=gsm8k_eval \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  "+trainer.validation_data_dir=${OUT_DIR}" \
  trainer.default_local_dir="${OUT_DIR}/checkpoints" \
  "$@"

echo "GSM8K eval dump: ${OUT_DIR}/0.jsonl (JSONL, one line per val sample; score 1=correct, 0=wrong)"
