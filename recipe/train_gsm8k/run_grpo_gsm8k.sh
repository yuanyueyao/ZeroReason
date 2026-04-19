#!/usr/bin/env bash
# Full GSM8K train split + GRPO (same hyperparameters / model defaults as few_data_thinking).
# Run from repo root:
#   bash recipe/train_gsm8k/run_grpo_gsm8k.sh
#
# First run downloads openai/gsm8k via HuggingFace datasets and writes:
#   recipe/train_gsm8k/data/train.parquet  (~7.5k rows)
#   recipe/train_gsm8k/data/test.parquet
#
# Optional env:
#   REBUILD=1              — regenerate parquets from HuggingFace
#   TRAIN_FILES=...        — override training parquet path
#   VAL_FILES=...          — override validation parquet (default: built test.parquet)
#   TRAIN_NUM_EXAMINE=2    — decoded train samples per reward batch (per data_source)
#   VAL_NUM_EXAMINE=1      — same for validation
#   USE_WANDB=1            — set 0 for console-only logging
#   WANDB_API_KEY / wandb login
#   WANDB_ENTITY, WANDB_MODE, WB_PROJECT, WB_EXPERIMENT

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -d /usr/local/cuda/lib64 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -e "${CONDA_PREFIX}/lib/libcudart.so" ]]; then
  export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "${DATA_DIR}"

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
  python3 "${SCRIPT_DIR}/build_gsm8k_parquet.py" --out_dir "${DATA_DIR}"
fi

TRAIN_PARQUET="${TRAIN_FILES:-${DATA_DIR}/train.parquet}"

if [[ -n "${VAL_FILES:-}" ]]; then
  VAL_PATH="${VAL_FILES}"
elif [[ -f "${DATA_DIR}/test.parquet" ]]; then
  VAL_PATH="${DATA_DIR}/test.parquet"
elif [[ -f "/root/autodl-tmp/verl/data/gsm8k/test.parquet" ]]; then
  echo "WARN: ${DATA_DIR}/test.parquet missing; using verl data/gsm8k/test.parquet (prompt style may differ from this recipe's train)." >&2
  VAL_PATH="/root/autodl-tmp/verl/data/gsm8k/test.parquet"
else
  echo "WARN: no validation parquet; using train parquet for val (smoke only)." >&2
  VAL_PATH="${TRAIN_PARQUET}"
fi

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"

TRAIN_NUM_EXAMINE="${TRAIN_NUM_EXAMINE:-2}"

export WANDB_MODE="${WANDB_MODE:-online}"

WB_PROJECT="${WB_PROJECT:-train_gsm8k}"
WB_EXPERIMENT="${WB_EXPERIMENT:-grpo_gsm8k_full_train}"

if [[ "${USE_WANDB:-1}" == "1" ]]; then
  TRAINER_LOGGER="[console,wandb]"
else
  TRAINER_LOGGER="[console]"
fi

python3 "${SCRIPT_DIR}/main_ppo.py" \
  algorithm.adv_estimator=grpo \
  +reward_model.train_num_examine="${TRAIN_NUM_EXAMINE}" \
  +reward_model.val_num_examine="${VAL_NUM_EXAMINE:-1}" \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PATH}" \
  data.shuffle=True \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-8}" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_PER_GPU:-2}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_PER_GPU:-2}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTIL:-0.5}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH_PER_GPU:-2}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name="${WB_PROJECT}" \
  trainer.experiment_name="${WB_EXPERIMENT}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-1}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq="${TEST_FREQ:-20}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-10}" \
  trainer.val_before_train=True \
  trainer.default_local_dir="${CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints/grpo_gsm8k_full}" \
  "$@"
