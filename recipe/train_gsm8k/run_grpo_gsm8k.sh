#!/usr/bin/env bash
# Full GSM8K train split + GRPO（官方 RayPPO 链路，便于与 recipe/RLSD 的 GRPO 超参对齐）。
# 在仓库根目录执行：
#   bash recipe/train_gsm8k/run_grpo_gsm8k.sh
#
# 默认数据：若存在标准 GSM8K parquet（与 RLSD 相同数据源），则直接使用：
#   <VERL_ROOT>/data/gsm8k/train.parquet
#   <VERL_ROOT>/data/gsm8k/test.parquet
# （由 examples/data_preprocess/gsm8k.py 生成）。否则在本 recipe 下构建（prompt 为 #### 风格，与 RLSD 不同）。
#
# Optional env:
#   REBUILD=1              — 仅当使用 recipe/train_gsm8k/data 时，强制重建 parquet
#   TRAIN_FILES=... VAL_FILES=... — 覆盖数据路径
#   MODEL_PATH CHECKPOINT_DIR N_GPUS_PER_NODE CUDA_VISIBLE_DEVICES CONDA_ENV
#   TRAIN_BATCH_SIZE ROLLOUT_N PPO_MINI_BATCH_SIZE 等 — 按需覆盖
#   USE_WANDB=0 — 仅用 console

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

STANDARD_GSM8K_DIR="${VERL_ROOT}/data/gsm8k"
MODEL_PATH="${MODEL_PATH:-/data3/yyy/models/Qwen2.5-3B-Instruct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data3/yyy/verl/checkpoints/train_gsm8k_grpo}"

CONDA_ENV="${CONDA_ENV:-verl}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

if [[ -d /usr/local/cuda/lib64 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -e "${CONDA_PREFIX}/lib/libcudart.so" ]]; then
  export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

declare -a PY=(python3)
if command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python3 -c "import ray" >/dev/null 2>&1; then
  PY=(conda run -n "${CONDA_ENV}" --no-capture-output python3)
fi

DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "${DATA_DIR}"

if [[ -z "${TRAIN_FILES:-}" ]] && [[ -f "${STANDARD_GSM8K_DIR}/train.parquet" ]]; then
  DEFAULT_TRAIN="${STANDARD_GSM8K_DIR}/train.parquet"
else
  DEFAULT_TRAIN="${DATA_DIR}/train.parquet"
fi

TRAIN_PARQUET="${TRAIN_FILES:-${DEFAULT_TRAIN}}"

if [[ "${TRAIN_PARQUET}" == "${DATA_DIR}/train.parquet" ]] && { [[ "${REBUILD:-0}" == "1" ]] || [[ ! -f "${DATA_DIR}/train.parquet" ]]; }; then
  "${PY[@]}" "${SCRIPT_DIR}/build_gsm8k_parquet.py" --out_dir "${DATA_DIR}"
  TRAIN_PARQUET="${TRAIN_FILES:-${DATA_DIR}/train.parquet}"
fi

if [[ -n "${VAL_FILES:-}" ]]; then
  VAL_PATH="${VAL_FILES}"
elif [[ "${TRAIN_PARQUET}" == "${STANDARD_GSM8K_DIR}/train.parquet" ]] && [[ -f "${STANDARD_GSM8K_DIR}/test.parquet" ]]; then
  VAL_PATH="${STANDARD_GSM8K_DIR}/test.parquet"
elif [[ -f "${DATA_DIR}/test.parquet" ]]; then
  VAL_PATH="${DATA_DIR}/test.parquet"
else
  echo "WARN: 无可用验证 parquet，val 退回 train（仅冒烟）." >&2
  VAL_PATH="${TRAIN_PARQUET}"
fi

echo "TRAIN_PARQUET=${TRAIN_PARQUET}"
echo "VAL_PATH=${VAL_PATH}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"

TRAIN_NUM_EXAMINE="${TRAIN_NUM_EXAMINE:-2}"

export WANDB_MODE="${WANDB_MODE:-online}"

WB_PROJECT="${WB_PROJECT:-rlsd}"
WB_EXPERIMENT="${WB_EXPERIMENT:-grpo_gsm8k_full_train}"

if [[ "${USE_WANDB:-1}" == "1" ]]; then
  TRAINER_LOGGER="[console,wandb]"
else
  TRAINER_LOGGER="[console]"
fi

"${PY[@]}" "${SCRIPT_DIR}/main_ppo.py" \
  algorithm.adv_estimator=grpo \
  +reward_model.train_num_examine="${TRAIN_NUM_EXAMINE}" \
  +reward_model.val_num_examine="${VAL_NUM_EXAMINE:-1}" \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PATH}" \
  data.shuffle=True \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-8}" \
  data.max_prompt_length=1024 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
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
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTIL:-0.6}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH_PER_GPU:-2}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name="${WB_PROJECT}" \
  trainer.experiment_name="${WB_EXPERIMENT}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq="${TEST_FREQ:-20}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-10}" \
  trainer.val_before_train=True \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  "$@"
