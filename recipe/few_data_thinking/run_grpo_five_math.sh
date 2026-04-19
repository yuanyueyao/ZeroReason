#!/usr/bin/env bash
# GRPO on explicitly repeated five-problem train set. Run from repo root:
#   bash recipe/few_data_thinking/run_grpo_five_math.sh
#
# Optional env:
#   NUM_CYCLES=200         — passes to build script (default 200 → 1000 rows)
#   REBUILD=1              — regenerate train.parquet before training
#   VAL_FILES=...          — validation parquet(s); default tries verl gsm8k test
#   TRAIN_NUM_EXAMINE=2    — print this many decoded train samples per reward batch (per data_source)
#   VAL_NUM_EXAMINE=1      — same for validation
#   USE_WANDB=1            — set 0 to disable Weights & Biases (console only)
#   WANDB_API_KEY          — or run: wandb login (metrics go to project trainer.project_name)
#   WANDB_ENTITY           — optional; team/username on wandb.ai
#   WANDB_MODE             — online | offline | disabled (default: online)
#   WB_PROJECT / WB_EXPERIMENT — override trainer.project_name / trainer.experiment_name for wandb

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# vLLM + FlashInfer JIT build links with -lcudart; conda/miniconda ld often lacks CUDA on LIBRARY_PATH.
# Match recipe/my_project/run.sh so ninja can find libcudart when compiling sampling ops.
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
TRAIN_PARQUET="${DATA_DIR}/train.parquet"
NUM_CYCLES="${NUM_CYCLES:-200}"

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -f "${TRAIN_PARQUET}" ]]; then
  python3 "${SCRIPT_DIR}/build_five_math_parquet.py" --out_dir "${DATA_DIR}" --num_cycles "${NUM_CYCLES}"
fi

# Default validation: full GSM8K test if present (user preprocesses elsewhere).
DEFAULT_VAL="/root/autodl-tmp/verl/data/gsm8k/test.parquet"
if [[ -n "${VAL_FILES:-}" ]]; then
  VAL_PATH="${VAL_FILES}"
elif [[ -f "${DEFAULT_VAL}" ]]; then
  VAL_PATH="${DEFAULT_VAL}"
else
  echo "WARN: no VAL_FILES set and ${DEFAULT_VAL} missing; using train parquet for val (smoke only)." >&2
  VAL_PATH="${TRAIN_PARQUET}"
fi

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"

# Decoded train samples per batch (per data_source); see NaiveRewardManager in verl.
TRAIN_NUM_EXAMINE="${TRAIN_NUM_EXAMINE:-2}"

export WANDB_MODE="${WANDB_MODE:-online}"

WB_PROJECT="${WB_PROJECT:-few_data_thinking}"
WB_EXPERIMENT="${WB_EXPERIMENT:-grpo_five_math_repeated}"

if [[ "${USE_WANDB:-1}" == "1" ]]; then
  TRAINER_LOGGER="[console,wandb]"
else
  TRAINER_LOGGER="[console]"
fi

# Hydra struct config: new keys under reward_model need a leading '+'.
python3 "${SCRIPT_DIR}/main_ppo.py" \
  algorithm.adv_estimator=grpo \
  +reward_model.train_num_examine="${TRAIN_NUM_EXAMINE}" \
  +reward_model.val_num_examine="${VAL_NUM_EXAMINE:-1}" \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PATH}" \
  data.shuffle=False \
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
  trainer.default_local_dir="${CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints/grpo_five_math_repeated}" \
  "$@"
