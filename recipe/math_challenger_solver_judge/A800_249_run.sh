#!/usr/bin/env bash
# math_challenger_solver_judge: challenger (A) + solver (B) + trainable judge (J), <score>...</score>.
# Requires 3× trainer.n_gpus_per_node (pools A / B / J) × nnodes GPUs when all three use the same per-node count.
# Layout mirrors recipe/math_challenger_solver/scripts/A800-250-run.sh
set -eux
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

# vLLM v1 + FlashInfer JIT compiles sampling kernels with nvcc then links with -lcudart.
# /usr/bin/nvcc is CUDA 11.5 (system); the real CUDA toolkit lives under /usr/local/cuda.
# Without CUDA_HOME + PATH pointing to the correct nvcc, ninja picks up the wrong compiler
# and the include chain breaks with "math.h: No such file or directory".
if [[ -d /usr/local/cuda/lib64 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Seed A's history from GSM8K train parquet (optional). Disable: GSM8K_INITIAL_HISTORY=none
IH_PARQUET="${GSM8K_INITIAL_HISTORY:-/data3/yyy/verl/data/gsm8k/train.parquet}"

HIST_SEED_ARGS=()
if [[ -n "${GSM8K_INITIAL_HISTORY_SEED:-}" ]]; then
  HIST_SEED_ARGS+=(+math_challenger.initial_history_seed="${GSM8K_INITIAL_HISTORY_SEED}")
fi

python3 -m recipe.math_challenger_solver_judge.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_TRAIN:-/data3/yyy/verl/data/gsm8k/test.parquet}" \
  data.val_files="${DATA_VAL:-/data3/yyy/verl/data/gsm8k/test.parquet}" \
  data.train_batch_size=1024 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  actor_rollout_ref_b.actor.optim.lr=3e-6 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH:-/data3/yyy/models/Qwen3-4B-Base}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.n=8 \
  trainer.n_gpus_per_node=2 \
  trainer.n_gpus_per_node_b=2 \
  trainer.n_gpus_per_node_j=2 \
  trainer.nnodes=1 \
  trainer.total_training_steps=500 \
  trainer.logger=['console','wandb'] \
  trainer.test_freq=20 \
  trainer.project_name=math_challenger_solver_judge \
  trainer.experiment_name="${EXPERIMENT_NAME:-Qwen3-4B-Base}" \
  reward_model.num_examine=2 \
  math_challenger.history_window_size=10 \
  math_challenger.log_groups_per_step=1 \
  math_challenger.log_A_samples_per_step=8 \
  math_challenger.use_problem_history=False \
  math_challenger.initial_history_gsm8k_parquet="${IH_PARQUET}" \
  math_challenger.initial_history_num_problems="${GSM8K_INITIAL_HISTORY_N:-10}" \
  judge.entropy_zero=false \
  judge.no_train=true \
  judge.judge_rollout_n=1 \
  judge.actor_rollout_ref.model.path="${JUDGE_MODEL_PATH:-/data3/yyy/models/Qwen3-4B-Instruct-2507}" \
  math_challenger_judge.alpha=0.1 \
  math_challenger_judge.score_min=1.0 \
  math_challenger_judge.score_max=10.0 \
  math_challenger_judge.judge_format_weight=0.5 \
  math_challenger_judge.judge_range_weight=0.5 \
  math_challenger_judge.judge_dominant=true \
  "${HIST_SEED_ARGS[@]}" \
  "$@"
