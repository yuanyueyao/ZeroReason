#!/usr/bin/env bash
# Example: math challenger (A: ```problem``` only) + B majority-vote pseudo labels.
# Requires 2× trainer.n_gpus_per_node × nnodes GPUs (pool_A + pool_B).
# vLLM chunked prefill: rollout.max_num_batched_tokens >= max_prompt_length + max_response_length.
set -eux
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

# Seed A's history from GSM8K train parquet by default. Disable: GSM8K_INITIAL_HISTORY=none
IH_PARQUET="${GSM8K_INITIAL_HISTORY:-/data0/yyy/verl/data/gsm8k/train.parquet}"

HIST_SEED_ARGS=()
if [[ -n "${GSM8K_INITIAL_HISTORY_SEED:-}" ]]; then
  HIST_SEED_ARGS+=(+math_challenger.initial_history_seed="${GSM8K_INITIAL_HISTORY_SEED}")
fi

python3 -m recipe.math_challenger_solver.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_TRAIN:-/data0/yyy/verl/data/gsm8k/test.parquet}" \
  data.val_files="${DATA_VAL:-/data0/yyy/verl/data/gsm8k/test.parquet}" \
  data.train_batch_size=1024 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH:-/data0/yyy/models/Qwen3-4B-Instruct-2507}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.n=8 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_training_steps=600 \
  trainer.logger=['console','wandb'] \
  trainer.test_freq=20 \
  trainer.project_name=math_challenger_solver \
  trainer.experiment_name=qwen3-4b-instruct-2507 \
  reward_model.num_examine=2 \
  math_challenger.history_window_size=10 \
  math_challenger.log_groups_per_step=1 \
  math_challenger.log_A_samples_per_step=8 \
  math_challenger.use_problem_history=False \
  math_challenger.initial_history_gsm8k_parquet="${IH_PARQUET}" \
  math_challenger.initial_history_num_problems="${GSM8K_INITIAL_HISTORY_N:-10}" \
  "${HIST_SEED_ARGS[@]}" \
  "$@"
