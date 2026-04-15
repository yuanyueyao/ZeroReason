# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
export CUDA_VISIBLE_DEVICES=0,1
# vLLM/FlashInfer JIT links with -lcudart; conda nvcc may not place libcudart on the link path.
if [ -d /usr/local/cuda/lib64 ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
python3 -m recipe.my_project.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.diversity_penalty_coeff=0.1 \
    algorithm.diversity_penalty_method=jaccard \
    algorithm.diversity_penalty_kwargs={} \
    data.train_files=/root/autodl-tmp/verl/data/gsm8k/test.parquet \
    data.val_files="[/root/autodl-tmp/verl/data/gsm8k/test.parquet,/root/autodl-tmp/verl/data/mbpp/test.parquet]" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_fit_competition' \
    trainer.experiment_name='test_Qwen2.5-1.5B-Instruct' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1200 \
    trainer.val_before_train=True \
    trainer.default_local_dir=/root/autodl-tmp/verl/checkpoints/Qwen2.5-1.5B-Instruct $@
    # trainer.resume_mode=auto \
    # trainer.wandb_run_id=23coh2so \
    # trainer.wandb_resume=allow $@
