# Copyright 2026 the verl recipe authors
"""Hydra entry: dual-pool math challenger + majority-vote rewards."""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from recipe.math_challenger_solver.math_trainer import MathChallengerTrainer
from recipe.math_challenger_solver.reward import MathMajorityRewardManager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer_A = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        tokenizer_B = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        from recipe.my_project.ray_trainer import Role

        role_worker_mapping = {
            Role.ActorRollout_A: ray.remote(actor_rollout_cls),
            Role.ActorRollout_B: ray.remote(actor_rollout_cls),
            Role.RefPolicy_A: ray.remote(actor_rollout_cls),
            Role.RefPolicy_B: ray.remote(actor_rollout_cls),
        }

        pool_id_A = "pool_A"
        pool_id_B = "pool_B"
        resource_pool_spec = {
            pool_id_A: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            pool_id_B: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout_A: pool_id_A,
            Role.ActorRollout_B: pool_id_B,
            Role.RefPolicy_A: pool_id_A,
            Role.RefPolicy_B: pool_id_B,
        }

        import recipe.math_challenger_solver.reward  # noqa: F401 — registers math_majority_reward

        reward_fn = MathMajorityRewardManager(
            tokenizer_A,
            config.reward_model.get("num_examine", 0),
            tokenizer_B=tokenizer_B,
            rollout_n=config.actor_rollout_ref.rollout.n,
            **config.reward_model.get("reward_kwargs", {}),
        )

        from verl.workers.reward_manager import NaiveRewardManager

        from recipe.my_project.val_b_compute_score import val_b_compute_score

        val_reward_fn = NaiveRewardManager(
            tokenizer=tokenizer_B,
            num_examine=2,
            compute_score=val_b_compute_score,
            reward_fn_key=config.data.reward_fn_key,
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer_A, None)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer_A, None)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = MathChallengerTrainer(
            config=config,
            tokenizer_A=tokenizer_A,
            tokenizer_B=tokenizer_B,
            processor=None,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit_competition()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )


def create_rl_sampler(data_config, dataset):
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    return SequentialSampler(data_source=dataset)


if __name__ == "__main__":
    main()
