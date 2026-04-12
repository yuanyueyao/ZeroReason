# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from recipe.my_project.ray_trainer import MyTrainer
from recipe.my_project.reward import CompetitionRewardManager
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer_A = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        tokenizer_B = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)


        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")


        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        actor_rollout_cls =  ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup



        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        from recipe.my_project.ray_trainer import Role
        print("success")

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout_A: ray.remote(actor_rollout_cls),
            Role.ActorRollout_B: ray.remote(actor_rollout_cls),
            Role.RefPolicy_A: ray.remote(actor_rollout_cls),
            Role.RefPolicy_B: ray.remote(actor_rollout_cls),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
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


        # Register custom RewardManagers (e.g. @register("word_count") in recipe/my_project/reward.py).
        import recipe.my_project.reward  # noqa: F401

        # Load the reward manager for training and validation.
        reward_fn = CompetitionRewardManager(
            tokenizer_A,
            5,
            tokenizer_B=tokenizer_B,
            rollout_n=config.actor_rollout_ref.rollout.n,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = CompetitionRewardManager(
            tokenizer_A,
            1,
            tokenizer_B=tokenizer_B,
            rollout_n=config.actor_rollout_ref.rollout.n,
            **config.reward_model.get("reward_kwargs", {}),
        )
        # Naive + default_compute_score for periodic GSM8K eval of model B (see MyTrainer._my_validate).
        gsm8k_b_cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        with open_dict(gsm8k_b_cfg):
            gsm8k_b_cfg.reward_model.reward_manager = "naive"
        val_reward_fn_gsm8k_b = load_reward_manager(gsm8k_b_cfg, tokenizer_B, num_examine=2)
        # reward_fn = load_reward_manager(config, tokenizer_A, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        # val_reward_fn = load_reward_manager(config, tokenizer_A, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        print("reward_fn", reward_fn)
        print("val_reward_fn", val_reward_fn)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer_A, None)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer_A, None)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = MyTrainer(
            config=config,
            tokenizer_A=tokenizer_A,
            tokenizer_B=tokenizer_B,
            processor=None,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            val_reward_fn_gsm8k_b=val_reward_fn_gsm8k_b,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        print("init success")
        # out = trainer.infer_test("Write a program to print the sum of two numbers")
        # print(out)
        out = trainer.fit_competition()
        print(out)
        print("fit_competition success")



def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
