# Copyright 2026 the verl recipe authors
"""Hydra entry: three-pool math challenger + trainable judge."""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from recipe.math_challenger_solver_judge.judge_trainer import MathChallengerJudgeTrainer
from recipe.math_challenger_solver_judge.reward import MathMajorityJudgeRewardManager
from recipe.my_project.ray_trainer import ResourcePoolManager, Role


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

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        with open_dict(config):
            if OmegaConf.select(config, "judge") is None:
                config.judge = OmegaConf.create({})
            # 始终以 actor_rollout_ref 为 base，再将 judge.actor_rollout_ref（可能只含 model.path 等
            # 少量覆盖项）叠加其上，从而支持 CLI 通过 judge.actor_rollout_ref.model.path= 指定不同模型。
            _judge_override = OmegaConf.select(config, "judge.actor_rollout_ref", default=None)
            _full_j = OmegaConf.merge(OmegaConf.create({}), config.actor_rollout_ref)
            if _judge_override is not None:
                _full_j = OmegaConf.merge(_full_j, _judge_override)
            with open_dict(config.judge):
                config.judge.actor_rollout_ref = _full_j
            with open_dict(config.judge.actor_rollout_ref.rollout):
                # 与 A/B 的 rollout.n 解耦；GRPO 需 judge 每题多条样本（默认 4，见 judge.judge_rollout_n）
                jn = int(OmegaConf.select(config, "judge.judge_rollout_n", default=4) or 4)
                config.judge.actor_rollout_ref.rollout.n = max(1, jn)

        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        judge_cfg_path = OmegaConf.select(config, "judge.actor_rollout_ref.model.path", default=None)
        if judge_cfg_path is not None and str(judge_cfg_path) != str(config.actor_rollout_ref.model.path):
            j_shm = OmegaConf.select(config, "judge.actor_rollout_ref.model.use_shm", default=None)
            use_shm_j = j_shm if j_shm is not None else config.actor_rollout_ref.model.get("use_shm", False)
            judge_path = copy_to_local(str(judge_cfg_path), use_shm=use_shm_j)
        else:
            judge_path = local_path

        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer_A = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        tokenizer_B = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        tokenizer_J = hf_tokenizer(judge_path, trust_remote_code=trust_remote_code)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0 and not is_version_ge(pkg="vllm", minver="0.7.3"):
                raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        no_train_j = bool(OmegaConf.select(config, "judge.no_train", default=False))

        role_worker_mapping = {
            Role.ActorRollout_A: ray.remote(ActorRolloutRefWorker),
            Role.ActorRollout_B: ray.remote(ActorRolloutRefWorker),
            Role.ActorRollout_J: ray.remote(ActorRolloutRefWorker),
            Role.RefPolicy_A: ray.remote(ActorRolloutRefWorker),
            Role.RefPolicy_B: ray.remote(ActorRolloutRefWorker),
        }
        if not no_train_j:
            role_worker_mapping[Role.RefPolicy_J] = ray.remote(ActorRolloutRefWorker)

        nn = int(config.trainer.nnodes)
        ga = int(config.trainer.n_gpus_per_node)
        raw_b = config.trainer.get("n_gpus_per_node_b", None)
        raw_j = config.trainer.get("n_gpus_per_node_j", None)
        gb = int(raw_b) if raw_b is not None else ga
        gj = int(raw_j) if raw_j is not None else ga
        resource_pool_spec = {
            "pool_A": [ga] * nn,
            "pool_B": [gb] * nn,
            "pool_J": [gj] * nn,
        }
        mapping = {
            Role.ActorRollout_A: "pool_A",
            Role.ActorRollout_B: "pool_B",
            Role.ActorRollout_J: "pool_J",
            Role.RefPolicy_A: "pool_A",
            Role.RefPolicy_B: "pool_B",
        }
        if not no_train_j:
            mapping[Role.RefPolicy_J] = "pool_J"

        import recipe.math_challenger_solver_judge.reward  # noqa: F401

        mcj = OmegaConf.select(config, "math_challenger_judge", default={}) or {}
        reward_fn = MathMajorityJudgeRewardManager(
            tokenizer_A,
            int(config.reward_model.get("num_examine", 0)),
            tokenizer_B=tokenizer_B,
            rollout_n=int(config.actor_rollout_ref.rollout.n),
            alpha=float(mcj.get("alpha", 0.1)),
            beta=float(mcj.get("beta", 0.1)),
            beta_penalty=float(mcj.get("beta_penalty", 0.0)),
            judge_dominant=bool(mcj.get("judge_dominant", False)),
        )

        from verl.workers.reward_manager import NaiveRewardManager
        from verl.utils.dataset.rl_dataset import collate_fn
        from recipe.my_project.val_b_compute_score import val_b_compute_score

        val_reward_fn = NaiveRewardManager(
            tokenizer=tokenizer_B,
            num_examine=2,
            compute_score=val_b_compute_score,
            reward_fn_key=config.data.reward_fn_key,
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer_A, None)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer_A, None)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = MathChallengerJudgeTrainer(
            config=config,
            tokenizer_A=tokenizer_A,
            tokenizer_B=tokenizer_B,
            tokenizer_J=tokenizer_J,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
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
    from verl.utils.import_utils import load_extern_type

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError("custom dataset must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")
    return dataset_cls(data_files=data_paths, tokenizer=tokenizer, processor=processor, config=data_config)


def create_rl_sampler(data_config, dataset):
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        g = torch.Generator()
        g.manual_seed(int(data_config.get("seed", 1)))
        return RandomSampler(data_source=dataset, generator=g)
    return SequentialSampler(data_source=dataset)


if __name__ == "__main__":
    main()
