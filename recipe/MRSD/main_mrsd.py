"""
MRSD 训练主入口。完全对齐 verl 官方 recipe 模式（参考 recipe/entropy/main_entropy.py）。

用法：
    bash recipe/MRSD/run_mrsd.sh
    python recipe/MRSD/main_mrsd.py [hydra overrides]
"""

import os
import socket

# 必须在 import torch/verl 之前设置
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
if "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="mrsd_trainer", version_base=None)
def main(config):
    run_mrsd(config)


def run_mrsd(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARNING",
                "TORCH_COMPILE_DISABLE": "1",
                "CUDA_HOME": "/usr/local/cuda",
            }},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils.dataset.rl_dataset import collate_fn

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # ── 模型路径 & Tokenizer ──────────────────────────────────────
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))

        # ── Worker 类（MRSD 自定义 worker，替换内部 actor 的 loss）────
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from recipe.MRSD.mrsd.mrsd_worker import MRSDActorRolloutWorker
        from recipe.MRSD.mrsd.mrsd_trainer import MRSDTrainer
        from recipe.MRSD.mrsd.dataset import MRSDDataset

        actor_rollout_cls = MRSDActorRolloutWorker
        ray_worker_group_cls = RayWorkerGroup

        # ── Role → Worker 映射（GRPO 模式：无 critic，无 RM，无 ref）──
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {Role.ActorRollout: global_pool_id}

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        # ── 标准 verl 数据集（只用于满足 dataloader 初始化，不参与 MRSD 训练）
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, None)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, None)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # ── MRSD 死区问题数据集 ───────────────────────────────────────
        mrsd_cfg = config.mrsd
        mrsd_problems_path = config.data.mrsd_problems_path
        print(f"[main] 加载死区问题: {mrsd_problems_path}")
        mrsd_dataset = MRSDDataset.from_pass_at_k_results(
            pass_at_k_jsonl=mrsd_problems_path,
            type_b_only=True,
            graduation_interval=int(mrsd_cfg.graduation_interval),
            graduation_pass_at_k=int(mrsd_cfg.graduation_pass_at_k),
        )
        print(f"[main] 共 {len(mrsd_dataset)} 道死区题目")

        # ── 训练器 ────────────────────────────────────────────────────
        trainer = MRSDTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            mrsd_dataset=mrsd_dataset,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
