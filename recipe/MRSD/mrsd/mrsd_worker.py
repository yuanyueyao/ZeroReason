"""
RLSD 自定义 Worker。

继承 verl ActorRolloutRefWorker：
  1. init_model: 将 Actor 替换为 RLSDPPOActor
  2. update_actor: 对 SD 分支，先用 ref_policy 计算 ref_logits，再调 actor.update_policy

约束：仅修改 recipe/MRSD/ 目录。
"""

from __future__ import annotations

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class MRSDActorRolloutWorker(ActorRolloutRefWorker):
    """
    RLSD Actor + Rollout + Ref Worker。

    区别于基类：
      - init_model: 替换 Actor 为 RLSDPPOActor
      - update_actor: SD 分支在本地计算 ref_logits（避免跨 worker 传输 V 维 logits）
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()

        if not self._is_actor:
            return

        import sys
        from pathlib import Path
        _recipe_root = Path(__file__).parent.parent.parent.parent
        if str(_recipe_root) not in sys.path:
            sys.path.insert(0, str(_recipe_root))

        from recipe.MRSD.mrsd.mrsd_actor import RLSDPPOActor

        self.actor = RLSDPPOActor(
            config=self.config.actor,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        """
        覆写 update_actor：
          - 对 SD 分支，将 ref_policy 传给 actor 供其逐 micro-batch 计算 ref_logits
          - 对 GRPO 分支，直接走标准 update
        """
        import psutil
        from codetiming import Timer
        from verl.utils.device import get_torch_device

        data = data.to("cpu")
        assert self._is_actor

        mode = data.meta_info.get("rlsd_mode", "sd")

        # 将 ref_policy 引用传入 actor（SD 分支需要）
        if mode == "sd" and self._is_ref:
            self.actor._ref_module = self.ref_policy.actor_module
        else:
            self.actor._ref_module = None

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        # 清理引用
        self.actor._ref_module = None
        return output
