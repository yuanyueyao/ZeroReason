"""
MRSD 自定义 Worker。

继承 verl ActorRolloutRefWorker，覆写 init_model 中 Actor 的实例化，
将 DataParallelPPOActor 替换为 MRSDPPOActor（使用 KL loss）。

其余方法（generate_sequences、compute_log_prob、save_checkpoint、load_checkpoint）
全部复用父类，保持与 verl 生态的兼容性。

约束：仅修改 recipe/MRSD/ 目录。
"""

from __future__ import annotations

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class MRSDActorRolloutWorker(ActorRolloutRefWorker):
    """
    MRSD Actor + Rollout Worker。

    与基类 ActorRolloutRefWorker 的唯一区别：
      - init_model 中将 DataParallelPPOActor 替换为 MRSDPPOActor

    注意：必须保留 @register 装饰器，否则 MAGIC_ATTR 丢失，
    RayWorkerGroup 无法发现并绑定该方法。
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        调用父类 init_model，然后用 MRSDPPOActor 替换已创建的 self.actor。
        父类已经：
          1. 构建 FSDP 模型 self.actor_module_fsdp
          2. 构建 optimizer、lr_scheduler
          3. 构建 vllm rollout（如果 _is_rollout）
          4. 创建 self.actor = DataParallelPPOActor(...)
        我们只需在此基础上替换 self.actor。
        """
        super().init_model()

        # 只有 actor role 才需要替换
        if not self._is_actor:
            return

        # 延迟 import 避免循环依赖
        import sys
        from pathlib import Path
        _recipe_root = Path(__file__).parent.parent.parent.parent
        if str(_recipe_root) not in sys.path:
            sys.path.insert(0, str(_recipe_root))

        from recipe.MRSD.mrsd.mrsd_actor import MRSDPPOActor

        # 用 MRSDPPOActor 替换标准 DataParallelPPOActor
        # 保持相同的 config、actor_module、optimizer 参数
        self.actor = MRSDPPOActor(
            config=self.config.actor,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )
