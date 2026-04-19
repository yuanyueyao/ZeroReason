"""
快速诊断 wandb 连接状况的独立测试脚本。
用法：python3 recipe/my_project/test_wandb.py
"""
import os
import time

print("=== wandb 诊断脚本 ===")
print(f"WANDB_INIT_TIMEOUT env: {os.environ.get('WANDB_INIT_TIMEOUT', '(未设置)')}")
print(f"WANDB_MODE env: {os.environ.get('WANDB_MODE', '(未设置)')}")

import wandb

print(f"\nwandb 版本: {wandb.__version__}")

# 1. 检查登录状态
print("\n[1] 检查登录状态...")
try:
    api = wandb.Api()
    viewer = api.viewer
    print(f"  已登录，用户: {viewer}")
except Exception as e:
    print(f"  登录检查失败: {e}")

# 2. 测试 wandb.init，带更长的超时
print("\n[2] 尝试 wandb.init（timeout=180s）...")
t0 = time.time()
try:
    run = wandb.init(
        project="verl_fit_competition",
        name="test_connection",
        settings=wandb.Settings(init_timeout=180),
        mode="online",
    )
    elapsed = time.time() - t0
    print(f"  成功！耗时 {elapsed:.1f}s，run_id={run.id}")
    run.finish()
    print("  run.finish() 完成")
except Exception as e:
    elapsed = time.time() - t0
    print(f"  失败，耗时 {elapsed:.1f}s，错误: {e}")
    print("\n  建议：网络不稳定，尝试 offline 模式")
    print("  可在 run.sh 里加：export WANDB_MODE=offline")
    print("  或改 trainer.logger=['console'] 去掉 wandb")
