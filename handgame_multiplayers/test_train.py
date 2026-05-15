#!/usr/bin/env python3
"""快速训练测试脚本 - 用于调试当前配置是否能正常运行"""

import sys
import torch
from ai import (
    BattleNet,
    BattleGenome,
    TrainConfig,
    TrainSearchConfig,
    train_selfplay,
)


def test_train():
    """运行一个简短的训练测试"""
    print("=" * 60)
    print("Starting training test")
    print("=" * 60)
    print()

    # 使用减小的配置以加速测试
    config = TrainConfig(
        episodes=5,  # 只训练 5 局快速测试
        max_rounds=10,  # 每局最多 10 回合
        batch_size=32,  # 小批次
        replay_size=256,  # 小经验池
        search_depth=1,  # 浅层搜索（快速）
        max_search_depth=2,  # 最多 2 层
        cfr_iters=8,  # 少迭代（快速）
        depth_increase_patience=50,  # 不容易升深度
        plot_interval=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_search=TrainSearchConfig(
            cfr_iters=8,
            frontier_min_actions=2,
        ),
    )

    print(f"Configuration:")
    print(f"  - Episodes: {config.episodes}")
    print(f"  - Max rounds: {config.max_rounds}")
    print(f"  - Search depth: {config.search_depth}")
    print(f"  - Device: {config.device}")
    print(f"  - CFR iters: {config.cfr_iters}")
    print(f"  - Train search CFR iters: {config.train_search.cfr_iters}")
    print()

    try:
        print("[1/3] Initializing model...")
        model = BattleNet(BattleGenome())
        print(f"      Model parameters: {sum(p.numel() for p in model.parameters())}")
        print()

        print("[2/3] Starting training...")
        trainer = train_selfplay(model, config)
        print()

        print("[3/3] Training statistics:")
        print(f"      Total loss records: {len(trainer.loss_history)}")
        if trainer.loss_history:
            print(f"      Initial loss: {trainer.loss_history[0]:.4f}")
            print(f"      Final loss: {trainer.loss_history[-1]:.4f}")
            print(f"      Minimum loss: {min(trainer.loss_history):.4f}")
        print()

        print("=" * 60)
        print("[OK] Training test completed successfully!")
        print("=" * 60)
        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("[FAIL] Training test failed!")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print()

        # 打印完整的错误堆栈
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_train()
    sys.exit(0 if success else 1)

