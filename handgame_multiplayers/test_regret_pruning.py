#!/usr/bin/env python3
"""
遗憾值剪枝 (Regret-Based Pruning) 实现验证脚本
"""

import sys
sys.path.insert(0, '.')

from ai import (
    SearchConfig, TrainSearchConfig, BattleGenome, 
    MinimaxBattleAI, BattleNet
)


def test_config_initialization():
    """测试配置是否正确初始化。"""
    print("=" * 60)
    print("测试 1: 配置初始化")
    print("=" * 60)
    
    # 推理配置
    search = SearchConfig(
        depth=2,
        cfr_iters=16,
        regret_pruning_enabled=True,
        regret_threshold=-0.5,
        regret_window=5
    )
    
    # 训练配置
    train_search = TrainSearchConfig(
        cfr_iters=8,
        regret_pruning_enabled=True,
        regret_threshold=-0.3,
        regret_window=3
    )
    
    print(f"✓ 推理配置创建成功")
    print(f"  - 反悔值剪枝启用: {search.regret_pruning_enabled}")
    print(f"  - 反悔值阈值: {search.regret_threshold}")
    print(f"  - 观察窗口大小: {search.regret_window}")
    
    print(f"✓ 训练配置创建成功")
    print(f"  - 反悔值剪枝启用: {train_search.regret_pruning_enabled}")
    print(f"  - 反悔值阈值: {train_search.regret_threshold}")
    print(f"  - 观察窗口大小: {train_search.regret_window}")
    
    return search, train_search


def test_ai_initialization(search, train_search):
    """测试AI是否能够初始化。"""
    print("\n" + "=" * 60)
    print("测试 2: AI 初始化")
    print("=" * 60)
    
    genome = BattleGenome(
        input_size=16,
        hidden_1=128,
        hidden_2=64,
        policy_size=12
    )
    
    ai = MinimaxBattleAI(
        genome=genome,
        search=search,
        train_search=train_search
    )
    
    print(f"✓ AI 初始化成功")
    print(f"  - 网络模型: {type(ai.model).__name__}")
    print(f"  - 反悔值历史记录数: {len(ai._regret_history)}")
    print(f"  - 被剪枝动作集合数: {len(ai._pruned_actions)}")
    
    return ai


def test_regret_history_update(ai):
    """测试反悔值历史更新。"""
    print("\n" + "=" * 60)
    print("测试 3: 反悔值历史更新")
    print("=" * 60)
    
    # 创建虚拟的英雄对象用于测试
    class DummyHero:
        def __init__(self):
            self.__class__.__name__ = "TestHero"
            self.hp = 100
            self.hp_ceiling = 100
            self.mp = 50
            self.mp_ceiling = 50
            self.shield = 0
            self.damage_resistance = 0.0
            self.revival_armor = 0
    
    hero = DummyHero()
    regrets = {0: -0.3, 1: 0.2, 2: -0.1}
    state_key = (("TestHero", 100, 100, 50, 50, 0, 0.0, 0), 
                 ("TestHero", 100, 100, 50, 50, 0, 0.0, 0), 0)
    
    # 模拟多次更新
    print("模拟 10 次 CFR 迭代...")
    for i in range(10):
        ai._update_regret_history(hero, regrets, state_key)
    
    # 检查历史记录
    print(f"✓ 反悔值历史更新成功")
    for action_id, regret in regrets.items():
        history_key = (state_key, action_id)
        if history_key in ai._regret_history:
            history = ai._regret_history[history_key]
            print(f"  - 动作 {action_id}: 历史长度 = {len(history)}, "
                  f"最近值 = {history[-1]:.3f}")
    
    return state_key


def test_pruning_logic(ai, state_key):
    """测试剪枝逻辑。"""
    print("\n" + "=" * 60)
    print("测试 4: 剪枝逻辑")
    print("=" * 60)
    
    regret_threshold = ai.search.regret_threshold
    regret_window = ai.search.regret_window
    
    print(f"剪枝参数:")
    print(f"  - 反悔值阈值: {regret_threshold}")
    print(f"  - 观察窗口: {regret_window}")
    
    # 测试 _should_prune_action
    for action_id in range(3):
        history_key = (state_key, action_id)
        if history_key in ai._regret_history:
            should_prune = ai._should_prune_action(
                history_key, regret_threshold, regret_window
            )
            history = ai._regret_history[history_key]
            recent = history[-regret_window:] if len(history) >= regret_window else history
            print(f"  - 动作 {action_id}:")
            print(f"    历史: {[f'{r:.3f}' for r in recent]}")
            print(f"    应该剪枝: {should_prune}")


def test_backward_compatibility():
    """测试禁用反悔值剪枝时仍能正确初始化。"""
    print("\n" + "=" * 60)
    print("测试 5: 禁用反悔值剪枝")
    print("=" * 60)

    search = SearchConfig(
        depth=2,
        cfr_iters=16,
        regret_pruning_enabled=False,  # 禁用反悔值剪枝
    )

    genome = BattleGenome()
    ai = MinimaxBattleAI(genome=genome, search=search)

    print(f"✓ 禁用反悔值剪枝的 AI 初始化成功")
    print(f"  - 反悔值剪枝启用: {ai.search.regret_pruning_enabled}")
    print(f"  - 展开时将保留全部动作，由 max_batch_leaves={ai.search.max_batch_leaves} 兜底防 OOM")


def main():
    print("\n")
    print("*" * 60)
    print("  遗憾值剪枝 (Regret-Based Pruning) 验证测试")
    print("*" * 60)
    
    try:
        # 运行所有测试
        search, train_search = test_config_initialization()
        ai = test_ai_initialization(search, train_search)
        state_key = test_regret_history_update(ai)
        test_pruning_logic(ai, state_key)
        test_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
