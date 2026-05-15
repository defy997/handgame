# 参数分离总结 - 快速参考

## 核心变更

### 之前（混乱）
```python
# 搜索配置中混合了推理和训练参数
SearchConfig(
    depth=2,
    cfr_iters=32,
    frontier_low=0.5,           # ← 这是训练参数吗？推理参数吗？
    frontier_high=1.0,
    frontier_min_actions=2,
)

# TrainConfig 中又有冗余的 training_frontier_* 参数
TrainConfig(
    cfr_iters=16,
    training_frontier_low=0.5,  # ← 重复，容易出错
    training_frontier_high=1.0,
    training_frontier_min_actions=2,
)
```

### 现在（清晰）
```python
# 推理配置 - 用于游戏运行时
SearchConfig(
    depth=2,
    cfr_iters=16,
    frontier_low=0.5,
    frontier_high=1.0,
    frontier_min_actions=1,
)

# 训练配置 - 用于自博弈训练
TrainSearchConfig(
    cfr_iters=16,
    frontier_low=0.5,
    frontier_high=1.0,
    frontier_min_actions=2,
)

# 训练超参数 - 用于优化器、学习率等
TrainConfig(
    episodes=1000,
    search_depth=1,
    max_search_depth=6,
    train_search=TrainSearchConfig(...),  # 明确指定训练搜索参数
)
```

## 三层参数体系

```
MinimaxBattleAI
├─ search: SearchConfig           ← 推理模式（choose_action 使用）
├─ train_search: TrainSearchConfig ← 训练模式（_search_policy 使用）
└─ _is_training: bool             ← 模式标志

SelfPlayTrainer
├─ config: TrainConfig
│  └─ train_search: TrainSearchConfig
└─ search_agent: MinimaxBattleAI  ← 共享 AI，通过 _is_training 切换模式
```

## 使用流程

### 游戏运行（推理模式）
```
Game.choose_action()
  ↓
MinimaxBattleAI.choose_action()
  ├─ _is_training = False        (默认)
  ├─ 使用 self.search 参数
  └─ 返回 ActionPlan

参数来源: SearchConfig
```

### 模型训练（训练模式）
```
SelfPlayTrainer._play_episode()
  ├─ _search_policy()
  │  ├─ self.search_agent._is_training = True
  │  ├─ 使用 self.train_search 参数
  │  └─ self.search_agent._is_training = False (finally)
  │
  └─ 参数来源: TrainSearchConfig
```

## 关键方法

```python
# MinimaxBattleAI 中的参数获取方法
def _get_cfr_iters(self) -> int:
    """根据 _is_training 返回相应的 CFR 迭代次数"""
    if self._is_training:
        return self.train_search.cfr_iters
    return self.search.cfr_iters

def _get_frontier_params(self) -> Tuple[float, float, int]:
    """根据 _is_training 返回相应的前沿筛选参数"""
    if self._is_training:
        return (self.train_search.frontier_low,
                self.train_search.frontier_high,
                self.train_search.frontier_min_actions)
    return (self.search.frontier_low,
            self.search.frontier_high,
            self.search.frontier_min_actions)
```

## 配置调整流程

```
需求分析
├─ 游戏类型（单人/多人）
├─ 设备性能（低端/中端/高端）
└─ 响应时间要求（实时/离线）
  ↓
调整 SearchConfig
├─ depth: 设备性能 + 游戏复杂度
├─ cfr_iters: 响应时间要求
├─ frontier_*: 动作搜索范围
└─ cache_size: 内存限制
  ↓
调整 TrainSearchConfig（可选）
├─ cfr_iters: 训练速度 vs 质量 trade-off
└─ frontier_*: 训练数据多样性
  ↓
实验验证 (evaluate_battle)
```

## 参数优先级

1. **SearchConfig.depth** - 影响最大（搜索树大小）
   - 每增加 1，计算量 ∝ action_count^depth

2. **SearchConfig.cfr_iters** - 影响其次（CFR 收敛质量）
   - 影响纳什均衡近似精度

3. **frontier_min_actions** - 影响最小（剪枝强度）
   - 只影响搜索宽度，不影响深度

## 常见陷阱 ❌

- ❌ 直接修改 `search_agent.search.cfr_iters` 而不使用 `_get_cfr_iters()`
- ❌ 在训练时忘记设置 `_is_training = True`
- ❌ 在运行时混用训练和推理参数
- ❌ 假设旧的 `training_frontier_*` 参数仍然有效

## 验证代码工作 ✓

```python
# 确认参数分离正确
from ai import MinimaxBattleAI, SearchConfig, TrainSearchConfig, BattleGenome

ai = MinimaxBattleAI(
    BattleGenome(),
    SearchConfig(depth=2, cfr_iters=16),
    TrainSearchConfig(cfr_iters=32),
)

# 推理模式
assert ai._is_training == False
assert ai._get_cfr_iters() == 16  ✓

# 训练模式
ai._is_training = True
assert ai._get_cfr_iters() == 32  ✓
ai._is_training = False
```
