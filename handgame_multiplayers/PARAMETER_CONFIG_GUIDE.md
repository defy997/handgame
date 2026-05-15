# AI 参数配置指南 - 训练vs推理参数分离

## 问题背景

之前的代码中，训练参数和推理参数混在一起，导致：
- 参数体系混乱，难以维护
- 训练和推理时使用不一致的参数
- 难以针对不同场景（单人、多人、不同设备）优化参数

## 解决方案：双参数体系

### 1. 推理模式参数 (`SearchConfig`)

用于**实际游戏运行时**的搜索配置。可根据设备人数调整：

```python
from ai import SearchConfig

# 单人快速响应（低延迟）
inference_config = SearchConfig(
    depth=2,              # 展开深度（浅）
    cfr_iters=16,         # CFR 迭代次数（少）
    frontier_low=0.5,
    frontier_high=1.0,
    frontier_min_actions=1,
)

# 多人或高端设备（高质量）
inference_config_high = SearchConfig(
    depth=4,              # 更深的搜索
    cfr_iters=32,         # 更多 CFR 迭代
    frontier_low=0.3,     # 更多候选动作
    frontier_high=1.0,
    frontier_min_actions=2,
)
```

### 2. 训练模式参数 (`TrainSearchConfig`)

用于**自博弈训练时**的搜索配置。在 `TrainConfig` 中指定：

```python
from ai import TrainConfig, TrainSearchConfig

config = TrainConfig(
    episodes=1000,
    max_rounds=30,
    search_depth=1,           # 训练初始深度
    max_search_depth=6,       # 动态提升上限
    
    # 训练时搜索参数
    train_search=TrainSearchConfig(
        cfr_iters=16,         # 训练时用较少 CFR 迭代以加速
        frontier_low=0.5,
        frontier_high=1.0,
        frontier_min_actions=2,
    ),
)
```

## 参数职责分工表

| 参数组件 | 职责 | 使用场景 | 默认值 |
|---------|------|---------|--------|
| `SearchConfig` | 推理时搜索策略 | 游戏运行、choose_action | depth=2, cfr_iters=16 |
| `TrainSearchConfig` | 训练时搜索策略 | 自博弈生成训练数据 | cfr_iters=16, frontier_min_actions=2 |
| `TrainConfig` | 训练超参数 | 神经网络训练循环 | 1000 episodes, lr=1e-3 |

## MinimaxBattleAI 两种工作模式

```python
from ai import MinimaxBattleAI, SearchConfig, TrainSearchConfig

# 初始化时指定两套配置
ai = MinimaxBattleAI(
    genome=genome,
    search=SearchConfig(depth=2),           # 推理配置
    train_search=TrainSearchConfig(...)     # 训练配置
)

# 推理模式（默认）
ai._is_training = False
action_plan = ai.choose_action(hero, enemy)  # 使用 search 参数

# 训练模式
ai._is_training = True
policy = ai._cfr_root_policy(hero, enemy)    # 使用 train_search 参数
ai._is_training = False  # 记得关闭
```

## 使用示例

### 示例 1：快速开发（单人，低延迟）

```python
from ai import BattleNet, MinimaxBattleAI, SearchConfig, TrainSearchConfig, BattleGenome
from game import Game
from heros.basic import BasicHero

# 推理用的轻量级配置
inference = SearchConfig(
    depth=1,
    cfr_iters=8,
    frontier_min_actions=1,
)

# 创建 AI
model = BattleNet(BattleGenome())
ai = MinimaxBattleAI(BattleGenome(), inference)

# 游戏运行
game = Game(BasicHero("Player"), BasicHero("AI"))
game.set_player_ai(1, profile_name="quick")
```

### 示例 2：训练新模型

```python
from ai import BattleNet, TrainConfig, TrainSearchConfig, train_selfplay

# 训练配置：逐步加深
train_config = TrainConfig(
    episodes=5000,
    search_depth=1,
    max_search_depth=4,
    train_search=TrainSearchConfig(
        cfr_iters=16,
        frontier_low=0.5,
        frontier_high=1.0,
        frontier_min_actions=2,
    ),
)

model = BattleNet()
trainer = train_selfplay(model, train_config)
```

### 示例 3：多人游戏（设备好的情况）

```python
from ai import SearchConfig

# 多人时需要更强的 AI 以应对复杂局面
high_quality_config = SearchConfig(
    depth=3,
    cfr_iters=32,
    cache_size=131072,  # 2x 缓存
    frontier_low=0.3,   # 保留更多候选
    frontier_high=1.0,
    frontier_min_actions=2,
)

# 分别配置 2 个 AI
game.set_player_ai(1, 
    genome=BattleGenome(),
    search=high_quality_config,
)
game.set_player_ai(2,
    genome=BattleGenome(),
    search=high_quality_config,
)
```

## 参数优化建议

### 针对设备性能调整

**低端设备（移动/旧 PC）：**
```python
SearchConfig(
    depth=1,
    cfr_iters=8,
    cache_size=8192,
    frontier_min_actions=1,
)
```

**中端设备（标准 PC）：**
```python
SearchConfig(
    depth=2,
    cfr_iters=16,
    cache_size=65536,
    frontier_min_actions=1,
)
```

**高端设备（GPU / 多核）：**
```python
SearchConfig(
    depth=4,
    cfr_iters=32,
    cache_size=131072,
    frontier_low=0.2,
    frontier_high=1.0,
    frontier_min_actions=3,
)
```

### 针对游戏人数调整

| 人数 | 推荐 depth | 推荐 cfr_iters | 原因 |
|------|-----------|----------------|------|
| 1v1（对战） | 2-3 | 16-32 | 简单局面，快速响应 |
| 2v2 | 2 | 16 | 局面复杂度中等 |
| 多人（3+） | 1-2 | 8-16 | 计算复杂度指数增长 |

## 向后兼容性

旧代码中的 `training_frontier_*` 参数已弃用，但为了兼容性：
- 仍在 `TrainConfig` 中可见
- 内部已替换为 `train_search` 配置
- 建议逐步迁移到新体系

## 常见问题

**Q: 如何为不同人数设置不同的 AI？**

A: 在 `game.py` 中分别调用 `set_player_ai()` 传入不同的 `SearchConfig`。

**Q: 训练时需要手动设置 `_is_training`？**

A: 不需要。`SelfPlayTrainer` 会自动在 `_search_policy()` 中切换模式。

**Q: 如何测试参数效果？**

A: 使用 `evaluate_battle()` 函数对比两个配置的胜率。

```python
from ai import evaluate_battle

config1 = SearchConfig(depth=2, cfr_iters=16)
config2 = SearchConfig(depth=3, cfr_iters=32)

metrics = evaluate_battle(model1, model2, episodes=100, search_depth=2)
print(f"Model1 胜率: {metrics['win_rate']:.2%}")
```
