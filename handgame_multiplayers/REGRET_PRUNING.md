# 遗憾值剪枝 (Regret-Based Pruning) 策略

## 概述

本实现采用**博弈安全的遗憾值剪枝策略**替代原有的静态分剪枝，这是一种更灵活且理论上更健全的方法。

### 核心优势

1. **博弈安全**: 不会永久排除某个动作。如果环境变化，被剪枝的分支可以自动重新激活。
2. **自适应**: 根据历史反悔值动态调整，而非依赖静态的网络评估分数。
3. **梯度性**: 不是二进制的"剪枝/保留"，而是基于持续的反悔值累积做出决策。

## 工作原理

### 反悔值的定义

在CFR (Counterfactual Regret Minimization) 框架中，**反悔值**定义为：

$$\text{regret}_a = \max(0, v_a - v_{\text{avg}})$$

其中：
- $v_a$ 是动作$a$的期望价值
- $v_{\text{avg}}$ 是当前策略下的平均价值

负的反悔值表示该动作比平均情况更差，即"无论如何都亏"。

### 剪枝条件

一个动作 $a$ 在状态 $s$ 下被剪枝，当且仅当：

$$\text{regret}_a[t] < \text{threshold}, \quad \forall t \in [i-w+1, i]$$

其中：
- $w$ 是观察窗口大小（连续 $w$ 次迭代）
- $\text{threshold}$ 是反悔值阈值（通常为负值，如 -0.5）

### 重新激活机制

如果所有候选动作都被剪枝（或剩余动作太少），系统会根据**历史累积反悔值**重新激活表现最好的动作，确保总是保留最少数量的候选。

## 配置参数

### SearchConfig（推理时配置）

```python
@dataclass
class SearchConfig:
    # ... 其他参数 ...
    regret_pruning_enabled: bool = True          # 启用反悔值剪枝
    regret_threshold: float = -0.5               # 反悔值阈值
    regret_window: int = 5                       # 连续观察窗口大小
```

### TrainSearchConfig（训练时配置）

```python
@dataclass
class TrainSearchConfig:
    # ... 其他参数 ...
    regret_pruning_enabled: bool = True          # 训练时是否启用
    regret_threshold: float = -0.5               # 训练时的反悔值阈值
    regret_window: int = 5                       # 训练时的窗口大小
```

## 参数调整指南

### regret_threshold

- **范围**: 通常在 -1.0 ~ 0.0 之间
- **较低值**（如 -0.2）: 更激进的剪枝，探索较少
- **较高值**（如 -1.0）: 保守的剪枝，探索更多
- **推荐**: -0.5 ~ -0.3（平衡探索和剪枝）

### regret_window

- **范围**: 通常在 3 ~ 10 之间
- **较小值**（如 3）: 剪枝更快，但可能过于激进
- **较大值**（如 10）: 需要更多历史证据，剪枝更谨慎
- **推荐**: 5 ~ 7（需要充分的历史证据）

## 反向兼容性

如果需要禁用反悔值剪枝回退到静态分剪枝，只需设置：

```python
search = SearchConfig(regret_pruning_enabled=False)
```

此时将自动使用原有的 frontier 参数（frontier_low, frontier_high）进行静态分剪枝。

## 实现细节

### 关键方法

- `_apply_regret_based_pruning()`: 执行反悔值剪枝
- `_should_prune_action()`: 判断动作是否满足剪枝条件
- `_update_regret_history()`: 记录反悔值历史
- `_get_action_regret_sum()`: 计算动作的累积反悔值

### 内存管理

- 每个状态-动作对保留最多 50 次迭代的反悔值历史，防止内存溢出
- 被剪枝的动作集合按状态维护，避免跨状态污染

## 性能影响

- **正面**: 减少不必要的搜索分支，加快决策速度
- **中性**: 反悔值记录和查询的开销很小（O(1) 平均情况）
- **负面**: 初期可能需要积累足够的历史才能有效剪枝

## 实验建议

1. **基准对比**: 对比启用和禁用反悔值剪枝的性能
2. **参数扫描**: 尝试不同的 threshold 和 window 组合
3. **收敛监控**: 观察 AI 的学习曲线和决策多样性变化
4. **对抗测试**: 验证重新激活机制在环境变化时的表现

## 参考文献

- Hart, S., & Mas-Colell, A. (2000). "A simple adaptive procedure leading to correlated equilibrium"
- Lanctot, M., et al. (2009). "Monte Carlo Tree Search in real-time games"
