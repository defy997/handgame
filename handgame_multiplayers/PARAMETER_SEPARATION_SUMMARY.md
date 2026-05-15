# 参数分离改进总结

## 问题症状

用户的配置：
```python
SearchConfig(
    depth=6,
    cfr_iters=32,
    frontier_min_actions=1  # ← 这在推理时太少了
)

TrainConfig(
    cfr_iters=16,
    training_frontier_min_actions=2  # ← 这和推理不一样
)
```

**症状：**
- depth > 1 时容易卡死（已修复：缓存 key 和深度传递问题）
- 不清楚哪些参数是用于推理的，哪些用于训练的
- 难以为不同场景（单人/多人/不同设备）单独调整参数
- 训练和推理时用的参数不一致

## 改进方案

### 1. 新增 `TrainSearchConfig` 类

```python
@dataclass
class TrainSearchConfig:
    """训练时的搜索配置。"""
    cfr_iters: int = 16
    frontier_low: float = 0.5
    frontier_high: float = 1.0
    frontier_min_actions: int = 2
```

### 2. 重新定义 `SearchConfig`

现在明确用于**推理**：

```python
@dataclass
class SearchConfig:
    """推理时搜索配置（实际运行时使用）。"""
    depth: int = 2              # ← 推理用的深度
    cfr_iters: int = 16         # ← 推理用的 CFR 迭代
    cache_size: int = 65536
    value_mix: float = 0.7
    frontier_low: float = 0.5
    frontier_high: float = 1.0
    frontier_min_actions: int = 1  # ← 推理时保留少一点
```

### 3. MinimaxBattleAI 支持双参数配置

```python
class MinimaxBattleAI:
    def __init__(
        self,
        genome: BattleGenome,
        search: SearchConfig,              # 推理参数
        train_search: Optional[TrainSearchConfig] = None,  # 训练参数
    ):
        self.search = search
        self.train_search = train_search or TrainSearchConfig()
        self._is_training = False  # 模式标志
```

### 4. 智能参数获取方法

```python
def _get_cfr_iters(self) -> int:
    """根据当前模式返回合适的 CFR 迭代次数"""
    if self._is_training:
        return self.train_search.cfr_iters
    return self.search.cfr_iters

def _get_frontier_params(self) -> Tuple[float, float, int]:
    """根据当前模式返回合适的前沿筛选参数"""
    if self._is_training:
        return (self.train_search.frontier_low,
                self.train_search.frontier_high,
                self.train_search.frontier_min_actions)
    return (self.search.frontier_low,
            self.search.frontier_high,
            self.search.frontier_min_actions)
```

### 5. TrainConfig 整合 TrainSearchConfig

```python
@dataclass
class TrainConfig:
    # ... 其他训练超参数 ...
    train_search: TrainSearchConfig = field(default_factory=TrainSearchConfig)
```

## 改进对比表

| 方面 | 改进前 | 改进后 |
|------|--------|--------|
| **参数混乱** | `SearchConfig` 中混合推理和训练参数 | `SearchConfig` 只用于推理，`TrainSearchConfig` 用于训练 |
| **参数冲突** | `frontier_min_actions=1` vs `training_frontier_min_actions=2` 容易出错 | 明确分离，通过 `_is_training` 标志自动切换 |
| **配置方式** | 需要用 `getattr()` 动态读取旧参数 | 直接在初始化时传入，清晰明了 |
| **多场景支持** | 难以为不同人数/设备调整 | 可以为推理和训练分别优化 |
| **代码可维护性** | 容易遗漏参数或设置错误 | 参数职责明确，易于维护 |
| **向后兼容** | N/A | 旧的 `training_frontier_*` 参数保留但已弃用 |

## 代码重构检查清单

- [x] 创建 `TrainSearchConfig` 类
- [x] 更新 `SearchConfig` 文档（推理专用）
- [x] 修改 `MinimaxBattleAI.__init__()` 接收两套配置
- [x] 添加 `_is_training` 模式标志
- [x] 实现 `_get_cfr_iters()` 方法
- [x] 实现 `_get_frontier_params()` 方法
- [x] 更新 `_cfr_root_policy()` 使用 `_get_cfr_iters()`
- [x] 更新 `_select_frontier_actions()` 使用 `_get_frontier_params()`
- [x] 更新 `TrainConfig` 包含 `train_search` 字段
- [x] 更新 `SelfPlayTrainer.__init__()` 传入两套配置
- [x] 更新 `SelfPlayTrainer._search_policy()` 使用模式标志
- [x] 修复 `_evaluate_pair()` 缓存 key 不匹配问题
- [x] 修复 `_evaluate_state()` 深度参数递减问题
- [x] 修复 `_select_frontier_actions()` 无限循环问题
- [x] 验证所有参数配置能正常初始化
- [x] 创建参数配置指南文档

## 性能影响

- **内存**：无增加（参数对象本身很小）
- **速度**：无损耗（参数获取是 O(1) 操作）
- **代码行数**：+~20 行（新方法）
- **可读性**：+↑↑↑（参数职责明确）

## 迁移指南

### 如果你之前用旧代码训练了模型

```python
# 旧方式（现在过时了）
TrainConfig(
    cfr_iters=16,
    training_frontier_low=0.5,
    training_frontier_high=1.0,
    training_frontier_min_actions=2,
)

# 新方式（推荐）
TrainConfig(
    cfr_iters=16,  # 保留用于向后兼容（但不使用）
    train_search=TrainSearchConfig(
        cfr_iters=16,
        frontier_low=0.5,
        frontier_high=1.0,
        frontier_min_actions=2,
    ),
)
```

## 测试代码

```python
from ai import SearchConfig, TrainSearchConfig, MinimaxBattleAI, BattleGenome

# 创建 AI，指定推理和训练参数
ai = MinimaxBattleAI(
    BattleGenome(),
    search=SearchConfig(depth=2, cfr_iters=16),
    train_search=TrainSearchConfig(cfr_iters=32),
)

# 推理模式（默认）
assert ai._is_training == False
cfr_iters = ai._get_cfr_iters()
assert cfr_iters == 16  # 使用推理参数

# 训练模式
ai._is_training = True
cfr_iters = ai._get_cfr_iters()
assert cfr_iters == 32  # 使用训练参数

# 记得关闭训练模式
ai._is_training = False
```

## 后续优化方向

1. **参数预设库**
   ```python
   PRESET_CONFIGS = {
       'fast': SearchConfig(depth=1, cfr_iters=8),
       'balanced': SearchConfig(depth=2, cfr_iters=16),
       'strong': SearchConfig(depth=3, cfr_iters=32),
   }
   ```

2. **自适应参数调整**
   - 根据设备性能自动调整 depth
   - 根据游戏人数自动调整 frontier_min_actions

3. **参数验证器**
   - 确保推理参数优先级 depth > cfr_iters > frontier
   - 确保训练参数合理性

4. **配置导出/导入**
   - 保存最优配置为 JSON
   - 在不同设备间迁移配置
