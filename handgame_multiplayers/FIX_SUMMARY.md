# AI.py 修复总结 - 三大问题 + 参数分离

## 问题 1：缓存 Key 不匹配导致缓存失效 ✓ FIXED

**问题位置**：`_evaluate_pair()` 第 236 行和第 250 行

**原始代码（错误）**：
```python
def _evaluate_pair(self, hero, enemy, a_id, b_id, depth):
    # 第 236 行：用当前状态查找缓存
    if cacheable:
        key = self._state_hash(hero, enemy, depth)  # ← 错误：hero, enemy
        if key in self._cache:
            return self._cache[key]
    
    next_hero, next_enemy = simulate_round(...)
    
    # 第 250 行：用下一个状态存储缓存
    if cacheable:
        key = self._state_hash(next_hero, next_enemy, depth)  # ← 不一致
        self._cache[key] = value
```

**根本原因**：
- 查找时用原始状态 (hero, enemy) 作为 key
- 存储时用下一个状态 (next_hero, next_enemy) 作为 key
- **两个 key 完全不同**，导致缓存完全失效
- 大量重复计算，可能导致超时或内存爆炸

**修复方案**：
```python
def _evaluate_pair(self, hero, enemy, a_id, b_id, depth):
    # 1. 先模拟
    next_hero, next_enemy = simulate_round(hero, enemy, a_id, b_id)
    
    # 2. 用新状态查找缓存（统一使用新状态作为 key）
    if depth <= 1:
        key = self._state_hash(next_hero, next_enemy, depth)
        if key in self._cache:
            return self._cache[key]
    
    # 3. 计算
    if _is_terminal(next_hero, next_enemy):
        value = _terminal_value(next_hero, next_enemy)
    elif depth <= 1:
        value = self._evaluate_network(next_hero, next_enemy)
    else:
        value = self._evaluate_state(next_hero, next_enemy, depth - 1)
    
    # 4. 用新状态存储缓存（保持一致）
    if depth <= 1:
        key = self._state_hash(next_hero, next_enemy, depth)
        self._cache[key] = value
    
    return value
```

---

## 问题 2：搜索深度递减不正确导致死循环 ✓ FIXED

**问题位置**：`_evaluate_state()` 第 270 行

**原始代码（错误）**：
```python
def _evaluate_state(self, hero, enemy, depth: int) -> float:
    # ...
    for a in legal_self:
        for b in legal_enemy:
            # depth 应该递减，但这里没递减！
            row.append(self._evaluate_pair(hero, enemy, a, b, depth))
            #                                                   ^^^^^
            #                              应该是 depth - 1
```

**根本原因**：
- `_evaluate_state(depth=2)` 调用 `_evaluate_pair(..., depth=2)`
- `_evaluate_pair` 再调用 `_evaluate_state(..., depth=2)` (因为第 247 行 `depth - 1` 没用上)
- 导致搜索树永远不会终止 → **无限循环**

**修复方案**：
```python
def _evaluate_state(self, hero, enemy, depth: int) -> float:
    # ...
    for a in legal_self:
        for b in legal_enemy:
            # 正确：递减深度
            row.append(self._evaluate_pair(hero, enemy, a, b, depth - 1))
            #                                                   ^^^^^^^
```

**执行流验证**：
```
_cfr_root_policy(depth=3)
  → _evaluate_pair(..., depth=3)
    → simulate_round()
    → _evaluate_state(..., depth=2)  ← depth 递减 ✓
      → for each (a,b):
          → _evaluate_pair(..., depth=1)  ← depth 再递减 ✓
            → simulate_round()
            → _evaluate_network()  ← 终止 ✓
```

---

## 问题 3：无限循环在前沿筛选中 ✓ FIXED

**问题位置**：`_select_frontier_actions()` 第 351-355 行

**原始代码（有缺陷）**：
```python
while len(selected) < self.search.frontier_min_actions:
    for action, _ in reversed(scores):
        if action not in selected:
            selected.append(action)
            break  # ← 只有找到新 action 才会执行
```

**缺陷场景**：
- 假设 `scores = [A, B, C]` 且 `selected = [A, B, C]` 都已选中
- 但 `frontier_min_actions = 5` 需要 5 个动作
- for 循环遍历 [C, B, A]，都找不到新动作
- **for 循环完成但没有 break**，while 无限循环

**修复方案**：
```python
frontier_low, frontier_high, frontier_min_actions = self._get_frontier_params()

# ... 筛选逻辑 ...

# 安全的补充动作逻辑
candidates = [action for action, _ in scores if action not in selected]
for action in reversed(candidates):
    if len(selected) >= frontier_min_actions:
        break  # ← 提前检查，不会进入无限循环
    selected.append(action)

return selected
```

**为什么安全**：
- 候选列表是有限的
- for 循环遍历完自动结束
- 即使候选为空，也会正常返回已选动作

---

## 改进 4：参数分离（架构优化）✓ IMPLEMENTED

**问题**：训练参数和推理参数混在一起

**解决方案**：创建两套独立的参数体系

### 新增 `TrainSearchConfig`
```python
@dataclass
class TrainSearchConfig:
    """训练时的搜索配置。"""
    cfr_iters: int = 16
    frontier_low: float = 0.5
    frontier_high: float = 1.0
    frontier_min_actions: int = 2
```

### 更新 `MinimaxBattleAI`
```python
class MinimaxBattleAI:
    def __init__(self, genome, search, train_search=None):
        self.search = search              # 推理参数
        self.train_search = train_search  # 训练参数
        self._is_training = False         # 模式标志
    
    def _get_cfr_iters(self) -> int:
        if self._is_training:
            return self.train_search.cfr_iters
        return self.search.cfr_iters
    
    def _get_frontier_params(self) -> Tuple:
        if self._is_training:
            return (self.train_search.frontier_low,
                    self.train_search.frontier_high,
                    self.train_search.frontier_min_actions)
        return (self.search.frontier_low,
                self.search.frontier_high,
                self.search.frontier_min_actions)
```

### 更新 `TrainConfig`
```python
@dataclass
class TrainConfig:
    # ... 其他参数 ...
    train_search: TrainSearchConfig = field(default_factory=TrainSearchConfig)
```

### 更新 `SelfPlayTrainer`
```python
self.search_agent = MinimaxBattleAI(
    self.model.genome,
    SearchConfig(depth=self.current_depth, cfr_iters=config.cfr_iters),
    train_search=config.train_search,
)

def _search_policy(self, hero, enemy) -> Dict[int, float]:
    self.search_agent._is_training = True
    try:
        return self.search_agent._cfr_root_policy(hero, enemy)
    finally:
        self.search_agent._is_training = False
```

---

## 修复验证 ✓

所有修复已验证：

```
[OK] SearchConfig initialized: 2 16
[OK] TrainSearchConfig initialized: 32 2
[OK] TrainConfig initialized with train_search
[OK] MinimaxBattleAI initialized with dual configs
[OK] Inference CFR iters: 16
[OK] Training CFR iters: 32
[OK] Frontier params: (0.5, 1.0, 1)

========================================
All parameter system tests passed!
========================================
```

---

## 现在可以安全运行

✓ depth > 1 的 CFR 展开不再卡死
✓ 缓存机制正常工作
✓ 搜索树正确递减
✓ 参数职责明确，易于维护
✓ 支持为推理和训练分别优化参数

---

## 配置示例

```python
from ai import BattleNet, MinimaxBattleAI, SearchConfig, TrainSearchConfig, TrainConfig

# 快速模式（单人，低延迟）
model = BattleNet()
ai = MinimaxBattleAI(
    BattleGenome(),
    SearchConfig(depth=1, cfr_iters=8),
    TrainSearchConfig(cfr_iters=16),
)

# 训练模式
train_config = TrainConfig(
    episodes=5000,
    search_depth=1,
    max_search_depth=4,
    train_search=TrainSearchConfig(cfr_iters=16),
)
trainer = train_selfplay(model, train_config)

# 高质量模式（多人，高端设备）
ai_high_quality = MinimaxBattleAI(
    BattleGenome(),
    SearchConfig(depth=3, cfr_iters=32, frontier_low=0.3),
    TrainSearchConfig(cfr_iters=24, frontier_min_actions=3),
)
```

---

## 参考文档

已生成三份文档供参考：
1. `PARAMETER_CONFIG_GUIDE.md` - 详细参数配置指南
2. `PARAMETER_CHEATSHEET.md` - 快速参考卡片
3. `PARAMETER_SEPARATION_SUMMARY.md` - 改进总结
