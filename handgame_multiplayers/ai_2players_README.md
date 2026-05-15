# ai.py — 双人对战 AI 设计文档

> 本文档对应仓库根目录 `ai.py`（双人对战版）。它是一套基于 **CFR + 价值/策略网络** 的对战 AI，
> 配套自博弈训练流程、动态深度调度与显存安全的批量树展开。

---

## 1. 模块总览

`ai.py` 是一个自包含模块，对外的核心导出按职责分成 6 组：

| 组别 | 主要符号 | 用途 |
|------|---------|------|
| 配置 | `SearchConfig`, `TrainSearchConfig`, `BattleGenome`, `PlayerAIConfig`, `TrainConfig` | dataclass，描述网络结构 / 推理搜索 / 训练搜索 / 训练流程 |
| 决策结果 | `ActionPlan` | `choose_action` 的返回类型 |
| 神经网络 | `BattleNet` | PyTorch 价值/策略双头网络 |
| 推理 AI | `MinimaxBattleAI` | CFR + 网络的对战决策器 |
| 训练 | `SelfPlayTrainer`, `LiveLossPlotter`, `train_selfplay`, `save_training_run`, `evaluate_battle` | 自博弈训练、实时绘图、保存与评估 |
| 工具函数 | `encode_state`, `simulate_round`, `_apply_actions`, `_regret_matching`, `_nash_value`, `_hero_signature`, `_is_terminal`, `_terminal_value`, `_policy_to_vector`, `_sample_from_policy`, `_apply_dirichlet_noise` | 状态编码、回合模拟、CFR 数学原语、采样、噪声等 |

外部依赖：`torch`、`numpy`，以及 `heros.basic.BasicHero`（在训练 / 评估时延迟导入）。

---

## 2. 状态编码与回合模拟

### 2.1 `encode_state(hero, enemy) -> Tensor (1, 16)`

把双方英雄各 8 维属性拼成 16 维特征并按 ceiling 归一化，是网络的唯一输入。

每个英雄打包 8 维，顺序与含义：

| index | 字段 | 含义 |
|-------|------|------|
| 0 | `hp` | 当前血量 |
| 1 | `hp_ceiling` | 血量上限 |
| 2 | `mp` | 当前蓝量 |
| 3 | `mp_ceiling` | 蓝量上限 |
| 4 | `shield` | 护盾 |
| 5 | `damage_resistance` | 减伤 |
| 6 | `revival_armor` | 复活甲 |
| 7 | `alive` | `1.0 if hp>0 else 0.0` |

归一化分母用各属性的 ceiling / 当前最大值，避免数值发散。返回形状 `(1, 16)`，方便和 batched 评估走同一条路径（`squeeze(0)` 后再 `stack`）。

### 2.2 `simulate_round(hero, enemy, hero_action, enemy_action)`

完整模拟一回合：

1. `deepcopy` 双方，避免污染输入状态；
2. 每个角色 `reset_stacks()` 清空当回合的临时叠加；
3. `_apply_actions` 按 **类型优先级**（`self-attack` → `group-attack` → `anti-group-attack` → `attack` → `defense` → `control` → 其它）执行双方动作；动作执行前会跳过血量 ≤0 且无复活甲的角色；
4. 依次 `update_status` → `apply_stamp` → `update_status`，让护盾、印记、buff/debuff 全部结算到位；
5. 返回新的 `(hero, enemy)`。

`_apply_actions` 的优先级表是写死的，由 `hero.action_list[i][4]`（`action_type` 字段）决定每个动作落在哪个时序桶。

### 2.3 终局判定

- `_is_terminal(hero, enemy)`：任一方 `hp <= 0` 即终局；
- `_terminal_value(hero, enemy)`：返回我方视角的胜负值，`{+1, 0, -1}`，双败按平局算。

---

## 3. 神经网络 `BattleNet`

```
input(16) → Linear(128) → ReLU → Linear(64) → ReLU
            ├─ value_head: Linear(64,1) → Sigmoid     # 价值预测，∈ (0,1)
            └─ policy_head: Linear(64, policy_size)   # 策略 logits（未归一）
```

- `_init_small_weights`：所有参数均匀初始化到 `[-1e-3, 1e-3]`，让早期网络近似输出常数，避免训练初期的方差爆炸。
- 价值头输出 ∈ (0,1)，外部使用时通过 `value*2 - 1` 映射到 `[-1,1]`，与终局值同尺度。
- 策略头输出 logits，训练用 `log_softmax + 交叉熵` 蒸馏 CFR 的策略分布。

### `BattleGenome` 字段

| 字段 | 默认 | 说明 |
|------|------|------|
| `input_size` | 16 | 必须等于 `encode_state` 维度 |
| `hidden_1` | 128 | 第一隐藏层宽度 |
| `hidden_2` | 64 | 第二隐藏层宽度 |
| `value_scale` | 1.0 | 占位字段，目前未在前向中使用 |
| `policy_size` | 12 | 策略输出维度，必须 ≥ 任意英雄的 `action_list` 长度 |

---

## 4. 推理 AI `MinimaxBattleAI`

### 4.1 概念地图

`MinimaxBattleAI` 解决的是一个 **同时行动的零和博弈**：双方每回合各选 1 个动作，结果由 `simulate_round` 决定。直接 minimax 不适用（同时行动无明确"先手"），所以采用 **CFR (Counterfactual Regret Minimization)** 去近似纳什均衡策略，对未达终局的状态用 `BattleNet.value_head` 做局面评估。

完整调用链：

```
choose_action(hero, enemy)
  └─ _cfr_root_policy(hero, enemy)
        ├─ _build_payoff_matrix_batched(...)        ← 一次 GPU 批量评估
        │     ├─ expand(...)          [递归构子树, 收集叶状态]
        │     ├─ _evaluate_network_batch(...)  [单次前向]
        │     └─ collapse(...)        [Nash 折叠子树]
        └─ for k in range(cfr_iters):
              regret matching → 更新 regrets → 累计 strategy
        return 平均策略
```

返回 `ActionPlan(action_id, action_name, target_index)`。`target_index` 当前固定为 1（敌方），仅当动作 `info[3]`（`need_target`）为真时才设置。

### 4.2 CFR 与后悔匹配

- `_regret_matching(regrets)`：把后悔值的正部归一化为概率，全部 ≤0 时退化为均匀分布。这是 CFR 论文中的标准 Hart-Mas-Colell 后悔匹配。
- 根节点 CFR：在 `_cfr_root_policy` 里跑 `cfr_iters` 次 (`_get_cfr_iters()`)；每次基于双方当前后悔值求新策略，然后更新双方后悔值，并把每次的策略累加到 `strat_sum_self`，最后归一化得到 **平均策略** —— 这是 CFR 收敛到纳什均衡的策略。
- 子树 Nash 值：`_nash_value(actions_a, actions_b, payoff, iters)` 在递归折叠子树时也跑 `iters` 次 CFR，但不返回策略，只返回博弈值（`expected`），用作上层的 payoff 元素。

### 4.3 批量树展开 `_build_payoff_matrix_batched`

**问题背景**：原始实现是 DFS + batch=1 的 GPU 调用，`depth=2` 单次 `choose_action` 会触发 ~3.3 万次单样本前向，绝大部分时间都耗在 GPU kernel launch 与 PCIe 单元素拷贝上。

**新做法**：把整次 `choose_action` 的所有叶节点收集到一个张量里，一次前向算完，再纯 CPU 折叠。具体三阶段：

1. **第 1 阶段（CPU 展开）** —— `expand(h, e, d)` 递归：
   - 终局节点直接打 `("terminal", value)` 标签；
   - `d <= 1`、或 `len(features_buffer) >= max_batch_leaves`、或一方无合法动作时，把 `(h, e)` 塞进 `features_buffer`，打 `("leaf", idx)` 标签；
   - 否则枚举每对 `(a, b)`，`simulate_round` 后递归展开。
   - 子节点的合法动作集会先经过 `_select_frontier_actions` 做反悔值剪枝（见 4.4），控制分支宽度。
2. **第 2 阶段（一次 GPU）** —— `_evaluate_network_batch(features_buffer)` 把所有叶状态 `stack` 成 `(N, 16)` 一次性送进网络。
3. **第 3 阶段（CPU 折叠）** —— `collapse(node)` 自底向上：终局/叶子直接返回值；内部节点收集子矩阵后调 `_nash_value` 求该子博弈的均衡值。最终得到 `payoff_matrix[(a, b)]`，喂给根 CFR。

> **关键性质**：根 CFR 的 `cfr_iters` 次迭代复用同一份 `payoff_matrix`，因为叶子值只与状态有关，与策略权重无关 —— 无需在 CFR 循环内重复触发 GPU。

### 4.4 反悔值剪枝（`_select_frontier_actions` + `_apply_regret_based_pruning`）

**作用**：限制 `expand()` 在每个内部节点展开的 `(a, b)` 对数，控制叶子总量，防止指数爆炸。**纯 CPU**，不触发 GPU 调用。

**判定**：对于状态 `state_key` 下的动作 `a`，如果它最近 `regret_window` 次 CFR 迭代里反悔值 **全部** 低于 `regret_threshold`（默认 -0.5），则认为"怎么选都亏"，临时剪掉。

**复活机制**：剪枝后保留动作数低于 `frontier_min_actions` 时，从被剪枝集合中按 **历史累积反悔值** 倒序复活，直到补满下限。

**博弈安全性**：剪枝是基于 **观察到的真实后悔值**，不是基于网络估值预言。即使网络早期不准，被错杀的动作只要后续重新被尝试到（CFR 的探索性 + 复活机制），其反悔值会自然回升从而退出剪枝集合。这与"基于静态网络分前 X% 截断"那种博弈不安全的方案有本质差别 —— 后者一旦网络评估错就永远偏，没有自我纠错。

**历史维护**：`_update_regret_history` 在每次根 CFR 迭代结束时调用，把当前根节点的 regrets 追加到 `_regret_history[(state_key, action_id)]`，并裁剪到最长 50 项。

> **当前局限**：反悔值历史只在 **根节点** 上累积（CFR 只在根跑完整循环；子树用 `_nash_value` 但不写历史）。所以深层节点首次访问时通常没有历史 → `_select_frontier_actions` 退化为"保留全部动作"。这种情况下显存兜底完全靠 `max_batch_leaves`（见 4.6）。

### 4.5 网络评估接口与主动性增强 Score

| 方法 | 用途 | 何时触发 GPU |
|------|------|-------------|
| `_evaluate_network(hero, enemy)` | 单点查询，便于调试或单次估值 | 每次 1 次（batch=1，γ≠0 时 batch=2） |
| `_evaluate_network_batch(states)` | 批量查询，主路径 | 每次 1 次（batch=N，γ≠0 时 batch=2N） |

两个接口都按下式给叶节点打分，让 CFR 在搜索时把"对手处境好"也当成对自己的惩罚，从而表现得更主动：

$$
Score = V(Self_{after}) - \gamma \cdot V(Opp_{after})
$$

- $V(Self_{after})$：把 `(hero, enemy)` 顺序喂给 `encode_state` 再过价值头；
- $V(Opp_{after})$：把 `(enemy, hero)` **交换顺序** 喂同一网络，等价于从对手视角看同一局面；
- $\gamma$ 来自 `SearchConfig.value_gamma` / `TrainSearchConfig.value_gamma`，默认 1.0。

**实现细节**：γ≠0 时，每个 `(h, e)` 同时编码 `(h, e)` 和 `(e, h)`，拼接成 `(2N, 16)` 一次性前向（仍然只触发 1 次 GPU），然后按公式合成；γ=0 时短路成原始单视角，省掉一半计算。

**γ 的几种典型取值**：

| γ | 行为 |
|---|------|
| 0 | 等价于只看 V_self，最保守 |
| 1（默认） | 对一个 **完美反对称** 的网络等价于把 V_self 翻倍；对训练不足的网络则补足不对称性 |
| >1 | 显式偏向"打掉对手"而非"保全自己"，更激进 |
| <0 | （不推荐）反向，会让 AI 主动让对手变好 |

**与训练 loss 的关系**：本式仅作用于 **搜索时的叶节点打分**，不影响 `_update_model` 里的 MSE/CE 监督信号 —— 网络仍按真实终局胜负回归（target ∈ {-1, 0, 1}）。改 γ 是改 CFR 的 payoff 信号，不是改训练目标。

**与反悔值剪枝的交互**：γ>0 会放大叶节点分值的绝对范围，导致 CFR 累积的反悔值幅度也变大。如果你显著调大 γ（比如 γ=2），可以同步把 `regret_threshold` 也按比例调大（更深的负值，例如 -1.0），避免误剪。

`_evaluate_network_batch` 把张量一次性 `to(device, non_blocking=True)`，然后单次 `model.forward`，最后 `cpu().tolist()` 取回；γ≠0 时多一段 list comprehension 合成 Score。

### 4.6 显存兜底 `max_batch_leaves`

最坏情况下叶子数随 `depth` 指数增长（见下表，`|A|=|B|=12`）：

| depth | 不剪枝 | 剪到每边 6 | 剪到每边 4 |
|-------|--------|------------|------------|
| 2 | 20 736 | 1 296 | 256 |
| 3 | 2.99 M | 46 656 | 4 096 |
| 4 | 430 M | 1.68 M | 65 536 |
| 5 | 62 B | 60 M | 1.05 M |
| 6 | 8.9 T | 2.18 B | 16.8 M |

仅靠反悔值剪枝在深层不一定够（深层无历史），所以 `expand()` 在 `len(features_buffer) >= max_batch_leaves` 时把后续节点强制当作叶子（直接走网络估值）。这样：

- 显存使用 ≤ `O(max_batch_leaves × hidden_1 × 4 B)`（默认 200k × 128 × 4 ≈ 100 MB 量级激活，加上输入）；
- 行为上等价于"前面以全展开为主，后面以网络评估代替深层 CFR"。

如果遇到 OOM，把 `max_batch_leaves` 调小；如果显存富余且想要更深的精确展开，调大。

### 4.7 状态签名 `_state_hash` / `_hero_signature`

`_hero_signature` 收集 `(类名, hp, hp_ceiling, mp, mp_ceiling, shield, damage_resistance, revival_armor)`；`_state_hash` 加上 `depth` 后作为反悔值历史与剪枝集合的字典键。注意当前签名不区分训练/推理模式，两种模式共享 `_regret_history` 与 `_pruned_actions` —— 实际使用中只跑一种模式即可，不要混。

---

## 5. 配置参数完整清单

### 5.1 `SearchConfig`（推理时搜索）

| 参数 | 默认 | 含义 |
|------|------|------|
| `depth` | 2 | 搜索展开层数（根算第 0 层，叶算 1） |
| `cfr_iters` | 16 | 根 CFR 与子树 Nash 折叠的迭代次数 |
| `frontier_min_actions` | 1 | 反悔值剪枝后最少保留的动作数 |
| `regret_pruning_enabled` | True | 是否启用反悔值剪枝 |
| `regret_threshold` | -0.5 | 反悔值阈值，连续低于此值则剪枝 |
| `regret_window` | 5 | 连续观察窗口，N 次都低于阈值才剪 |
| `max_batch_leaves` | 200_000 | 批量展开叶节点预算（OOM 兜底） |
| `value_gamma` | 1.0 | 叶节点 Score = V(self) − γ·V(opp) 的系数，>0 增强主动性 |

### 5.2 `TrainSearchConfig`（训练时搜索，与推理对称）

| 参数 | 默认 | 含义 |
|------|------|------|
| `cfr_iters` | 16 | 训练时 CFR 迭代次数（通常更小以提速） |
| `frontier_min_actions` | 2 | 训练时最少保留动作数（建议 ≥2 以保留探索） |
| `regret_pruning_enabled` | True | 训练时是否启用反悔值剪枝 |
| `regret_threshold` | -0.5 | 训练时反悔值阈值 |
| `regret_window` | 5 | 训练时观察窗口 |
| `max_batch_leaves` | 200_000 | 训练时叶节点预算 |
| `value_gamma` | 1.0 | 训练时的对手惩罚系数 |

> **注意**：训练里没有独立的 `depth` 字段，是因为 `SelfPlayTrainer._maybe_increase_depth` 会动态把 `current_depth` 写进 `search_agent.search.depth`。

### 5.3 训练 / 推理参数独立性

`MinimaxBattleAI` 通过 `_is_training` 标志在两套 dataclass 之间切换，所有 helper 严格分流：

| 参数 | helper | 推理来源 | 训练来源 |
|------|--------|---------|---------|
| `cfr_iters` | `_get_cfr_iters()` | `search.cfr_iters` | `train_search.cfr_iters` |
| `frontier_min_actions` | `_get_min_actions()` | `search.frontier_min_actions` | `train_search.frontier_min_actions` |
| `regret_pruning_enabled` | `_get_regret_params()[0]` | `search.regret_pruning_enabled` | `train_search.regret_pruning_enabled` |
| `regret_threshold` | `_get_regret_params()[1]` | `search.regret_threshold` | `train_search.regret_threshold` |
| `regret_window` | `_get_regret_params()[2]` | `search.regret_window` | `train_search.regret_window` |
| `max_batch_leaves` | `_get_max_batch_leaves()` | `search.max_batch_leaves` | `train_search.max_batch_leaves` |
| `value_gamma` | `_get_value_gamma()` | `search.value_gamma` | `train_search.value_gamma` |
| `depth` | （直接读） | `search.depth` | （由训练动态写回 `search.depth`） |

模式切换发生在 `SelfPlayTrainer._search_policy`：用 `try/finally` 保证不论是否抛异常，最终都把 `_is_training` 复位为 `False`。

### 5.4 `BattleGenome`、`PlayerAIConfig`、`ActionPlan`

- `BattleGenome` 见 §3 表格。
- `PlayerAIConfig`：`(enabled, genome, search, label)` 四元组，多人对战外层用来给每个玩家挂一份独立配置。
- `ActionPlan(action_id, action_name, target_index)`：决策返回。

### 5.5 `TrainConfig`（训练流程）

| 参数 | 默认 | 含义 |
|------|------|------|
| `episodes` | 1000 | 总训练回合数（自博弈次数） |
| `max_rounds` | 30 | 单局最大回合数（防止平局死循环） |
| `batch_size` | 128 | 每次梯度更新的样本数 |
| `replay_size` | 2048 | 经验回放池容量（FIFO 截断） |
| `learning_rate` | 1e-3 | Adam 学习率 |
| `weight_decay` | 1e-4 | Adam 权重衰减 |
| `cfr_iters` | 16 | **已弃用**，仅作为 `SearchConfig` 初始化时的 fallback；实际生效的是 `train_search.cfr_iters` |
| `search_depth` | 1 | 初始搜索深度 |
| `max_search_depth` | 6 | 搜索深度上限 |
| `depth_increase_patience` | 30 | 多少次更新 loss 不再下降才提升一层深度 |
| `depth_threshold_percentile` | 50.0 | 用前 N 个样本均值的 X% 作为深度通过阈值（当前未硬性使用，只记录） |
| `depth_threshold_warmup` | 20 | 用前多少次 loss 估算阈值 |
| `max_seconds` | 0.0 | 训练时间上限（秒），0 表示不限 |
| `lr_reduce_patience` | 20 | 多少次 loss 不下降后 LR 衰减；同时也是 early stop 在最低 LR 上的容忍次数 |
| `lr_reduce_factor` | 0.5 | LR 衰减倍率 |
| `min_lr` | 1e-6 | LR 下限 |
| `dirichlet_alpha` | 0.3 | 自博弈策略噪声 α |
| `dirichlet_epsilon` | 0.1 | 噪声混合比例 |
| `plot_interval` | 1 | 每多少 episode 刷新一次 loss 曲线 |
| `device` | "auto" | "cpu" / "cuda" / "auto"；CLI 会自动展开 auto |
| `train_search` | `TrainSearchConfig()` | 训练时的搜索/剪枝参数 |

---

## 6. 自博弈训练 `SelfPlayTrainer`

### 6.1 训练主循环 `train()`

每个 episode 流程：

1. 检查 `max_seconds` 时间预算；
2. `_play_episode()` 自博弈一局，得到带胜负标签的样本；
3. `_push_episode()` 入池（FIFO 截断到 `replay_size`）；
4. 经验池足够后，`_update_model()` 采样 `batch_size` 训练一次；
5. `scheduler.step(loss)` 触发 `ReduceLROnPlateau` 的 LR 衰减；
6. `_maybe_increase_depth(loss)` 决定是否提升 `current_depth`；
7. `_should_early_stop(loss)` 判定提前终止；
8. 每隔 `plot_interval` 在控制台打印并刷新实时 loss 曲线。

### 6.2 自博弈 `_play_episode`

每回合：

1. 调 `_search_policy(hero, enemy)`，得到 CFR 平均策略 π；
2. 用 `_apply_dirichlet_noise(π, α, ε)` 注入探索噪声；
3. `_sample_from_policy(π)` 采样动作；
4. 把 `(features, π_vector, perspective)` 暂存（perspective ∈ {+1, -1}，对应 p1/p2 视角）；
5. `simulate_round(p1, p2, act1, act2)` 推进；
6. 终局后用 `_terminal_value` 给两个视角各自打胜负标签 `target = result × perspective`，加进 `labeled` 返回。

注意训练样本是 `(features_16, policy_12, target_scalar)`：网络要同时学策略蒸馏（让 logits 接近 CFR 策略）和价值回归（让 value head 接近终局结果）。

### 6.3 损失 `_update_model`

```
value_loss = MSE(target, 2 * sigmoid(value) - 1)
policy_loss = - mean(sum(π * log_softmax(logits)))
total = value_loss + policy_loss
```

权重 1:1，没有显式系数；调权可以通过修改 `policy_loss` 前的常数实现。

### 6.4 动态深度 `_maybe_increase_depth`

策略：维护 `depth_best_loss`；每次 loss 没改善就 `depth_stall_count += 1`；连续 `depth_increase_patience` 次都没改善就 `current_depth += 1`，并：

- 把新深度同步到 `search_agent.search.depth`；
- **清空** 经验池（避免旧深度的数据污染新深度的训练）；
- 重置深度统计与阈值。

`_record_depth_loss` 顺手记录每个深度的 loss 序列，并在采集到 `depth_threshold_warmup` 个样本后用前 N 个均值的百分比生成一个 `depth_thresholds[depth]`（目前只是记录，未参与硬决策）。

### 6.5 提前终止 `_should_early_stop`

只有当 LR 已经被推到 `min_lr`，并且接下来 `lr_reduce_patience` 次 loss 都没新低时，才停。换言之：必须先经历完整的 LR 衰减阶梯，才允许早停。

### 6.6 实时绘图 `LiveLossPlotter`

懒加载 `matplotlib`；如果导入失败就静默禁用，不影响训练。三条曲线：total / policy / value，自动 rescale。

---

## 7. 模型保存与评估

### 7.1 `save_training_run(...)`

把训练产物写到 `output_dir`：

- `<tag>_config.json`：序列化的 `TrainConfig`；
- `<tag>_loss.json`：`{"total":[...], "policy":[...], "value":[...], "lr":[...], "depth":[...]}`；
- `<tag>_loss.png`：matplotlib 静态导出；
- `<tag>_model.pt`：`{"model_state": state_dict()}` 形式的权重（`train_selfplay` 的 `--resume` 同时支持这种字典和裸 state_dict）。

`use_timestamp=True` 时文件名前缀会附加 `_YYYYMMDD_HHMMSS`，避免覆盖。

### 7.2 `evaluate_battle(model_a, model_b, episodes, max_rounds, search_depth, cfr_iters)`

让两套权重各自挂一个 `MinimaxBattleAI`，用相同 `SearchConfig` 互相对战 `episodes` 局，返回：

```
{"win_rate": ..., "loss_rate": ..., "draw_rate": ...}
```

胜负从 model_a 的视角算（`_terminal_value > 0` 计胜）。

---

## 8. CLI 入口（`__main__`）

`python ai.py [选项]` 直接训练 + 评估。可用参数：

| 选项 | 默认 | 含义 |
|------|------|------|
| `--resume PATH` | `None` | 从指定 ckpt 续训 |
| `--resume-latest` | False | 自动找 `ai/basic_vs_basic/` 下最新的 `basic_vs_basic_<YYYYMMDD>_model.pt` |
| `--device {cpu,cuda,auto}` | `auto` | 训练设备；`auto` = CUDA 可用就用 |
| `--minutes M` | 0 | 训练时间预算，分钟；写入 `TrainConfig.max_seconds` |
| `--eval-episodes N` | 50 | 训完后跑多少局评估 |
| `--eval-rounds N` | 30 | 评估每局最大回合 |

执行后会在 `ai/basic_vs_basic/` 下产出 4 类文件，并打印 `evaluate_battle` 的胜率。

---

## 9. 设计要点 & 性能特性

### 9.1 为什么 CFR 而不是 minimax / α-β

本游戏每回合双方 **同时出招**，没有"先手"概念，传统 minimax 需要展开成一个对手知情的 imperfect-info 树才能正确处理；CFR 是这类博弈的工业标准方案，能用渐进意义上收敛到纳什均衡的策略。

### 9.2 为什么把 payoff 矩阵从 CFR 循环里提出来

原实现每次根 CFR 迭代都重新走整棵树，但叶子值是 **状态的确定函数**，与 CFR 策略无关。预计算后：

- 同样 `cfr_iters=16` 时 GPU 调用从 ~16 倍降到 1 倍；
- 配合批量化，单次 `choose_action` 的 GPU 端开销从"上万次单样本"降到"一次大 batch"，端到端速度可以快一个数量级以上。

### 9.3 为什么必须保留反悔值剪枝

不剪枝时 `depth=4` 就会有 4 亿叶子（见 §4.6 表），任何显存都顶不住。反悔值剪枝是 **博弈安全** 的（基于真实后悔值，会自我纠错），且 **CPU-only**（不破坏批量化），是当前设计中唯一被允许的剪枝手段。

### 9.4 OOM 兜底的优先级

`expand()` 内的判断顺序：终局 → `d<=1` 或 **预算耗尽** → 没有合法动作 → 内部递归。一旦 `len(features_buffer) >= max_batch_leaves`，后续所有节点都直接走网络估值，等价于"前面深度大、后面退化为 1 步前瞻"。这是 **优雅降级**，不会抛异常。

### 9.5 已知小坑

- `TrainConfig.cfr_iters` 已弃用（保留只是为了不破坏旧 `SearchConfig` 的 fallback 初始化），实际训练 CFR 次数读 `train_search.cfr_iters`；不要在新代码里依赖前者。
- `BattleGenome.value_scale` 字段保留但未在前向中使用。
- `_regret_history` / `_pruned_actions` 不区分训练 / 推理模式，跨模式混用同一 AI 实例可能互相污染历史；正常用法是各自实例化。
- `_state_hash` 的 `depth` 作为键的一部分，意味着同样状态在不同深度的剪枝历史是分开的；这是有意为之（不同深度的 payoff 不同）。

---

## 10. 典型用法示例

### 10.1 训练

```python
from ai import BattleNet, BattleGenome, TrainConfig, TrainSearchConfig, train_selfplay

model = BattleNet(BattleGenome())
config = TrainConfig(
    episodes=2000,
    device="cuda",
    train_search=TrainSearchConfig(
        cfr_iters=8,
        frontier_min_actions=2,
        regret_threshold=-0.3,
        max_batch_leaves=100_000,
    ),
)
trainer = train_selfplay(model, config)
```

### 10.2 推理

```python
from ai import MinimaxBattleAI, BattleGenome, SearchConfig, BattleNet
import torch

genome = BattleGenome()
model = BattleNet(genome)
model.load_state_dict(torch.load("ai/basic_vs_basic/basic_vs_basic_YYYYMMDD_model.pt")["model_state"])

agent = MinimaxBattleAI(genome, SearchConfig(depth=3, cfr_iters=32))
agent.model = model           # 挂载训练好的权重
agent.model.to("cuda")

plan = agent.choose_action(my_hero, enemy_hero)
print(plan.action_id, plan.action_name, plan.target_index)
```

### 10.3 评估

```python
from ai import evaluate_battle
metrics = evaluate_battle(model_a, model_b, episodes=100, search_depth=2, cfr_iters=16)
print(metrics)  # {'win_rate': ..., 'loss_rate': ..., 'draw_rate': ...}
```

---

## 11. 与其他文档的关系

仓库里还有这几份相关 md，本文为概念/参数/算法的总览，更专题的细节请翻：

- `PARAMETER_CHEATSHEET.md` / `PARAMETER_CONFIG_GUIDE.md` / `PARAMETER_SEPARATION_SUMMARY.md`：各超参数的取值经验与训练 / 推理分离的细节（部分字段已被精简，以本文为准）。
- `REGRET_PRUNING.md`：反悔值剪枝的早期设计稿（实现已迭代，参数命名以本文与 `ai.py` 当前代码为准）。
- `FIX_SUMMARY.md`：早期缓存/递归 bug 修复记录（`_cache` 已随静态剪枝一起删除，可作为历史参考）。
- `ai_design_twoplayers.md`：双人对战 AI 的整体设计草案。
