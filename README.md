写在最前面
这次研究出发点其实是高中个人发明的课间放松小游戏，尝试训练出与人类抗衡的ai
研究的时间其实并不长也就是一周的时间，游戏的复杂度在冰冷的数字和cpu面前其实不算什么，归根结底就是博弈论的纳什均衡问题，只要设定好死亡后的参数，回溯模拟每一个回合，就能达到纳什均衡。我们常说deadline是第一生产力，也许兴趣interest才是第一内驱力，很少有这么为一件事，几天不出宿舍门的决心，最后也是与自己共勉。
# 掌心战争 AI — HandGame
一个基于博弈论的卡牌 AI 项目，为「掌心战争」双人对战游戏实现了两种 AI 策略：**Monte Carlo CFR 自博弈训练**与**精确 Nash 均衡求解**。

---
## 游戏规则

掌心战争是一个**双人同时行动**的零和回合制游戏。

| 初始状态 | 双方 HP=4，能量=2 |
|---|---|
| 每回合 | 双方同时选择技能，立即结算 |
| 胜利条件 | 将对方 HP 降至 0；或 30 回合后血量更高的一方获胜 |

### 技能表

| 编号 | 名称 | 能量消耗 | 说明 |
|------|------|----------|------|
| 0 | 单枪 | 1 | 基础攻击 |
| 1 | 双枪 | 2 | 中等攻击 |
| 2 | 三枪 | 2 | 高伤害攻击 |
| 3 | 大招 | 3 | 最强攻击，穿透部分防御 |
| 4 | 小防 | 0 | 基础防御，触发清零回血 |
| 5 | 大防 | 1 | 强防御，触发清零回血 |
| 6 | 能量 | 0 | 获得 +1 能量 |
| 7 | 反弹 | 1 | 弹回单枪/双枪伤害 |
| 8 | 清零 | 2 | 清空双方能量（对部分技能有效），对防御方回血 |

### 空城被动

当一方**回合开始前能量为 0** 且遭受攻击时，会触发空城矩阵，造成额外反伤。这是游戏中的核心机制之一。

---

## AI 方法

### 1. Monte Carlo CFR（`cfr_selfplay_train.py`）

基于**单步反事实遗憾最小化**的自博弈算法：

- 支持多进程并行训练（默认使用 `cpu_count - 2` 个进程）
- 内置启发式评估函数（`HeuristicParams`），权重在训练中自动学习
- LOOKAHEAD_DEPTH=3：每次模拟额外向前推演 3 回合
- 训练结果保存至 `cfr_output/cfr_strategy.json`

```bash
python cfr_selfplay_train.py            # 单进程训练
python cfr_selfplay_train.py parallel   # 多进程并行训练（推荐）
```

### 2. 有限视野 Nash 均衡（`nash_finite.py`）⭐ 推荐

**精确求解**掌心战争的 Nash 均衡策略，正确建模 30 回合上限：

- **反向归纳**（Backward Induction）：从第 30 回合结果倒推至第 1 回合，单次扫描无需迭代
- 状态空间：784 个非终局状态 × 30 回合 = 23,520 次 LP 求解
- 每回合策略独立，越接近时间结束时策略越倾向于利用血量优势
- 对称化处理消除浮点误差，可利用度约 **50.4%**（接近理论值 50%）
- 策略保存至 `cfr_output/nash_finite.npz`（约 1.2 MB）

```bash
python nash_finite.py          # 求解并保存（约 80 秒）
python nash_finite.py analyze  # 加载并分析已有策略
```

### 3. 精确 Nash 均衡（`cfr_exact.py`）

基于折现无限随机博弈（gamma=0.99）的值迭代 Nash 求解，已被有限视野版本替代。

```bash
python cfr_exact.py
```

---

## 快速开始

### 环境要求

```bash
pip install numpy scipy
# MC-CFR 训练额外需要：
pip install torch  # https://pytorch.org/
```

### 人机对战

```bash
# 使用预训练的有限视野 Nash AI（推荐，策略文件已包含在仓库中）
python play_nash.py

# 使用 MC-CFR AI（需要先训练或已有 cfr_output/cfr_strategy.json）
python cfr_play.py
```

### 重新训练

```bash
# 1. 训练 MC-CFR（需要 PyTorch）
python cfr_selfplay_train.py parallel

# 2. 重新计算有限视野 Nash（约 80 秒，需要 scipy）
python nash_finite.py

# 3. 导出策略到可读文本
python export_strategy.py both
```

### 打包成 exe（分发给朋友）

```bash
build_nash.bat   # 生成 dist/handgame_nash.exe + dist/cfr_output/nash_finite.npz
build_cfr.bat    # 生成带 MC-CFR AI 的 exe
```

---

## 项目结构

```
handgame/
├── 核心算法
│   ├── nash_finite.py          # 有限视野 Nash 求解器（推荐）
│   ├── cfr_exact.py            # 精确 Nash 求解器（无限视野）
│   └── cfr_selfplay_train.py   # MC-CFR 训练 + 游戏工具函数
│
├── 游戏与对战
│   ├── dqn_train.py            # 游戏规则常量（HP矩阵、技能表等）
│   ├── play_nash.py            # Nash AI 对战（打包专用，无外部依赖）
│   ├── cfr_play.py             # 双模式对战（Nash / MC-CFR）
│   └── handgame.py             # 原始 DQN 模式对战
│
├── 工具
│   ├── export_strategy.py      # 导出策略到可读 txt
│   ├── build_nash.bat          # 打包 Nash exe
│   └── build_cfr.bat           # 打包 CFR exe
│
├── cfr_output/
│   ├── nash_finite.npz         # 预计算的有限视野 Nash 策略
│   └── cfr_strategy.json       # 训练好的 MC-CFR 策略
│
└── handgame_multiplayers/      # 多人版本（独立子项目）
```

---

## 技术细节

### 为什么选择有限视野 Nash？

传统折现无限博弈（gamma<1）只是近似建模了回合上限规则。有限视野反向归纳精确地将「第 30 回合血量多的一方获胜」纳入求解框架，使策略在接近时间限制时自然地产生变化——例如：

- **早期（第1回合）**：「平局血量」状态 +0.29，以普通博弈为主
- **后期（第28回合）**：同样状态 +0.70，策略大幅偏向保护血量优势

### 可利用度（Exploitability）

```
[可利用度]  BR_P1=48.8%  BR_P2=52.0%  avg_exploit=50.4%
```

Nash 均衡的理论可利用度为 50%（零和对称游戏中最佳响应者恰好能赢 50%）。本项目的求解结果 avg=50.4% 表明策略质量极高，误差来源于采样统计噪声（n_eval=500）。

---

## 子项目：多人版本

`handgame_multiplayers/` 为独立的多人对战子项目，有独立的 README 和依赖管理。


