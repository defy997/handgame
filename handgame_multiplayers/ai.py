from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import json
import os
import argparse
import time
from datetime import datetime
import math
import random
import torch
from torch import nn
import numpy as np

@dataclass
class SearchConfig:
    """推理时搜索配置（实际运行时使用）。

    Args:
        depth: 搜索展开层数。
        cfr_iters: CFR 迭代次数。
        frontier_min_actions: 最少保留动作数。
        regret_pruning_enabled: 是否启用反悔值剪枝（博弈安全方案）。
        regret_threshold: 反悔值阈值（负值，如 -0.5，表示怎么选都亏）。
        regret_window: 连续观察窗口大小（连续 N 次迭代反悔值低于阈值则剪枝）。
        max_batch_leaves: 批量树展开的叶节点预算（防止 GPU 显存 OOM）。
        value_gamma: 叶节点评估公式 Score = V(self) - γ·V(opp) 的系数。
            γ=0 等价于只看自己；γ>0 时把"对手处境好"作为额外惩罚，提升主动性。
    """
    depth: int = 3
    cfr_iters: int = 16
    frontier_min_actions: int = 1
    regret_pruning_enabled: bool = True
    regret_threshold: float = -0.5
    regret_window: int = 5
    max_batch_leaves: int = 200_000
    value_gamma: float = 2.0


@dataclass
class TrainSearchConfig:
    """训练时搜索配置（自博弈训练使用）。

    Args:
        cfr_iters: 训练时 CFR 迭代次数（通常比推理少）。
        frontier_min_actions: 训练时最少保留动作数。
        regret_pruning_enabled: 是否在训练时启用反悔值剪枝。
        regret_threshold: 训练时反悔值阈值。
        regret_window: 训练时连续观察窗口大小。
        max_batch_leaves: 训练时批量树展开的叶节点预算。
        value_gamma: 训练时叶节点评估的对手惩罚系数（见 SearchConfig.value_gamma）。
    """
    cfr_iters: int = 16
    frontier_min_actions: int = 2
    regret_pruning_enabled: bool = True
    regret_threshold: float = -0.5
    regret_window: int = 5
    max_batch_leaves: int = 200_000
    value_gamma: float = 1.0


@dataclass
class BattleGenome:
    """网络结构超参数。

    Args:
        input_size: 输入特征维度。
        hidden_1: 第一隐藏层宽度。
        hidden_2: 第二隐藏层宽度。
        value_scale: 价值头缩放因子。
        policy_size: 策略输出维度。
    """
    input_size: int = 16
    hidden_1: int = 128
    hidden_2: int = 64
    value_scale: float = 1.0
    policy_size: int = 12


@dataclass
class PlayerAIConfig:
    """单个玩家 AI 配置。"""
    enabled: bool = True
    genome: BattleGenome = field(default_factory=BattleGenome)
    search: SearchConfig = field(default_factory=SearchConfig)
    label: str = "default"


@dataclass
class ActionPlan:
    """动作决策结果。"""
    action_id: int
    action_name: str
    target_index: Optional[int]


class BattleNet(nn.Module):
    """自博弈价值/策略网络。"""
    def __init__(self, genome: BattleGenome) -> None:
        """初始化网络结构。

        Args:
            genome: 网络结构参数。
        """
        super().__init__()
        self.genome = genome
        self.trunk = nn.Sequential(
            nn.Linear(genome.input_size, genome.hidden_1),
            nn.ReLU(),
            nn.Linear(genome.hidden_1, genome.hidden_2),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(genome.hidden_2, 1),
            nn.Sigmoid(),
        )
        self.policy_head = nn.Linear(genome.hidden_2, genome.policy_size)
        self._init_small_weights()

    def _init_small_weights(self) -> None:
        """将所有参数初始化到 [-1e-3, 1e-3]。"""
        for param in self.parameters():
            nn.init.uniform_(param, -1e-3, 1e-3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向计算。

        Args:
            x: 特征张量，形状 (B, input_size)。

        Returns:
            value: 价值预测 (B, 1)。
            logits: 策略 logits (B, policy_size)。
        """
        trunk = self.trunk(x)
        value = self.value_head(trunk)
        logits = self.policy_head(trunk)
        return value, logits


class MinimaxBattleAI:
    """基于 CFR + 价值网络的对战 AI。"""

    def __init__(
        self,
        genome: BattleGenome,
        search: SearchConfig,
        train_search: Optional[TrainSearchConfig] = None,
    ) -> None:
        """初始化搜索 AI。

        Args:
            genome: 网络结构参数。
            search: 推理时搜索配置。
            train_search: 训练时搜索配置（可选，用于自博弈训练）。
        """
        if torch is None:
            raise RuntimeError(
                "PyTorch 未安装，无法初始化 AI。请安装 torch 后重试。"
            )
        self.genome = genome
        self.search = search  # 推理模式
        self.train_search = train_search or TrainSearchConfig()  # 训练模式
        self.model = BattleNet(genome)
        self._is_training = False  # 标志当前是否在训练模式

        # 反悔值剪枝相关：记录每个状态下各动作的反悔值历史
        # Key: (state_hash, action_id), Value: deque of regrets
        self._regret_history: Dict[Tuple, List[float]] = {}
        # Key: state_hash, Value: set of currently pruned action_ids
        self._pruned_actions: Dict[Tuple, set] = {}

    def _get_cfr_iters(self) -> int:
        """获取当前模式的 CFR 迭代次数。"""
        if self._is_training:
            return self.train_search.cfr_iters
        return self.search.cfr_iters

    def _get_min_actions(self) -> int:
        """获取当前模式的最少保留动作数。"""
        if self._is_training:
            return self.train_search.frontier_min_actions
        return self.search.frontier_min_actions

    def _get_regret_params(self) -> Tuple[bool, float, int]:
        """获取当前模式的反悔值剪枝参数。

        Returns:
            (enabled, threshold, window)
        """
        if self._is_training:
            return (
                self.train_search.regret_pruning_enabled,
                self.train_search.regret_threshold,
                self.train_search.regret_window,
            )
        return (
            self.search.regret_pruning_enabled,
            self.search.regret_threshold,
            self.search.regret_window,
        )

    def _get_max_batch_leaves(self) -> int:
        """获取当前模式的批量展开叶节点预算。"""
        if self._is_training:
            return self.train_search.max_batch_leaves
        return self.search.max_batch_leaves

    def _get_value_gamma(self) -> float:
        """获取当前模式的对手惩罚系数 γ（见叶节点评估公式）。"""
        if self._is_training:
            return self.train_search.value_gamma
        return self.search.value_gamma

    def choose_action(self, hero, enemy) -> ActionPlan:
        """给定双方状态，选择当前动作。

        Args:
            hero: 我方英雄实例。
            enemy: 敌方英雄实例。

        Returns:
            ActionPlan: 动作方案。
        """
        action_ids = self._available_actions(hero)
        if not action_ids:
            return ActionPlan(0, "攒能", None)

        policy = self._cfr_root_policy(hero, enemy)
        best_action = max(policy, key=policy.get)
        target = self._default_target(hero, enemy, best_action)
        name = self._action_name(hero, best_action)
        return ActionPlan(best_action, name, target)

    def _default_target(self, hero, enemy, action_id: int) -> Optional[int]:
        """为动作选择默认目标。"""
        need_target = False
        for info in hero.action_list:
            if info[0] == action_id:
                need_target = bool(info[3])
                break
        return 1 if need_target else None

    def _action_name(self, hero, action_id: int) -> str:
        """根据动作 ID 返回动作名称。"""
        for info in hero.action_list:
            if info[0] == action_id:
                return str(info[1])
        return "未知"

    def _available_actions(self, hero) -> List[int]:
        """获取英雄当前可行动作 ID 列表。"""
        available = hero.get_available_actions()
        return [action_id for action_id, _, _ in available]

    def _cfr_root_policy(self, hero, enemy) -> Dict[int, float]:
        """在根节点运行 CFR，得到策略分布。

        优化：先用 _build_payoff_matrix_batched 一次性 GPU 批量算出 |A|×|B| payoff，
        再进入 CFR 迭代（纯 CPU 查表），避免在 CFR 循环内重复触发 GPU 调用。
        """
        legal_self = self._available_actions(hero)
        legal_enemy = self._available_actions(enemy)
        if not legal_self:
            return {0: 1.0}
        if not legal_enemy:
            return {a: 1.0 / len(legal_self) for a in legal_self}

        payoff_matrix = self._build_payoff_matrix_batched(
            hero, enemy, legal_self, legal_enemy, self.search.depth
        )

        regrets_self = {a: 0.0 for a in legal_self}
        regrets_enemy = {a: 0.0 for a in legal_enemy}
        strat_sum_self = {a: 0.0 for a in legal_self}

        cfr_iters = self._get_cfr_iters()
        for _ in range(cfr_iters):
            strat_self = _regret_matching(regrets_self)
            strat_enemy = _regret_matching(regrets_enemy)
            for a, p in strat_self.items():
                strat_sum_self[a] += p

            expected = 0.0
            for a in legal_self:
                row_value = 0.0
                for b in legal_enemy:
                    row_value += strat_enemy[b] * payoff_matrix[(a, b)]
                expected += strat_self[a] * row_value

            for a in legal_self:
                alt = sum(strat_enemy[b] * payoff_matrix[(a, b)] for b in legal_enemy)
                regrets_self[a] += alt - expected
            for b in legal_enemy:
                alt = sum(strat_self[a] * (-payoff_matrix[(a, b)]) for a in legal_self)
                regrets_enemy[b] += alt - (-expected)

            state_key = self._state_hash(hero, enemy, 0)  # 根节点深度记为 0
            self._update_regret_history(hero, regrets_self, state_key)

        total = sum(strat_sum_self.values())
        if total <= 0:
            return {a: 1.0 / len(legal_self) for a in legal_self}
        return {a: strat_sum_self[a] / total for a in legal_self}

    def _build_payoff_matrix_batched(
        self,
        hero,
        enemy,
        legal_self: List[int],
        legal_enemy: List[int],
        depth: int,
    ) -> Dict[Tuple[int, int], float]:
        """一次 GPU 调用算出根部 |A|×|B| payoff 矩阵。

        三阶段：
          1. 纯 CPU BFS/DFS 展开整棵子博弈树，把所有需要走价值网络的叶状态
             放进 features_buffer；当 features_buffer 长度达到 max_batch_leaves
             时强制把当前节点当叶子处理（OOM 兜底）。
          2. 一次 model 前向传播评估所有叶子。
          3. 自底向上用 _nash_value（每层一次 CFR）折叠子树，得到根 payoff。
        """
        features_buffer: List[Tuple] = []
        max_leaves = self._get_max_batch_leaves()
        cfr_iters = self._get_cfr_iters()

        def expand(h, e, d: int):
            if _is_terminal(h, e):
                return ("terminal", _terminal_value(h, e))
            if d <= 1 or len(features_buffer) >= max_leaves:
                idx = len(features_buffer)
                features_buffer.append((h, e))
                return ("leaf", idx)
            la = self._available_actions(h)
            le = self._available_actions(e)
            if not la or not le:
                idx = len(features_buffer)
                features_buffer.append((h, e))
                return ("leaf", idx)
            la = self._select_frontier_actions(h, e, la, d)
            le = self._select_frontier_actions(e, h, le, d)
            kids: Dict[Tuple[int, int], Tuple] = {}
            for a in la:
                for b in le:
                    nh, ne = simulate_round(h, e, a, b)
                    kids[(a, b)] = expand(nh, ne, d - 1)
            return ("node", la, le, kids)

        root_kids: Dict[Tuple[int, int], Tuple] = {}
        for a in legal_self:
            for b in legal_enemy:
                nh, ne = simulate_round(hero, enemy, a, b)
                root_kids[(a, b)] = expand(nh, ne, depth - 1)

        leaf_values = (
            self._evaluate_network_batch(features_buffer) if features_buffer else []
        )

        def collapse(node) -> float:
            kind = node[0]
            if kind == "terminal":
                return node[1]
            if kind == "leaf":
                return leaf_values[node[1]]
            _, la, le, kids = node
            payoff_rows = [[collapse(kids[(a, b)]) for b in le] for a in la]
            return _nash_value(la, le, payoff_rows, cfr_iters)

        return {
            (a, b): collapse(root_kids[(a, b)])
            for a in legal_self
            for b in legal_enemy
        }

    def _evaluate_network(self, hero, enemy) -> float:
        """使用价值网络评估单个状态（保留作为单点查询接口）。

        当 value_gamma != 0 时返回主动性增强后的分：
            Score = V(self_after) - γ · V(opp_after)
        其中 V(self) 由 encode_state(hero, enemy) 喂网络得到，
        V(opp) 由 encode_state(enemy, hero) 喂同一网络得到（对称视角）。
        """
        gamma = self._get_value_gamma()
        device = next(self.model.parameters()).device

        if gamma == 0.0:
            features = encode_state(hero, enemy).to(device, non_blocking=True)
            with torch.no_grad():
                value, _ = self.model(features)
            return float(value.item() * 2.0 - 1.0)

        # 双视角：一次前向算两个视角，再合成 Score
        feat = torch.stack(
            [encode_state(hero, enemy).squeeze(0), encode_state(enemy, hero).squeeze(0)],
            dim=0,
        ).to(device, non_blocking=True)
        with torch.no_grad():
            values, _ = self.model(feat)
        v = (values.squeeze(1) * 2.0 - 1.0).cpu().tolist()
        return v[0] - gamma * v[1]

    def _evaluate_network_batch(self, states_list: List[Tuple]) -> List[float]:
        """批量评估多个状态（一次 GPU 调用）。

        当 value_gamma != 0 时把每个 (h, e) 同时编码为 (h, e) 与 (e, h)，
        一次前向得到 V_self 和 V_opp，再返回主动性增强分：
            Score[i] = V_self[i] - γ · V_opp[i]
        γ=0 时短路成原始单视角，省一半计算。

        Args:
            states_list: [(hero, enemy), ...] 状态对列表

        Returns:
            [score, ...] 评估值列表
        """
        if not states_list:
            return []

        gamma = self._get_value_gamma()
        n = len(states_list)
        device = next(self.model.parameters()).device

        self_rows = [encode_state(h, e).squeeze(0) for h, e in states_list]
        if gamma == 0.0:
            features_batch = torch.stack(self_rows, dim=0)  # (N, 16)
            features_batch = features_batch.to(device, non_blocking=True)
            with torch.no_grad():
                values, _ = self.model(features_batch)
            return (values.squeeze(1) * 2.0 - 1.0).cpu().tolist()

        opp_rows = [encode_state(e, h).squeeze(0) for h, e in states_list]
        features_batch = torch.stack(self_rows + opp_rows, dim=0)  # (2N, 16)
        features_batch = features_batch.to(device, non_blocking=True)

        with torch.no_grad():
            values, _ = self.model(features_batch)  # (2N, 1)

        v = (values.squeeze(1) * 2.0 - 1.0).cpu().tolist()
        v_self, v_opp = v[:n], v[n:]
        return [vs - gamma * vo for vs, vo in zip(v_self, v_opp)]

    def _state_hash(self, hero, enemy, depth: int) -> Tuple:
        """将状态与动作映射为缓存键。"""
        return (
            _hero_signature(hero),
            _hero_signature(enemy),
            depth,
        )

    def _select_frontier_actions(self, hero, enemy, actions: List[int], depth: int) -> List[int]:
        """选择前沿动作（仅使用反悔值剪枝，GPU-free）。

        反悔值剪枝是博弈安全的：当一个动作的最近 window 次反悔值都低于阈值时跳过，
        若历史不足或剪枝后动作过少则回退到保留全部动作，由 _build_payoff_matrix_batched
        中的 max_batch_leaves 预算兜底防止显存 OOM。
        """
        min_actions = self._get_min_actions()
        if len(actions) <= min_actions:
            return actions

        regret_pruning_enabled, _threshold, _window = self._get_regret_params()
        if not regret_pruning_enabled:
            return actions

        state_key = self._state_hash(hero, enemy, depth)
        selected = self._apply_regret_based_pruning(actions, state_key, min_actions)
        return selected if selected else actions

    def _apply_regret_based_pruning(
        self, actions: List[int], state_key: Tuple, min_actions: int
    ) -> List[int]:
        """基于反悔值历史的剪枝策略（博弈安全，纯 CPU）。

        如果一个动作在连续 window 次迭代中，反悔值都低于 threshold，则暂时跳过。
        当剪枝后动作数低于 min_actions 时，从被剪枝集合中按历史累积反悔值复活动作。
        返回 None 表示反悔值历史完全不足以进行剪枝。
        """
        _enabled, regret_threshold, regret_window = self._get_regret_params()

        # 初始化或获取此状态下被剪枝的动作集合
        if state_key not in self._pruned_actions:
            self._pruned_actions[state_key] = set()

        selected = []
        num_pruned = 0

        for action in actions:
            history_key = (state_key, action)
            should_prune = self._should_prune_action(history_key, regret_threshold, regret_window)
            if not should_prune:
                selected.append(action)
            else:
                self._pruned_actions[state_key].add(action)
                num_pruned += 1

        # 如果没有动作被剪枝，或者被剪枝后还有充足的动作，直接返回结果
        if num_pruned == 0 or len(selected) >= min_actions:
            return selected if selected else None

        # 如果剪枝后动作太少，重新激活被剪枝但历史最好的动作
        pruned_actions = list(self._pruned_actions[state_key])
        pruned_actions.sort(
            key=lambda a: self._get_action_regret_sum((state_key, a)),
            reverse=True
        )
        for action in pruned_actions:
            if len(selected) >= min_actions:
                break
            if action not in selected:
                selected.append(action)

        return selected if len(selected) > 0 else None

    def _should_prune_action(
        self, history_key: Tuple, threshold: float, window: int
    ) -> bool:
        """判断一个动作是否应该被剪枝。
        
        如果历史中连续window次的反悔值都低于threshold，则返回True。
        """
        if history_key not in self._regret_history:
            return False
        
        history = self._regret_history[history_key]
        if len(history) < window:
            return False
        
        # 检查最近window次的反悔值是否都低于阈值
        recent = history[-window:]
        return all(r < threshold for r in recent)

    def _get_action_regret_sum(self, history_key: Tuple) -> float:
        """获取一个动作的累积反悔值（用于决定重新激活哪个动作）。"""
        if history_key not in self._regret_history:
            return 0.0
        return sum(self._regret_history[history_key])

    def _update_regret_history(self, hero, regrets: Dict[int, float], state_key: Tuple = None) -> None:
        """更新反悔值历史记录。
        
        Args:
            hero: 英雄对象。
            regrets: 反悔值字典。
            state_key: 状态键（如果不提供，则使用hero的签名）。
        """
        if state_key is None:
            state_key = _hero_signature(hero)
        
        for action_id, regret in regrets.items():
            history_key = (state_key, action_id)
            if history_key not in self._regret_history:
                self._regret_history[history_key] = []
            
            self._regret_history[history_key].append(regret)
            
            # 限制历史长度以避免内存溢出
            max_history_len = 50
            if len(self._regret_history[history_key]) > max_history_len:
                self._regret_history[history_key].pop(0)


@dataclass
class TrainConfig:
    """训练配置。

    Args:
        episodes: 总训练回合数。
        max_rounds: 单局最大回合数。
        batch_size: 训练批大小。
        replay_size: 经验池容量。
        learning_rate: 学习率。
        weight_decay: 权重衰减。
        cfr_iters: 训练时 CFR 迭代次数（已弃用，使用 train_search.cfr_iters）。
        search_depth: 初始搜索深度。
        max_search_depth: 最大搜索深度。
        depth_increase_patience: 深度提升等待次数（与 lr_reduce_patience 类似）。
        depth_threshold_percentile: 深度提升阈值分位数。
        depth_threshold_warmup: 统计阈值的样本数。
        max_seconds: 训练时间上限（秒），0 表示不限制。
        lr_reduce_patience: 学习率衰减等待次数。
        lr_reduce_factor: 学习率衰减倍率。
        min_lr: 学习率下限。
        dirichlet_alpha: 迪利克雷噪声参数。
        dirichlet_epsilon: 噪声混合比例。
        plot_interval: loss 采样间隔。
        device: 训练设备。
        train_search: 训练时的搜索配置（CFR 和前沿筛选参数）。
        checkpoint_interval: 每 N 个 episode 周期保存一次中间 checkpoint；0 表示禁用。
        checkpoint_dir: 中间 checkpoint 的保存目录；空字符串表示禁用周期保存。
        checkpoint_tag: 中间 checkpoint 文件名前缀（最终生成 {tag}_ep{N}_model.pt / _config.json）。
    """
    episodes: int = 500
    max_rounds: int = 50
    batch_size: int = 128
    replay_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    cfr_iters: int = 16  # 已弃用，保留以兼容性
    search_depth: int = 1
    max_search_depth: int = 3
    depth_increase_patience: int = 50
    depth_threshold_percentile: float = 50.0
    depth_threshold_warmup: int = 20
    max_seconds: float = 0.0
    lr_reduce_patience: int = 20
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.1
    plot_interval: int = 1
    device: str = "auto"
    train_search: TrainSearchConfig = field(default_factory=TrainSearchConfig)
    checkpoint_interval: int = 50
    checkpoint_dir: str = ""
    checkpoint_tag: str = "checkpoint"


class LiveLossPlotter:
    """实时绘制训练 loss 曲线。"""
    def __init__(self) -> None:
        """初始化绘图器。"""
        self.enabled = False
        self.total_losses: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self._init_plot()

    def _init_plot(self) -> None:
        """尝试初始化 matplotlib 交互窗口。"""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            self.enabled = False
            self.plt = None
            return
        self.enabled = True
        self.plt = plt
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots()
        self.total_line, = self.ax.plot([], [], color="#2563eb", label="total")
        self.policy_line, = self.ax.plot([], [], color="#16a34a", label="policy")
        self.value_line, = self.ax.plot([], [], color="#dc2626", label="value")
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Update")
        self.ax.set_ylabel("Loss")
        self.ax.legend()
        self.fig.canvas.draw()

    def update(self, total_loss: float, policy_loss: float, value_loss: float) -> None:
        """追加 loss 并刷新曲线。

        Args:
            total_loss: 本次训练总 loss。
            policy_loss: 本次训练策略 loss。
            value_loss: 本次训练价值 loss。
        """
        self.total_losses.append(total_loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        if not self.enabled:
            return
        x = range(len(self.total_losses))
        self.total_line.set_data(x, self.total_losses)
        self.policy_line.set_data(x, self.policy_losses)
        self.value_line.set_data(x, self.value_losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class SelfPlayTrainer:
    """自博弈训练器。"""
    def __init__(self, model: BattleNet, config: TrainConfig) -> None:
        """初始化训练器。

        Args:
            model: 训练模型。
            config: 训练配置。
        """
        if torch is None:
            raise RuntimeError(
                "PyTorch 未安装，无法进行训练。请安装 torch 后重试。"
            )
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.lr_reduce_factor,
            patience=config.lr_reduce_patience,
            min_lr=config.min_lr,
        )
        self.buffer: List[Tuple[List[float], List[float], float]] = []
        self.plotter = LiveLossPlotter()
        self.loss_history: List[float] = []
        self.policy_loss_history: List[float] = []
        self.value_loss_history: List[float] = []
        self.lr_history: List[float] = []
        self.depth_history: List[int] = []
        self.current_depth = max(1, config.search_depth)
        self.loss_streak = 0
        self.depth_best_loss: Optional[float] = None
        self.depth_stall_count = 0
        self.last_loss: Optional[float] = None
        self.best_loss: Optional[float] = None
        self.lr_stall_count = 0
        self.loss_by_depth: Dict[int, List[float]] = {}
        self.depth_thresholds: Dict[int, float] = {}
        # 创建推理模式的搜索配置（深度会动态调整）
        self.search_agent = MinimaxBattleAI(
            self.model.genome,
            SearchConfig(depth=self.current_depth, cfr_iters=config.cfr_iters),
            train_search=config.train_search,
        )
        self.search_agent.model = self.model

    def train(self) -> List[float]:
        """执行训练主循环。

        Returns:
            Loss 列表。
        """
        start_time = time.time()
        for episode in range(1, self.config.episodes + 1):
            if self.config.max_seconds > 0:
                elapsed = time.time() - start_time
                if elapsed >= self.config.max_seconds:
                    print(f"[Train] time limit reached: {elapsed:.1f}s")
                    break
            episode_data = self._play_episode()
            self._push_episode(episode_data)

            if len(self.buffer) < self.config.batch_size:
                # 即使还没开始梯度更新，也按周期保存（便于早期排查）
                self._maybe_save_checkpoint(episode)
                continue

            loss, policy_loss, value_loss = self._update_model()
            self.loss_history.append(loss)
            self.policy_loss_history.append(policy_loss)
            self.value_loss_history.append(value_loss)
            self._record_depth_loss(loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_history.append(current_lr)
            self.depth_history.append(self.current_depth)
            self.scheduler.step(loss)
            self._maybe_increase_depth(loss)
            self._maybe_save_checkpoint(episode)
            if self._should_early_stop(loss):
                break
            if episode % self.config.plot_interval == 0:
                self.plotter.update(loss, policy_loss, value_loss)
                print(
                    f"[Episode {episode}] loss={loss:.4f} "
                    f"policy={policy_loss:.4f} value={value_loss:.4f} "
                    f"lr={current_lr:.2e} depth={self.current_depth}"
                )
        return self.loss_history

    def _maybe_save_checkpoint(self, episode: int) -> None:
        """按 checkpoint_interval 周期保存中间产物。

        当 checkpoint_dir 为空字符串或 checkpoint_interval <= 0 时为 no-op。
        每次保存：
          - {dir}/{tag}_ep{N:06d}_model.pt
              { 'model_state': ..., 'genome': asdict(genome), 'episode': N }
          - {dir}/{tag}_ep{N:06d}_config.json
              当前 TrainConfig 的快照（含动态 search_depth），供 cold_start_train 复活。
        """
        if self.config.checkpoint_interval <= 0:
            return
        if not self.config.checkpoint_dir:
            return
        if episode % self.config.checkpoint_interval != 0:
            return

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        base = f"{self.config.checkpoint_tag}_ep{episode:06d}"
        model_path = os.path.join(self.config.checkpoint_dir, f"{base}_model.pt")
        config_path = os.path.join(self.config.checkpoint_dir, f"{base}_config.json")

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "genome": asdict(self.model.genome),
                "episode": episode,
            },
            model_path,
        )

        # 把动态变化的 search_depth 写回快照，方便 cold-start 接着用
        snap = asdict(self.config)
        snap["search_depth"] = self.current_depth
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)

        print(f"[Checkpoint] ep{episode}: {model_path}")

    def _should_early_stop(self, loss: float) -> bool:
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.lr_stall_count = 0
            return False

        min_lr = self.config.min_lr
        current_lr = self.optimizer.param_groups[0]["lr"]
        if current_lr > min_lr + 1e-12:
            return False

        self.lr_stall_count += 1
        if self.lr_stall_count >= self.config.lr_reduce_patience:
            print("[Train] early stop: lr at min and loss not improving")
            return True
        return False

    def _maybe_increase_depth(self, loss: float) -> None:
        """根据 loss 变化动态提升搜索深度。"""
        if self.current_depth >= self.config.max_search_depth:
            return

        if self.depth_best_loss is None or loss < self.depth_best_loss:
            self.depth_best_loss = loss
            self.depth_stall_count = 0
            return

        self.depth_stall_count += 1
        if self.depth_stall_count >= self.config.depth_increase_patience:
            self.current_depth += 1
            self.depth_stall_count = 0
            self.depth_best_loss = None
            self.search_agent.search.depth = self.current_depth
            self.buffer.clear()
            self.loss_by_depth.clear()
            self.depth_thresholds.clear()
            print(f"[Depth] increased to {self.current_depth} (loss={loss:.4f})")

    def _record_depth_loss(self, loss: float) -> None:
        history = self.loss_by_depth.setdefault(self.current_depth, [])
        history.append(loss)
        if (
            self.current_depth not in self.depth_thresholds
            and len(history) >= self.config.depth_threshold_warmup
        ):
            threshold = float(
                np.mean(history[: self.config.depth_threshold_warmup])*self.config.depth_threshold_percentile/100.0
            )
            self.depth_thresholds[self.current_depth] = threshold

    def _play_episode(self) -> List[Tuple[List[float], List[float], float]]:
        """自博弈一局并返回带标签的数据。"""
        from heros.basic import BasicHero

        p1 = BasicHero("AI-1")
        p2 = BasicHero("AI-2")
        history: List[Tuple[List[float], List[float], int]] = []

        for _ in range(self.config.max_rounds):
            if _is_terminal(p1, p2):
                break
            # print(f"[Episode] round {_+1} - {p1.name}(hp={p1.hp}) vs {p2.name}(hp={p2.hp})")
            pi1 = self._search_policy(p1, p2)
            pi2 = self._search_policy(p2, p1)
            pi1 = _apply_dirichlet_noise(
                pi1,
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon,
            )
            pi2 = _apply_dirichlet_noise(
                pi2,
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon,
            )
            act1 = _sample_from_policy(pi1)
            act2 = _sample_from_policy(pi2)

            features_1 = encode_state(p1, p2).squeeze(0).tolist()
            features_2 = encode_state(p2, p1).squeeze(0).tolist()
            history.append((features_1, _policy_to_vector(pi1), 1))
            history.append((features_2, _policy_to_vector(pi2), -1))

            p1, p2 = simulate_round(p1, p2, act1, act2)

        result = _terminal_value(p1, p2)
        labeled = []
        for features, policy, perspective in history:
            target = result if perspective == 1 else -result
            labeled.append((features, policy, target))
        return labeled

    def _push_episode(self, episode_data: List[Tuple[List[float], List[float], float]]) -> None:
        """将一局数据写入经验池。"""
        self.buffer.extend(episode_data)
        if len(self.buffer) > self.config.replay_size:
            self.buffer = self.buffer[-self.config.replay_size :]

    def _update_model(self) -> Tuple[float, float, float]:
        """从经验池采样并更新模型。

        Returns:
            (total_loss, policy_loss, value_loss)。
        """
        batch = random.sample(self.buffer, self.config.batch_size)
        states, policies, targets = zip(*batch)

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        policy_tensor = torch.tensor(policies, dtype=torch.float32, device=self.device)
        target_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)

        value_pred, logits = self.model(states_tensor)
        value_pred = value_pred.squeeze(1) * 2.0 - 1.0

        value_loss = torch.mean((target_tensor - value_pred) ** 2)
        log_probs = torch.log_softmax(logits, dim=1)
        policy_loss = -torch.mean(torch.sum(policy_tensor * log_probs, dim=1))

        loss = value_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), float(policy_loss.item()), float(value_loss.item())

    def _search_policy(self, hero, enemy) -> Dict[int, float]:
        """基于当前深度返回 CFR 策略分布（训练模式）。"""
        # 进入训练模式：使用 train_search 参数
        self.search_agent._is_training = True
        try:
            return self.search_agent._cfr_root_policy(hero, enemy)
        finally:
            # 确保退出训练模式
            self.search_agent._is_training = False


def train_selfplay(
    model: BattleNet,
    config: Optional[TrainConfig] = None,
    resume_path: Optional[str] = None,
) -> SelfPlayTrainer:
    """训练入口：使用自博弈 + 实时 Loss 曲线。

    `resume_path` 只会加载权重，配置仍用调用方传入的 `config`。如果要把
    历史版本的 **配置 + 权重 + genome** 整体取回继续训练，请用
    `cold_start_train`。
    """
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    trainer = SelfPlayTrainer(model, config or TrainConfig())
    trainer.train()
    return trainer


def cold_start_train(
    checkpoint_base: str,
    overrides: Optional[Dict] = None,
) -> SelfPlayTrainer:
    """从指定版本前缀冷启动训练：恢复 model + genome + TrainConfig 全套状态。

    与 `train_selfplay(resume_path=...)` 的区别：
      - `train_selfplay` 只载权重，配置完全由调用方决定；
      - `cold_start_train` 把当时的 TrainConfig（含 train_search、checkpoint_*、
        动态调整后的 search_depth）一并取回，因此可以"接着训"而不是"换配置重训"。

    Args:
        checkpoint_base: 不带后缀的版本前缀路径，例如
            'ai/basic_vs_basic/basic_vs_basic_20260515' 或
            'ai/basic_vs_basic/basic_vs_basic_20260515_ep000100'。
            会读取 {base}_config.json 与 {base}_model.pt 两个文件。
        overrides: 顶层 TrainConfig 字段覆盖字典；
            如要覆盖嵌套的 train_search 子字段，传整段字典即可，例如
            {'episodes': 5000, 'device': 'cpu',
             'train_search': {'cfr_iters': 4, 'value_gamma': 2.0}}。
            未知字段会被自动过滤掉（容错），方便老版本 ckpt 升级。

    Returns:
        训练完成后的 SelfPlayTrainer（已经跑完 config.episodes）。
    """
    import dataclasses as _dc

    config_path = f"{checkpoint_base}_config.json"
    model_path = f"{checkpoint_base}_model.pt"

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    tc_fields = {f.name for f in _dc.fields(TrainConfig)}
    ts_fields = {f.name for f in _dc.fields(TrainSearchConfig)}
    bg_fields = {f.name for f in _dc.fields(BattleGenome)}

    # 嵌套 train_search 转 dataclass
    ts_dict = config_dict.get("train_search")
    if isinstance(ts_dict, dict):
        ts_filtered = {k: v for k, v in ts_dict.items() if k in ts_fields}
        config_dict["train_search"] = TrainSearchConfig(**ts_filtered)

    # 应用 overrides
    if overrides:
        for k, v in overrides.items():
            if k == "train_search" and isinstance(v, dict):
                base_ts = config_dict.get("train_search") or TrainSearchConfig()
                merged = asdict(base_ts) if not isinstance(base_ts, dict) else dict(base_ts)
                merged.update(v)
                merged = {k2: v2 for k2, v2 in merged.items() if k2 in ts_fields}
                config_dict["train_search"] = TrainSearchConfig(**merged)
            else:
                config_dict[k] = v

    tc_filtered = {k: v for k, v in config_dict.items() if k in tc_fields}
    config = TrainConfig(**tc_filtered)

    # 加载模型 + genome
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state", checkpoint)
        genome_dict = checkpoint.get("genome")
    else:
        state = checkpoint
        genome_dict = None

    if isinstance(genome_dict, dict):
        bg_filtered = {k: v for k, v in genome_dict.items() if k in bg_fields}
        genome = BattleGenome(**bg_filtered)
    else:
        genome = BattleGenome()  # 老 ckpt 没存 genome，回落到默认

    model = BattleNet(genome)
    model.load_state_dict(state)

    print(f"[ColdStart] from {checkpoint_base}")
    print(f"[ColdStart] genome: {asdict(genome)}")
    print(
        f"[ColdStart] episodes={config.episodes} device={config.device} "
        f"search_depth={config.search_depth} ckpt_interval={config.checkpoint_interval}"
    )

    trainer = SelfPlayTrainer(model, config)
    trainer.train()
    return trainer


def _safe_run_tag(text: str) -> str:
    """生成安全的文件名片段。"""
    safe = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in ("-", "_")):
            safe.append(ch)
        else:
            safe.append("_")
    tag = "".join(safe).strip("_")
    return tag or "run"


def save_training_run(
    output_dir: str,
    config: TrainConfig,
    losses: Sequence[float],
    battle_tag: str,
    model: Optional[BattleNet] = None,
    use_timestamp: bool = True,
    policy_losses: Optional[Sequence[float]] = None,
    value_losses: Optional[Sequence[float]] = None,
    lr_history: Optional[Sequence[float]] = None,
    depth_history: Optional[Sequence[int]] = None,
) -> Dict[str, str]:
    """保存训练产物。

    Args:
        output_dir: 输出目录。
        config: 训练配置。
        losses: loss 列表。
        battle_tag: 对战命名标签。
        model: 需要保存权重的模型。

    Returns:
        产物路径字典。
    """
    os.makedirs(output_dir, exist_ok=True)
    base = battle_tag
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{battle_tag}_{timestamp}"

    config_path = os.path.join(output_dir, f"{base}_config.json")
    loss_path = os.path.join(output_dir, f"{base}_loss.json")
    plot_path = os.path.join(output_dir, f"{base}_loss.png")
    model_path = os.path.join(output_dir, f"{base}_model.pt")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    loss_payload = {"total": list(losses)}
    if policy_losses is not None:
        loss_payload["policy"] = list(policy_losses)
    if value_losses is not None:
        loss_payload["value"] = list(value_losses)
    if lr_history is not None:
        loss_payload["lr"] = list(lr_history)
    if depth_history is not None:
        loss_payload["depth"] = list(depth_history)
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump(loss_payload, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        if model is not None:
            torch.save(
                {"model_state": model.state_dict(), "genome": asdict(model.genome)},
                model_path,
            )
            return {"config": config_path, "loss": loss_path, "model": model_path}
        return {"config": config_path, "loss": loss_path}

    if losses:
        plt.figure(figsize=(6, 4))
        x = range(1, len(losses) + 1)
        plt.plot(x, losses, color="#2563eb", label="total")
        if policy_losses is not None:
            plt.plot(x, policy_losses, color="#16a34a", label="policy")
        if value_losses is not None:
            plt.plot(x, value_losses, color="#dc2626", label="value")
        plt.title("Training Loss")
        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()
        if model is not None:
            torch.save(
                {"model_state": model.state_dict(), "genome": asdict(model.genome)},
                model_path,
            )
            return {
                "config": config_path,
                "loss": loss_path,
                "plot": plot_path,
                "model": model_path,
            }

    if model is not None:
        torch.save(
            {"model_state": model.state_dict(), "genome": asdict(model.genome)},
            model_path,
        )
        return {"config": config_path, "loss": loss_path, "model": model_path}

    return {"config": config_path, "loss": loss_path}


def evaluate_battle(
    model_a: BattleNet,
    model_b: BattleNet,
    episodes: int = 50,
    max_rounds: int = 30,
    search_depth: int = 1,
    cfr_iters: int = 16,
) -> Dict[str, float]:
    """评估两组参数的对战胜率。"""
    from heros.basic import BasicHero

    model_device = next(model_a.parameters()).device
    model_b.to(model_device)

    wins = 0
    losses = 0
    draws = 0

    search = SearchConfig(depth=search_depth, cfr_iters=cfr_iters)
    agent_a = MinimaxBattleAI(model_a.genome, search)
    agent_b = MinimaxBattleAI(model_b.genome, search)
    agent_a.model = model_a
    agent_b.model = model_b

    for _ in range(episodes):
        p1 = BasicHero("AI-A")
        p2 = BasicHero("AI-B")
        for _ in range(max_rounds):
            if _is_terminal(p1, p2):
                break
            act1 = agent_a.choose_action(p1, p2).action_id
            act2 = agent_b.choose_action(p2, p1).action_id
            p1, p2 = simulate_round(p1, p2, act1, act2)

        result = _terminal_value(p1, p2)
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1

    total = max(1, episodes)
    return {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
    }


def encode_state(hero, enemy) -> torch.Tensor:
    """将双方状态编码为 16 维向量。"""
    def pack(h) -> List[float]:
        return [
            h.hp,
            h.hp_ceiling,
            h.mp,
            h.mp_ceiling,
            h.shield,
            h.damage_resistance,
            h.revival_armor,
            1.0 if h.hp > 0 else 0.0,
        ]

    features = pack(hero) + pack(enemy)
    max_values = [
        hero.hp_ceiling or 1,
        hero.hp_ceiling or 1,
        hero.mp_ceiling or 1,
        hero.mp_ceiling or 1,
        max(hero.shield, 1),
        max(hero.damage_resistance, 1),
        max(hero.revival_armor, 1),
        1,
    ] * 2

    normalized = [
        (f / m) if m > 0 else 0.0
        for f, m in zip(features, max_values)
    ]
    return torch.tensor([normalized], dtype=torch.float32)


def simulate_round(hero, enemy, hero_action: int, enemy_action: int):
    """模拟一回合行动并返回新状态。"""
    hero = copy.deepcopy(hero)
    enemy = copy.deepcopy(enemy)
    players = [hero, enemy]

    for player in players:
        player.reset_stacks()

    _apply_actions(players, [(hero_action, 1), (enemy_action, 0)])
    for player in players:
        player.update_status()
    for player in players:
        player.apply_stamp()
    for player in players:
        player.update_status()

    return hero, enemy


def _apply_actions(players, actions: Sequence[Tuple[int, Optional[int]]]) -> None:
    """按规则顺序应用动作。"""
    action_order = {
        "self-attack": 0,
        "group-attack": 1,
        "anti-group-attack": 2,
        "attack": 3,
        "defense": 4,
        "control": 5,
        None: 6,
    }

    work_list = []
    for idx, (action_id, target_idx) in enumerate(actions):
        player = players[idx]
        info = next((info for info in player.action_list if info[0] == action_id), None)
        action_type = info[4] if info is not None else None
        work_list.append((player, action_id, target_idx, action_type))

    work_list.sort(key=lambda x: action_order.get(x[3], 999))
    for player, action_id, target_idx, _ in work_list:
        if player.hp <= 0 and player.revival_armor <= 0:
            continue
        target = players[target_idx] if target_idx is not None else None
        if target is not None and target.hp <= 0 and target.revival_armor <= 0:
            target = None
        player.apply_action(action_id, target, players)


def _regret_matching(regrets: Dict[int, float]) -> Dict[int, float]:
    """标准后悔匹配策略。"""
    positives = {a: max(0.0, r) for a, r in regrets.items()}
    total = sum(positives.values())
    if total <= 1e-8:
        uniform = 1.0 / len(regrets)
        return {a: uniform for a in regrets}
    return {a: positives[a] / total for a in regrets}


def _nash_value(actions_a: List[int], actions_b: List[int], payoff: List[List[float]], iters: int) -> float:
    """用 CFR 近似零和博弈值。"""
    regrets_a = {a: 0.0 for a in actions_a}
    regrets_b = {b: 0.0 for b in actions_b}

    value = 0.0
    for _ in range(iters):
        strat_a = _regret_matching(regrets_a)
        strat_b = _regret_matching(regrets_b)

        expected = 0.0
        for i, a in enumerate(actions_a):
            row_value = 0.0
            for j, b in enumerate(actions_b):
                row_value += strat_b[b] * payoff[i][j]
            expected += strat_a[a] * row_value

        for i, a in enumerate(actions_a):
            alt = sum(strat_b[actions_b[j]] * payoff[i][j] for j in range(len(actions_b)))
            regrets_a[a] += alt - expected
        for j, b in enumerate(actions_b):
            alt = sum(strat_a[actions_a[i]] * (-payoff[i][j]) for i in range(len(actions_a)))
            regrets_b[b] += alt - (-expected)

        value = expected

    return value


def _hero_signature(hero) -> Tuple:
    """用于缓存的英雄状态签名。"""
    return (
        hero.__class__.__name__,
        hero.hp,
        hero.hp_ceiling,
        hero.mp,
        hero.mp_ceiling,
        hero.shield,
        hero.damage_resistance,
        hero.revival_armor,
    )


def _is_terminal(hero, enemy) -> bool:
    """判断是否结束。"""
    return hero.hp <= 0 or enemy.hp <= 0


def _terminal_value(hero, enemy) -> float:
    """终局价值（胜负为 1/-1）。"""
    if hero.hp <= 0 and enemy.hp <= 0:
        return 0.0
    if enemy.hp <= 0:
        return 1.0
    if hero.hp <= 0:
        return -1.0
    return 0.0


def _policy_to_vector(policy: Dict[int, float], size: int = 12) -> List[float]:
    """将稀疏策略映射为定长向量。"""
    vec = [0.0] * size
    for action, prob in policy.items():
        if 0 <= action < size:
            vec[action] = prob
    return vec


def _sample_from_policy(policy: Dict[int, float]) -> int:
    """根据策略分布采样动作。"""
    actions = list(policy.keys())
    probs = [policy[a] for a in actions]
    return random.choices(actions, weights=probs, k=1)[0]


def _apply_dirichlet_noise(
    policy: Dict[int, float],
    alpha: float,
    epsilon: float,
) -> Dict[int, float]:
    if alpha <= 0 or epsilon <= 0:
        return policy

    actions = list(policy.keys())
    if not actions:
        return policy

    noise = torch.distributions.Dirichlet(
        torch.full((len(actions),), alpha)
    ).sample().tolist()
    mixed = {}
    for idx, action in enumerate(actions):
        mixed[action] = (1.0 - epsilon) * policy[action] + epsilon * noise[idx]
    return mixed


if __name__ == "__main__":
    from heros.basic import BasicHero
    import re

    parser = argparse.ArgumentParser(description="Self-play training")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from latest checkpoint in ai/basic_vs_basic",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu|cuda|auto",
    )
    parser.add_argument("--minutes", type=float, default=0.0, help="Train time in minutes")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Eval episodes")
    parser.add_argument("--eval-rounds", type=int, default=30, help="Eval max rounds")
    args = parser.parse_args()

    hero_a = BasicHero("AI-1")
    hero_b = BasicHero("AI-2")
    run_date = datetime.now().strftime("%Y%m%d")
    tag = f"basic_vs_basic_{run_date}"

    def _find_latest_model(path: str, context_tag: str) -> Optional[str]:
        if not os.path.isdir(path):
            return None
        pattern = re.compile(rf"^{re.escape(context_tag)}_(\d{{8}})_model\.pt$")
        candidates: list[tuple[str, str]] = []
        for name in os.listdir(path):
            match = pattern.match(name)
            if match:
                candidates.append((match.group(1), name))
        if not candidates:
            return None
        candidates.sort()
        return os.path.join(path, candidates[-1][1])

    run_config = TrainConfig()
    if args.device == "auto":
        run_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        run_config.device = args.device
    if args.minutes > 0:
        run_config.max_seconds = args.minutes * 60.0
    run_model = BattleNet(BattleGenome())
    resume_path = args.resume
    if resume_path is None and args.resume_latest:
        resume_path = _find_latest_model(os.path.join("ai", "basic_vs_basic"), "basic_vs_basic")
        if resume_path:
            print(f"[Resume] using latest: {resume_path}")
    trainer = train_selfplay(run_model, run_config, resume_path=resume_path)

    artifacts = save_training_run(
        os.path.join("ai", "basic_vs_basic"),
        run_config,
        trainer.loss_history,
        tag,
        model=run_model,
        use_timestamp=False,
        policy_losses=trainer.policy_loss_history,
        value_losses=trainer.value_loss_history,
        lr_history=trainer.lr_history,
        depth_history=trainer.depth_history,
    )
    opponent_model = BattleNet(BattleGenome())
    opponent_model.to(next(run_model.parameters()).device)
    metrics = evaluate_battle(
        run_model,
        opponent_model,
        episodes=args.eval_episodes,
        max_rounds=args.eval_rounds,
        search_depth=run_config.search_depth,
        cfr_iters=run_config.cfr_iters,
    )
    print("Saved training artifacts:", artifacts)
    print("Evaluation metrics:", metrics)
