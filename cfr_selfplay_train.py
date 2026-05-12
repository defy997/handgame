"""
CFR 自博弈训练 - 掌心战争
起始状态: 双方 HP=4, 能量=0

算法: 单步反事实遗憾 + 终局信号修正的 MC-CFR
克制链: 大招(3) > 单枪(0) > 清零(8) > 三枪(2) > 大招(3)
        反弹(7) > 单枪(0) | 单枪(0) 部分克制 双枪(1)/三枪(2)
清零条件: HP_MATRIX[对手技能][8] == 1 时回血 (对手出三枪/大招等)
空城被动: 0能量受攻击时额外反伤
"""

import numpy as np
import json
import os
from collections import defaultdict

from dqn_train import (
    NUM_SKILLS, MAX_HP, MAX_ENERGY,
    HP_MATRIX, HP_MATRIX_empty, ZERO_TABLE, SKILL_COST, SKILL_NAMES,
)

# ── 训练参数 ──────────────────────────────────────────────
INIT_HP    = 4      # 起始血量
INIT_EN    = 0     # 起始能量
MAX_ROUNDS = 30
GAMMA      = 0.99   # 折现率
BLEND_STEPS = 5     # 终局信号混合窗口

DEFAULT_SAVE_PATH = "cfr_output/cfr_strategy.npz"

# 克制链（攻方技能, 守方技能, 描述, 攻方最低能量, 守方最低能量）
# 只有双方能量同时满足时，克制关系才真正成立
# 完整克制关系由 COUNTER_ADV 矩阵显式定义（见下方），此处用于统计与日志
COUNTER_CHAIN = [
    # 主循环：大招 > 单枪 > 清零；三枪/双枪 > 大招；清零 > 三枪/大招
    (3, 0, "大招>单枪",    3, 1),   # 大招-3血穿透单枪
    (0, 8, "单枪>清零",    1, 2),   # 单枪打清零-2血且无能量清零
    (8, 2, "清零>三枪",    2, 2),   # 清零回血+双方清能
    (8, 3, "清零>大招",    2, 3),   # 清零回血+清大招能量
    (2, 3, "三枪>大招",    2, 3),   # 三枪打大招-2血
    (1, 3, "双枪>大招",    2, 3),   # 双枪打大招-2血且自身0受伤
    # 双枪克制
    (1, 8, "双枪>清零",    2, 2),   # 双枪打清零-3血
    # 反弹克制
    (7, 0, "反弹>单枪",    1, 1),   # 反弹弹回单枪-1
    (7, 1, "反弹>双枪",    1, 2),   # 反弹弹回双枪-2
    # 三枪克制
    (2, 1, "三枪>双枪",    2, 2),   # 三枪打双枪-1血
    (2, 7, "三枪>反弹",    2, 1),   # 三枪穿透反弹-1血
    # 大招克制
    (3, 7, "大招>反弹",    3, 1),   # 大招穿透反弹-3血
    # 清零额外
    (8, 7, "清零>反弹",    2, 1),   # 清零面对反弹回血
    # 单枪部分克制（HP-1，且能量更省：单枪费1 vs 双枪/三枪费2）
    (0, 1, "单枪>双枪",    1, 2),
    (0, 2, "单枪>三枪",    1, 2),
    # 能量效率克制（无直接伤害，靠能量经济差）
    (6, 5, "能量>大防",    0, 1),   # 能量+1能 vs 大防-1能，净差+2
    (6, 7, "能量>反弹",    0, 1),   # 能量+1能 vs 反弹-1能，净差+2
    (6, 4, "能量≥小防",   0, 0),   # 能量+1能 vs 小防±0能，净差+1（部分）
]

# 显式克制优势矩阵
# COUNTER_ADV[atk][def] = 出 atk 时对阵 def 的综合净优势（atk方视角，正值=atk占优）
#
# 三层叠加:
#   1. HP净优势:     HP_MATRIX[def][atk] - HP_MATRIX[atk][def]
#   2. 清零能量清除: +SKILL_COST[def]*0.4 (ZERO_TABLE 行)
#   3. 能量效率差:   (net_en[atk] - net_en[def]) * ENERGY_HP_VALUE
#      net_en[sk] = -SKILL_COST[sk] + (1 if sk==能量(6))
#      这一项让"用0费技能对阵1费技能"体现出能量经济优势
COUNTER_ADV = (HP_MATRIX.T - HP_MATRIX).astype(np.float32)

# 层2: 清零能量清除附加价值
for _d in range(NUM_SKILLS):
    if ZERO_TABLE[8][_d]:
        COUNTER_ADV[8][_d] += SKILL_COST[_d] * 0.4

# 层3: 能量效率差（1能 ≈ ENERGY_HP_VALUE HP）
# net_en[sk]: 使用该技能后的净能量变化（仅考虑费用和能量技能增益，不含ZERO_TABLE）
ENERGY_HP_VALUE = 0.35
_net_en = (-SKILL_COST + np.array([1 if s == 6 else 0
                                   for s in range(NUM_SKILLS)], dtype=np.int32)
           ).astype(np.float32)
COUNTER_ADV += (_net_en[:, None] - _net_en[None, :]) * ENERGY_HP_VALUE

# 层4: 空城反伤折减
# 当守方使用2费技能（双枪/三枪）恰好降至0能量时，空城被动对攻方造成1点反伤，
# 抵消 HP_MATRIX 里的1点优势。起始能量=2是标准场景，此时HP优势=0，实际收益≈1能量单位。
# 因此从 COUNTER_ADV 中减去被空城抵消的HP分量，使 单枪>双枪/三枪 = 0.35（纯能量效率）。
COUNTER_ADV[0][1] -= 1.0   # 单枪 vs 双枪
COUNTER_ADV[0][2] -= 1.0   # 单枪 vs 三枪

# 预计算零费技能（空城时唯一可出的技能）
_ZERO_COST = [s for s in range(NUM_SKILLS) if SKILL_COST[s] == 0]


def _danger_signal(pHP, pEn, a_legal):
    """
    P1 无能量时, 对手下回合对 P1 的最大致命威胁, 返回 ∈ [-2, 0]。
    -2 = 对手必然能重创/秒杀; 0 = 对手无法造成有效伤害。

    P1 无能量只能出 0 费技能：
      - 技能 6（能量）：使 P1 能量变 1，逃脱空城 → 用 HP_MATRIX
      - 其他 0 费技能（小防等）：P1 仍在空城 → 用 HP_MATRIX_empty
    对手技能全部枚举（不再依赖 SKILL_ATK 粗判断）。
    """
    if pEn > 0 or not a_legal:
        return 0.0
    max_dmg = 0
    for a2 in a_legal:
        for p1d in _ZERO_COST:
            if p1d == 6:
                # P1 打出能量技，能量变 1，不触发空城
                d = max(0, -int(HP_MATRIX[a2][p1d]))
            else:
                # P1 仍在空城，用空城矩阵
                d = max(0, -int(HP_MATRIX_empty[a2][p1d]))
            if d > max_dmg:
                max_dmg = d
    if max_dmg == 0:
        return 0.0
    lethality = min(max_dmg / max(pHP, 0.5), 2.0)
    urgency = max(0.0, 1.0 - (pHP - 1) / max(MAX_HP - 1, 1))
    return -lethality * (0.5 + 0.5 * urgency)  # ∈ [-2, 0]


# ── 可学习启发参数 ──────────────────────────────────────────

class HeuristicParams:
    """
    启发函数可学习权重。在 CFR 训练过程中, 用终局信号在线调整。

    解决核心盲点: 静态 COUNTER_ADV 不感知 HP 上下文。
    当 HP 低 + 无能量时, 即使本回合克制收益大, 下回合被必杀的风险应压倒一切。

    权重结构 self.w[phase][4]:
      [0] hp_w      HP 优势权重
      [1] en_w      能量优势权重
      [2] counter_w 克制对位权重
      [3] danger_w  濒死+无能量危险惩罚权重（最关键的可学习量）
    """
    LR = 0.003          # 基础学习率
    LR_DANGER = 0.012   # 危险系数专用学习率（更快响应致命情境）

    def __init__(self):
        # 初始值基于手工调参，训练后自动收敛
        self.w = np.array([
            [0.88, 0.05, 0.025, 0.30],  # phase 0: 双空城
            [0.60, 0.22, 0.10,  0.30],  # phase 1: 低能
            [0.48, 0.26, 0.10,  0.40],  # phase 2: 高能
        ], dtype=np.float64)

    def eval(self, pHP, pEn, aHP, aEn, p_legal, a_legal):
        """计算 P1 视角启发值, 包含危险惩罚"""
        if pHP <= 0: return -1.0
        if aHP <= 0: return  1.0

        ph = energy_phase(pEn, aEn)
        w  = self.w[ph]
        hp_adv = (pHP - aHP) / MAX_HP
        en_adv = (pEn - aEn) / (MAX_ENERGY + 1)

        counter = 0.0
        if p_legal and a_legal:
            p1g = max(min(float(COUNTER_ADV[p][a]) for a in a_legal) for p in p_legal)
            counter = float(np.clip(p1g, -MAX_HP, MAX_HP)) / MAX_HP

        danger = _danger_signal(pHP, pEn, a_legal)  # ∈ [-2, 0]

        extra = 0.0
        if ph == 1:
            extra = (0.08 if pEn >= 3 else 0.0) - (0.08 if aEn >= 3 else 0.0)
        elif ph == 2:
            extra = ((0.07 if pEn >= 3 else 0.0) - (0.07 if aEn >= 3 else 0.0)
                     + (-0.06 if pEn == 0 and aEn > 0 else 0.0))

        val = w[0]*hp_adv + w[1]*en_adv + w[2]*counter + w[3]*danger + extra
        return float(np.clip(val, -1.5, 1.5))

    def update_episode(self, records, final_v):
        """
        用一局的终局信号更新权重。

        records: list of (pHP, pEn, aHP, aEn, a_legal, pred_v)
          每条记录是某步棋后 P1 的实际状态及当时的启发预测值。
        final_v: 该局 P1 视角终局价值 ∈ [-1.25, 1.25]
        """
        for pHP, pEn, aHP, aEn, a_legal, pred_v in records:
            error = final_v - pred_v   # 正 = 实际比预测好; 负 = 预测过于乐观
            ph = energy_phase(pEn, aEn)
            w  = self.w[ph]
            hp_adv = (pHP - aHP) / MAX_HP
            en_adv = (pEn - aEn) / (MAX_ENERGY + 1)

            # HP / 能量权重: 小步全局更新
            w[0] = float(np.clip(w[0] + self.LR * error * hp_adv,  0.10, 1.50))
            w[1] = float(np.clip(w[1] + self.LR * error * en_adv,  0.00, 0.80))

            # 危险权重: 仅在濒死+无能量状态下更新, 且使用更高学习率
            if pEn == 0 and pHP <= 2:
                danger = _danger_signal(pHP, pEn, a_legal)
                if danger < -0.01:
                    # danger ∈ (-2, 0), error < 0 (预测太乐观) → 增大危险惩罚
                    w[3] = float(np.clip(
                        w[3] - self.LR_DANGER * error * (-danger),
                        0.00, 2.50
                    ))

    def weights_str(self):
        lines = []
        for ph, row in enumerate(self.w):
            lines.append(f"  phase{ph}: hp={row[0]:.3f} en={row[1]:.3f} "
                         f"ctr={row[2]:.3f} danger={row[3]:.3f}")
        return "\n".join(lines)


# 能量阶段划分
# phase 0: max(pEn,aEn) == 0          → 双方空城，仅0费技能可用
# phase 1: 1 <= max(pEn,aEn) <= 2     → 部分克制链可触发（无大招）
# phase 2: max(pEn,aEn) >= 3          → 完整克制链，大招可用
def energy_phase(pEn, aEn):
    m = max(pEn, aEn)
    if m == 0:   return 0
    if m <= 2:   return 1
    return 2

PHASE_LABELS = ["双空城", "低能(无大招)", "高能(完整克制链)"]


# ── 状态工具 ──────────────────────────────────────────────

def state_id(pHP, pEn, aHP, aEn):
    return (pHP * 7 + pEn) * 35 + (aHP * 7 + aEn)


def legal_actions(en):
    acts = [s for s in range(NUM_SKILLS) if SKILL_COST[s] <= en]
    return acts if acts else [6]


def game_step(pHP, pEn, aHP, aEn, pSk, aSk):
    """单步结算，返回 (pHP', pEn', aHP', aEn', done, winner)"""
    pEn2 = min(MAX_ENERGY, max(0, pEn - int(SKILL_COST[pSk])) + (1 if pSk == 6 else 0))
    aEn2 = min(MAX_ENERGY, max(0, aEn - int(SKILL_COST[aSk])) + (1 if aSk == 6 else 0))

    if ZERO_TABLE[pSk][aSk]: pEn2 = aEn2 = 0
    if ZERO_TABLE[aSk][pSk]: pEn2 = aEn2 = 0

    p_dmg  = int(HP_MATRIX[pSk][aSk])   # → aHP
    ai_dmg = int(HP_MATRIX[aSk][pSk])   # → pHP

    aHP2 = max(0, min(MAX_HP, aHP + p_dmg))
    pHP2 = max(0, min(MAX_HP, pHP + ai_dmg))
    if pHP2 <= 0 or aHP2 <= 0:
        return pHP2, pEn2, aHP2, aEn2, True, _winner(pHP2, aHP2)

    # 空城被动：对方能量为0且未被击杀，改用空城矩阵
    if aEn2 == 0:
        ai_dmg = int(HP_MATRIX_empty[aSk][pSk])
    if pEn2 == 0 and pHP + ai_dmg > 0:
        p_dmg  = int(HP_MATRIX_empty[pSk][aSk])

    aHP2 = max(0, min(MAX_HP, aHP + p_dmg))
    pHP2 = max(0, min(MAX_HP, pHP + ai_dmg))

    done = pHP2 <= 0 or aHP2 <= 0
    return pHP2, pEn2, aHP2, aEn2, done, _winner(pHP2, aHP2) if done else 0


def _winner(pHP, aHP):
    if pHP <= 0 and aHP <= 0: return 3
    if aHP <= 0: return 1
    return 2


def terminal_value(winner, pHP, aHP):
    """P1 视角终局价值 ∈ [-1.25, 1.25]"""
    if winner == 1: return 1.0 + pHP * 0.05
    if winner == 2: return -1.0 - aHP * 0.05
    return 0.0


def heuristic_value(pHP, pEn, aHP, aEn, p_legal=None, a_legal=None, params=None):
    """
    能量阶段感知的状态评估（P1 视角）

    Phase 0 (空城): 能量为 0 的空城被动主导，HP 优势权重极高
    Phase 1 (低能): 只有部分克制链可用，能量积累价值高
    Phase 2 (高能): 完整克制链激活，相对能量优势决定战略主动权

    p_legal / a_legal: 双方在当前能量下的合法技能集合。
    params: HeuristicParams 实例，非 None 时使用可学习权重评估。
    """
    if params is not None:
        return params.eval(pHP, pEn, aHP, aEn,
                           p_legal if p_legal is not None else [],
                           a_legal if a_legal is not None else [])
    if pHP <= 0: return -1.0
    if aHP <= 0: return  1.0

    phase = energy_phase(pEn, aEn)
    hp_adv = (pHP - aHP) / MAX_HP

    # ── 克制对位优势（需要知道双方可用技能集）─────────────────
    # P1 的最坏情况净优势：P1 最优选择下 P2 最差应对的克制收益
    counter_bal = 0.0
    if p_legal is not None and a_legal is not None and p_legal and a_legal:
        p1_guarantee = max(
            min(float(COUNTER_ADV[p][a]) for a in a_legal)
            for p in p_legal
        )
        counter_bal = float(np.clip(p1_guarantee, -MAX_HP, MAX_HP)) / MAX_HP * 0.10

    if phase == 0:
        # 双方空城：血量几乎决定一切；克制无从发挥，但保留小项
        return hp_adv * 0.88 + ((pEn - aEn) / (MAX_ENERGY + 1)) * 0.05 + counter_bal * 0.5

    elif phase == 1:
        # 低能阶段：克制链部分激活，传入的可用技能集决定具体对位
        en_adv = (pEn - aEn) / (MAX_ENERGY + 1)
        p1_ulti_ready = 0.08 if pEn >= 3 else 0.0
        p2_ulti_ready = 0.08 if aEn >= 3 else 0.0
        return hp_adv * 0.60 + en_adv * 0.22 + p1_ulti_ready - p2_ulti_ready + counter_bal

    else:
        # 高能阶段：完整克制链，counter_bal 权重最高
        en_adv = (pEn - aEn) / (MAX_ENERGY + 1)
        ulti_edge = (0.07 if pEn >= 3 else 0.0) - (0.07 if aEn >= 3 else 0.0)
        empty_risk = -0.08 if pEn == 0 and aEn > 0 else 0.0
        return hp_adv * 0.48 + en_adv * 0.26 + ulti_edge + empty_risk + counter_bal


def step_cf_value(php, pen, ahp, aen, alt_p, fixed_a, final_v, gamma, params=None):
    """
    P1 在当前状态出 alt_p, P2 出 fixed_a 的反事实价值估算。
    用 (1-gamma) * 即时启发 + gamma * 终局信号 混合。
    传入下一状态双方可用技能集，使启发函数感知克制对位。
    """
    nhp, nen, nahp, naen, done, w = game_step(php, pen, ahp, aen, alt_p, fixed_a)
    if done:
        return terminal_value(w, nhp, nahp)
    next_p_legal = legal_actions(nen)
    next_a_legal = legal_actions(naen)
    imm = heuristic_value(nhp, nen, nahp, naen, next_p_legal, next_a_legal, params)
    return (1 - gamma) * imm + gamma * final_v


# ── CFR 求解器 ────────────────────────────────────────────

N_STATES = (MAX_HP + 1) * (MAX_ENERGY + 1) * (MAX_HP + 1) * (MAX_ENERGY + 1)  # 1225


class CFRSolver:
    """
    双方独立 CFR，每个玩家维护:
      R[state][action] - 累积反事实遗憾
      S[state][action] - 累积策略权重（平均后趋向 Nash）
    """

    def __init__(self):
        self.R = [np.zeros((N_STATES, NUM_SKILLS), dtype=np.float64) for _ in range(2)]
        self.S = [np.zeros((N_STATES, NUM_SKILLS), dtype=np.float64) for _ in range(2)]
        self.counter_hits  = defaultdict(int)
        self.phase_steps   = [0, 0, 0]   # phase 0/1/2 各自的步数统计
        self.total_steps   = 0
        self.episode_count = 0
        self.h_params      = HeuristicParams()

    def current_strategy(self, player, sid, legal):
        """遗憾匹配 → 当前混合策略"""
        r = np.maximum(self.R[player][sid], 0)
        mask = np.zeros(NUM_SKILLS)
        mask[legal] = 1.0
        r = r * mask
        total = r.sum()
        if total > 1e-12:
            return r / total
        return mask / mask.sum()

    def average_strategy(self, player, sid, legal):
        """平均策略（Nash 均衡近似）"""
        s = self.S[player][sid] * np.array(
            [1.0 if a in set(legal) else 0.0 for a in range(NUM_SKILLS)]
        )
        total = s.sum()
        if total > 1e-12:
            return s / total
        mask = np.zeros(NUM_SKILLS)
        mask[legal] = 1.0 / len(legal)
        return mask

    def _rollout_value(self, nhp, nen, nahp, naen, g, final_v):
        """
        用一步真实展开替代纯启发估值（P1 视角）。
        枚举双方下一步所有合法动作，game_step 自动处理空城矩阵，
        自然权重了"下回合对手/自己能量为0"的各种概率。
        """
        ns   = state_id(nhp, nen, nahp, naen)
        npl  = legal_actions(nen)
        nal  = legal_actions(naen)
        sg1n = self.current_strategy(0, ns, npl)
        sg2n = self.current_strategy(1, ns, nal)
        imm  = 0.0
        for na1 in npl:
            for na2 in nal:
                nhp2, nen2, nahp2, naen2, d2, w2 = game_step(nhp, nen, nahp, naen, na1, na2)
                if d2:
                    vi = terminal_value(w2, nhp2, nahp2)
                else:
                    vi = heuristic_value(nhp2, nen2, nahp2, naen2,
                                         legal_actions(nen2), legal_actions(naen2),
                                         self.h_params)
                imm += sg1n[na1] * sg2n[na2] * vi
        return (1 - g) * imm + g * final_v

    def run_episode(self, init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN)):
        """运行完整一局，收集轨迹并执行 CFR 遗憾更新"""
        pHP, pEn, aHP, aEn = init
        traj = []
        h_records = []   # (pHP, pEn, aHP, aEn, a_legal, pred_v) for HeuristicParams
        done = False; winner = 3; rnd = 0

        while not done and rnd < MAX_ROUNDS:
            rnd += 1
            s  = state_id(pHP, pEn, aHP, aEn)
            l1 = legal_actions(pEn)
            l2 = legal_actions(aEn)
            sg1 = self.current_strategy(0, s, l1)
            sg2 = self.current_strategy(1, s, l2)
            a1  = int(np.random.choice(NUM_SKILLS, p=sg1))
            a2  = int(np.random.choice(NUM_SKILLS, p=sg2))
            traj.append((s, l1, l2, sg1, sg2, a1, a2, pHP, pEn, aHP, aEn))
            self.phase_steps[energy_phase(pEn, aEn)] += 1
            pHP, pEn, aHP, aEn, done, winner = game_step(pHP, pEn, aHP, aEn, a1, a2)

            # 记录实际后继状态的启发预测值，供 h_params 学习
            if not done:
                npl = legal_actions(pEn)
                nal = legal_actions(aEn)
                pred_v = heuristic_value(pHP, pEn, aHP, aEn, npl, nal, self.h_params)
                h_records.append((pHP, pEn, aHP, aEn, nal, pred_v))

        final_v = terminal_value(winner, pHP, aHP)
        T = len(traj)
        self.episode_count += 1
        self.total_steps   += T

        # ── 反向遍历轨迹，更新 CFR 遗憾 ───────────────────────
        for t, (s, l1, l2, sg1, sg2, a1, a2, php, pen, ahp, aen) in enumerate(traj):
            steps_to_end = T - t - 1
            # gamma 越大 → 越依赖终局信号；越小 → 越依赖即时启发
            g = GAMMA ** steps_to_end if steps_to_end < BLEND_STEPS else 0.0

            # P1 反事实价值向量（external sampling + 一步展开，空城矩阵自动枚举）
            cf1 = np.zeros(NUM_SKILLS)
            for alt in l1:
                v = 0.0
                for opp in l2:
                    nhp, nen, nahp, naen, d, w = game_step(php, pen, ahp, aen, alt, opp)
                    if d:
                        vi = terminal_value(w, nhp, nahp)
                    else:
                        vi = self._rollout_value(nhp, nen, nahp, naen, g, final_v)
                    v += sg2[opp] * vi
                cf1[alt] = v
            ev1 = float(sg1 @ cf1)

            # P2 反事实价值向量（external sampling + 一步展开，零和取负）
            cf2 = np.zeros(NUM_SKILLS)
            for alt in l2:
                v = 0.0
                for opp in l1:
                    nhp, nen, nahp, naen, d, w = game_step(php, pen, ahp, aen, opp, alt)
                    if d:
                        vi = -terminal_value(w, nhp, nahp)
                    else:
                        vi = -self._rollout_value(nhp, nen, nahp, naen, g, final_v)
                    v += sg1[opp] * vi
                cf2[alt] = v
            ev2 = float(sg2 @ cf2)

            # 更新遗憾
            for alt in l1:
                self.R[0][s][alt] += cf1[alt] - ev1
            for alt in l2:
                self.R[1][s][alt] += cf2[alt] - ev2

            # 更新策略累积（线性加权：越晚的迭代权重越高）
            weight = float(self.episode_count)
            self.S[0][s] += weight * sg1
            self.S[1][s] += weight * sg2

        # ── 更新可学习启发参数 ────────────────────────────────
        if h_records:
            self.h_params.update_episode(h_records, final_v)

        # ── 克制链统计（仅在双方能量满足条件时才计入）───────────
        for _, _, _, _, _, a1, a2, php, pen, ahp, aen in traj:
            for atk, dfn, lbl, en_atk, en_dfn in COUNTER_CHAIN:
                # P1 是攻方，P2 是守方：检查 P1 能量 >= en_atk 且 P2 能量 >= en_dfn
                if a1 == atk and a2 == dfn and pen >= en_atk and aen >= en_dfn:
                    self.counter_hits[lbl] += 1
                # 反向：P2 是攻方
                elif a2 == atk and a1 == dfn and aen >= en_atk and pen >= en_dfn:
                    self.counter_hits[lbl] += 1

        return winner, T


# ── 评估工具 ──────────────────────────────────────────────

def best_response_winrate(solver, player_br=2, n_eval=300,
                          init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN)):
    """
    player_br 方用针对对手平均策略的贪心最佳响应，
    估算其胜率作为可利用度上界。
    """
    fixed_player = 1 - (player_br - 1)  # 0 or 1
    br_wins = 0

    for _ in range(n_eval):
        pHP, pEn, aHP, aEn = init
        done = False; rnd = 0

        while not done and rnd < MAX_ROUNDS:
            rnd += 1
            s  = state_id(pHP, pEn, aHP, aEn)
            l1 = legal_actions(pEn)
            l2 = legal_actions(aEn)

            # 固定方：用平均策略采样
            fixed_sg = solver.average_strategy(fixed_player, s, l1 if fixed_player == 0 else l2)
            fixed_a  = int(np.random.choice(NUM_SKILLS, p=fixed_sg))

            # BR 方：对固定方当前策略求贪心最佳响应（1步展开）
            br_legal = l2 if player_br == 2 else l1
            best_a = br_legal[0]; best_v = float('-inf')
            for alt in br_legal:
                p1, a = (fixed_a, alt) if fixed_player == 0 else (alt, fixed_a)
                nhp, nen, nahp, naen, d, w = game_step(pHP, pEn, aHP, aEn, p1, a)
                if d:
                    v = terminal_value(w, nhp, nahp) * (1 if player_br == 1 else -1)
                else:
                    next_p_legal = legal_actions(nen)
                    next_a_legal = legal_actions(naen)
                    v = heuristic_value(nhp, nen, nahp, naen, next_p_legal, next_a_legal,
                                        solver.h_params) * (1 if player_br == 1 else -1)
                if v > best_v:
                    best_v = v; best_a = alt

            a1, a2 = (fixed_a, best_a) if fixed_player == 0 else (best_a, fixed_a)
            pHP, pEn, aHP, aEn, done, winner = game_step(pHP, pEn, aHP, aEn, a1, a2)

        if winner == player_br:
            br_wins += 1

    return br_wins / n_eval


# ── 策略分析 ──────────────────────────────────────────────

def print_strategy(solver, pHP, pEn, aHP, aEn, label=""):
    sid = state_id(pHP, pEn, aHP, aEn)
    l1  = legal_actions(pEn)
    l2  = legal_actions(aEn)
    avg1 = solver.average_strategy(0, sid, l1)
    avg2 = solver.average_strategy(1, sid, l2)
    tag  = label or f"HP={pHP} En={pEn} vs HP={aHP} En={aEn}"
    print(f"  [{tag}]")
    p1_top = [(SKILL_NAMES[a], avg1[a]) for a in range(NUM_SKILLS) if avg1[a] > 0.04]
    p2_top = [(SKILL_NAMES[a], avg2[a]) for a in range(NUM_SKILLS) if avg2[a] > 0.04]
    p1_str = "  ".join(f"{n}={p:.0%}" for n, p in sorted(p1_top, key=lambda x: -x[1]))
    p2_str = "  ".join(f"{n}={p:.0%}" for n, p in sorted(p2_top, key=lambda x: -x[1]))
    print(f"    P1: {p1_str}")
    print(f"    P2: {p2_str}")


def analyze_nash(solver, init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN)):
    print("\n[Nash 均衡混合策略分析 — 按能量阶段分层]")

    # ── Phase 0: 空城状态 ─────────────────────────────────
    print(f"\n  === Phase 0: 双方空城（0费技能支配）===")
    for hp in [3, 2, 1]:
        print_strategy(solver, hp, 0, hp, 0, f"对称 HP={hp} En=0")

    # ── Phase 1: 低能（1-2能，大招不可用）────────────────
    print(f"\n  === Phase 1: 低能区（克制链部分激活，无大招）===")
    phase1_states = [
        (*init,           "起始状态 3HP 2能(对称)"),
        (2, 2, 2, 2,      "均势 2HP 2能"),
        (2, 1, 2, 2,      "P1 1能 vs P2 2能"),
        (2, 2, 2, 1,      "P1 2能 vs P2 1能"),
        (3, 2, 3, 1,      "P1能量优势"),
    ]
    for state in phase1_states:
        print_strategy(solver, *state[:4], state[4])

    # ── Phase 2: 高能（3+能，完整克制链）─────────────────
    print(f"\n  === Phase 2: 高能区（完整克制链激活）===")
    phase2_states = [
        (3, 3, 3, 3,      "高能对称 3能"),
        (2, 3, 2, 2,      "P1 大招就绪 vs P2 2能"),
        (2, 2, 2, 3,      "P2 大招就绪 vs P1 2能"),
        (1, 3, 2, 3,      "P1濒死双方高能"),
        (3, 3, 3, 1,      "P2低能 P1大招压制"),
    ]
    for state in phase2_states:
        print_strategy(solver, *state[:4], state[4])

    # ── 起始状态详细 ──────────────────────────────────────
    print(f"\n  === 关键混合状态 ===")
    mixed_states = [
        (init[0], 0, init[2], 0,   "双方空城"),
        (2, 0, 2, 2,               "P1空城 P2有能量"),
        (1, 3, 3, 1,               "P1濒死高能 vs P2健康低能"),
    ]
    for state in mixed_states:
        print_strategy(solver, *state[:4], state[4])

    # ── 克制链能量条件统计 ──────────────────────────────
    total = solver.total_steps
    print(f"\n[克制链触发统计 - 仅统计双方能量条件满足的回合] (共 {total} 步)")
    for _, _, lbl, en_atk, en_dfn in COUNTER_CHAIN:
        cnt = solver.counter_hits.get(lbl, 0)
        rate = cnt / max(total, 1)
        bar  = "█" * int(rate * 50)
        print(f"  {lbl:14s}(攻≥{en_atk}能,守≥{en_dfn}能): "
              f"{cnt:5d}次 {rate:.3%} {bar}")


def save_strategy(solver, path=DEFAULT_SAVE_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, R0=solver.R[0], R1=solver.R[1],
             S0=solver.S[0], S1=solver.S[1],
             hw=solver.h_params.w)
    print(f"[保存] 策略 → {path}")


def load_strategy(solver, path=DEFAULT_SAVE_PATH):
    if not os.path.exists(path):
        return False
    d = np.load(path)
    solver.R[0][:] = d["R0"]; solver.R[1][:] = d["R1"]
    solver.S[0][:] = d["S0"]; solver.S[1][:] = d["S1"]
    if "hw" in d:
        solver.h_params.w[:] = d["hw"]
    print(f"[加载] 策略 ← {path}")
    return True


# ── 主训练循环 ────────────────────────────────────────────

def train_cfr(
    n_iters      = 60_000,
    log_every    = 3_000,
    eval_every   = 10_000,
    save_every   = 20_000,
    init_state   = (INIT_HP, INIT_EN, INIT_HP, INIT_EN),
    save_path    = DEFAULT_SAVE_PATH,
    resume       = True,
):
    solver = CFRSolver()
    if resume:
        load_strategy(solver, save_path)

    print(f"\n{'='*58}")
    print(f"  CFR 自博弈训练 - 掌心战争")
    print(f"  起始: HP={init_state[0]} 能量={init_state[1]}  (双方对称)")
    print(f"  克制链: 大招>单枪>清零>三枪>大招  |  反弹>单枪")
    print(f"  清零条件: HP_MATRIX[对手技能][8]==1 (回血触发)")
    print(f"  算法: MC-CFR + 终局信号修正 + 线性加权平均策略")
    print(f"  迭代: {n_iters}  log_every: {log_every}")
    print(f"{'='*58}\n")

    wins = [0, 0, 0]  # P1, P2, draw
    log_data = []

    for it in range(1, n_iters + 1):
        winner, rnd = solver.run_episode(init_state)
        wins[winner - 1] += 1

        if it % log_every == 0:
            tot = sum(wins)
            p1_wr = wins[0] / tot
            p2_wr = wins[1] / tot
            draw  = wins[2] / tot
            print(f"[{it:7d}] P1={p1_wr:.1%}  P2={p2_wr:.1%}  平={draw:.1%}  "
                  f"eps_count={solver.episode_count}")

            # 能量阶段分布
            ps = solver.phase_steps
            ps_tot = max(sum(ps), 1)
            print(f"         阶段分布: "
                  f"空城={ps[0]/ps_tot:.0%}  "
                  f"低能={ps[1]/ps_tot:.0%}  "
                  f"高能(克制链)={ps[2]/ps_tot:.0%}")

            top5_counters = sorted(solver.counter_hits.items(), key=lambda x: -x[1])[:3]
            if top5_counters:
                ctr_str = "  ".join(f"{lbl}:{cnt}" for lbl, cnt in top5_counters)
                print(f"         克制触发(能量满足): {ctr_str}")

            print_strategy(solver, *init_state, "起始状态")
            print(f"  启发参数:\n{solver.h_params.weights_str()}")
            log_data.append({"iter": it, "p1_wr": p1_wr, "p2_wr": p2_wr, "draw": draw})
            wins = [0, 0, 0]

        if it % eval_every == 0:
            br2 = best_response_winrate(solver, player_br=2, n_eval=300, init=init_state)
            br1 = best_response_winrate(solver, player_br=1, n_eval=300, init=init_state)
            exploit = (br1 + br2) / 2
            print(f"  [可利用度估算] BR_P1={br1:.1%}  BR_P2={br2:.1%}  "
                  f"Avg_exploit={exploit:.1%}")

        if it % save_every == 0:
            save_strategy(solver, save_path)

    save_strategy(solver, save_path)

    # 保存训练日志
    log_path = save_path.replace(".npz", "_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"[保存] 训练日志 → {log_path}")

    analyze_nash(solver, init_state)
    return solver


# ── 离线分析入口（加载已有策略）────────────────────────────

def analyze_saved(path=DEFAULT_SAVE_PATH,
                  init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN)):
    solver = CFRSolver()
    if not load_strategy(solver, path):
        print(f"找不到策略文件: {path}")
        return
    analyze_nash(solver, init)
    br2 = best_response_winrate(solver, player_br=2, n_eval=500, init=init)
    br1 = best_response_winrate(solver, player_br=1, n_eval=500, init=init)
    print(f"\n[可利用度] BR_P1={br1:.1%}  BR_P2={br2:.1%}  "
          f"理论Nash差距≈{(br1+br2)/2:.1%}")


# ── DQN 热启动：从 CFR 平均策略采集数据 ───────────────────

def generate_cfr_dataset(solver, n_games=5000,
                          init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN),
                          save_path="cfr_output/cfr_dataset.json"):
    """
    用 CFR 平均策略跑 n_games 局，生成 (state, action, outcome) 数据集。
    可用于预训练 DQNAgent（热启动），跳过随机探索阶段。
    """
    dataset = []
    wins = [0, 0, 0]

    for _ in range(n_games):
        pHP, pEn, aHP, aEn = init
        done = False; rnd = 0; traj = []

        while not done and rnd < MAX_ROUNDS:
            rnd += 1
            s  = state_id(pHP, pEn, aHP, aEn)
            l1 = legal_actions(pEn)
            l2 = legal_actions(aEn)
            sg1 = solver.average_strategy(0, s, l1)
            sg2 = solver.average_strategy(1, s, l2)
            a1  = int(np.random.choice(NUM_SKILLS, p=sg1))
            a2  = int(np.random.choice(NUM_SKILLS, p=sg2))
            traj.append((pHP, pEn, aHP, aEn, a1, a2))
            pHP, pEn, aHP, aEn, done, winner = game_step(pHP, pEn, aHP, aEn, a1, a2)

        wins[winner - 1] += 1
        final_v = terminal_value(winner, pHP, aHP)

        for phpt, pent, ahpt, aent, a1t, a2t in traj:
            dataset.append({
                "pHP": phpt, "pEn": pent, "aHP": ahpt, "aEn": aent,
                "a1": a1t, "a2": a2t,
                "winner": winner, "final_value": final_v,
            })

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    tot = sum(wins)
    print(f"[CFR数据集] {len(dataset)} 条  P1={wins[0]/tot:.1%}  P2={wins[1]/tot:.1%}")
    print(f"[保存] → {save_path}")
    return dataset


# ── 人类对战模式 ──────────────────────────────────────────

def _print_state(pHP, pEn, aHP, aEn, rnd, human_is_p1):
    h_hp, h_en = (pHP, pEn) if human_is_p1 else (aHP, aEn)
    ai_hp, ai_en = (aHP, aEn) if human_is_p1 else (pHP, pEn)
    print(f"\n{'─'*44}")
    print(f"  第 {rnd} 回合")
    print(f"  你    HP={h_hp}/{MAX_HP}  能量={h_en}")
    print(f"  AI    HP={ai_hp}/{MAX_HP}  能量={ai_en}")
    print(f"{'─'*44}")


def _print_skills(legal):
    print("  可选技能:")
    for sk in legal:
        print(f"    [{sk}] {SKILL_NAMES[sk]:<4}  (费:{SKILL_COST[sk]}能)")


def _ask_skill(legal):
    while True:
        try:
            raw = input(f"  >> 出招 {legal} (q退出): ").strip()
            if raw.lower() == "q":
                raise KeyboardInterrupt
            sk = int(raw)
            if sk in legal:
                return sk
            print(f"  无效，可用编号: {legal}")
        except ValueError:
            print("  请输入整数编号")


def _print_outcome(h_sk, ai_sk, pre_pHP, pHP, pre_aHP, aHP, human_is_p1):
    h_pre, h_post = (pre_pHP, pHP) if human_is_p1 else (pre_aHP, aHP)
    ai_pre, ai_post = (pre_aHP, aHP) if human_is_p1 else (pre_pHP, pHP)
    print(f"\n  你出: {SKILL_NAMES[h_sk]:<4}  |  AI出: {SKILL_NAMES[ai_sk]}")
    h_d = h_post - h_pre
    ai_d = ai_post - ai_pre
    if h_d < 0:
        print(f"  你  -{-h_d} HP  (剩{h_post}/{MAX_HP})")
    elif h_d > 0:
        print(f"  你  +{h_d} HP  (剩{h_post}/{MAX_HP})")
    if ai_d < 0:
        print(f"  AI  -{-ai_d} HP  (剩{ai_post}/{MAX_HP})")
    elif ai_d > 0:
        print(f"  AI  +{ai_d} HP  (剩{ai_post}/{MAX_HP})")
    if h_d == 0 and ai_d == 0:
        print(f"  双方无伤")


def human_vs_cfr(solver, human_player=1,
                 init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN),
                 n_games=None, update_cfr=True,
                 save_path=DEFAULT_SAVE_PATH):
    """
    人类 vs CFR AI 对战。

    human_player: 1=人类先手(P1)  2=人类后手(P2)
    update_cfr  : True=每局结束后用人类行动更新AI的CFR遗憾（AI持续从对局中学习）
    n_games     : None=无限循环，直到输入 q
    """
    human_is_p1 = (human_player == 1)
    ai_idx = 1 if human_is_p1 else 0   # CFR solver 中 AI 对应的 player 索引
    wins = {"人类": 0, "AI": 0, "平局": 0}
    game_idx = 0

    print(f"\n{'='*44}")
    print(f"  人类 vs CFR AI  (你是 P{human_player})")
    print(f"  输入技能编号出招，q 退出")
    print(f"  AI学习模式: {'开启' if update_cfr else '关闭'}")
    print(f"{'='*44}")

    try:
        while n_games is None or game_idx < n_games:
            game_idx += 1
            pHP, pEn, aHP, aEn = init
            done = False; rnd = 0; winner = 3
            traj = []       # (sid, l1, l2, a1, a2, pHP_pre, pEn_pre, aHP_pre, aEn_pre)
            h_records = []  # for HeuristicParams.update_episode

            print(f"\n  ===== 第 {game_idx} 局 =====")

            while not done and rnd < MAX_ROUNDS:
                rnd += 1
                sid = state_id(pHP, pEn, aHP, aEn)
                l1 = legal_actions(pEn)
                l2 = legal_actions(aEn)
                h_legal = l1 if human_is_p1 else l2
                ai_legal = l2 if human_is_p1 else l1

                _print_state(pHP, pEn, aHP, aEn, rnd, human_is_p1)
                _print_skills(h_legal)
                h_action = _ask_skill(h_legal)

                ai_sg = solver.average_strategy(ai_idx, sid, ai_legal)
                ai_action = int(np.random.choice(NUM_SKILLS, p=ai_sg))

                a1 = h_action if human_is_p1 else ai_action
                a2 = ai_action if human_is_p1 else h_action

                # 记录出招前状态，供 CFR 反事实更新使用
                traj.append((sid, l1, l2, a1, a2, pHP, pEn, aHP, aEn))
                solver.phase_steps[energy_phase(pEn, aEn)] += 1
                solver.total_steps += 1

                pre_pHP, pre_aHP = pHP, aHP
                pHP, pEn, aHP, aEn, done, winner = game_step(pHP, pEn, aHP, aEn, a1, a2)
                _print_outcome(h_action, ai_action, pre_pHP, pHP, pre_aHP, aHP, human_is_p1)

                if not done:
                    npl = legal_actions(pEn)
                    nal = legal_actions(aEn)
                    pred_v = heuristic_value(pHP, pEn, aHP, aEn, npl, nal, solver.h_params)
                    h_records.append((pHP, pEn, aHP, aEn, nal, pred_v))

            # 克制链统计
            for _, _, _, a1_, a2_, _, pen_, _, aen_ in traj:
                for atk, dfn, lbl, en_atk, en_dfn in COUNTER_CHAIN:
                    if a1_ == atk and a2_ == dfn and pen_ >= en_atk and aen_ >= en_dfn:
                        solver.counter_hits[lbl] += 1
                    elif a2_ == atk and a1_ == dfn and aen_ >= en_atk and pen_ >= en_dfn:
                        solver.counter_hits[lbl] += 1

            # 结局显示
            if winner == 1:
                result = "人类获胜" if human_is_p1 else "AI获胜"
            elif winner == 2:
                result = "AI获胜" if human_is_p1 else "人类获胜"
            else:
                result = "平局"
            key = "人类" if "人类" in result else ("AI" if "AI" in result else "平局")
            wins[key] += 1
            tot = sum(wins.values())
            print(f"\n  结局: {result}")
            print(f"  战绩  人类 {wins['人类']}  AI {wins['AI']}  平 {wins['平局']}  (共{tot}局)")

            # ── CFR 在线更新（固定人类行动，更新 AI 方遗憾）──────
            if update_cfr and traj:
                final_v = terminal_value(winner, pHP, aHP)
                solver.episode_count += 1
                weight = float(solver.episode_count)
                T = len(traj)

                for t, (s, l1, l2, a1, a2, php, pen, ahp, aen) in enumerate(traj):
                    steps_to_end = T - t - 1
                    g = GAMMA ** steps_to_end if steps_to_end < BLEND_STEPS else 0.0

                    if human_is_p1:
                        # 人类固定 a1，更新 P2(AI) 的遗憾
                        sg2 = solver.current_strategy(1, s, l2)
                        cf2 = np.zeros(NUM_SKILLS)
                        for alt in l2:
                            nhp2, nen2, nahp2, naen2, d2, w2 = game_step(
                                php, pen, ahp, aen, a1, alt)
                            if d2:
                                cf2[alt] = -terminal_value(w2, nhp2, nahp2)
                            else:
                                npl = legal_actions(nen2)
                                nal = legal_actions(naen2)
                                imm = -heuristic_value(nhp2, nen2, nahp2, naen2, npl, nal,
                                                       solver.h_params)
                                cf2[alt] = (1 - g) * imm + g * (-final_v)
                        ev2 = float(sg2 @ cf2)
                        for alt in l2:
                            solver.R[1][s][alt] += cf2[alt] - ev2
                        solver.S[1][s] += weight * sg2
                    else:
                        # 人类固定 a2，更新 P1(AI) 的遗憾
                        sg1 = solver.current_strategy(0, s, l1)
                        cf1 = np.zeros(NUM_SKILLS)
                        for alt in l1:
                            cf1[alt] = step_cf_value(php, pen, ahp, aen, alt, a2, final_v, g,
                                                     solver.h_params)
                        ev1 = float(sg1 @ cf1)
                        for alt in l1:
                            solver.R[0][s][alt] += cf1[alt] - ev1
                        solver.S[0][s] += weight * sg1

                if h_records:
                    solver.h_params.update_episode(h_records, final_v)

                print(f"  [AI 已从本局更新策略，共 {solver.episode_count} 局经验]")
                save_strategy(solver, save_path)

    except KeyboardInterrupt:
        pass

    print(f"\n[退出] 人类 {wins['人类']}  AI {wins['AI']}  平 {wins['平局']}")
    return wins


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_saved()
    elif len(sys.argv) > 1 and sys.argv[1] == "dataset":
        solver = CFRSolver()
        load_strategy(solver)
        generate_cfr_dataset(solver)
    elif len(sys.argv) > 1 and sys.argv[1] == "human":
        # python cfr_selfplay_train.py human [1|2] [no_learn]
        solver = CFRSolver()
        load_strategy(solver, DEFAULT_SAVE_PATH)
        hp = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] in ("1", "2") else 1
        learn = not (len(sys.argv) > 3 and sys.argv[3] == "no_learn")
        human_vs_cfr(solver, human_player=hp, update_cfr=learn, save_path=DEFAULT_SAVE_PATH)
    else:
        train_cfr()
