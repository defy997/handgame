"""
精确 Nash 均衡求解器 - 掌心战争

零和随机博弈 (Stochastic Game) Nash 值迭代:
  - 每个状态建收益矩阵，用 LP 求该状态的 Nash 均衡
  - 值迭代直到收敛（gamma<1 保证压缩性，无需启发值、无需采样）
  - 收敛结果是真正的 Nash 均衡策略，exploit 理论上趋向 0

与 cfr_selfplay_train.py 的区别:
  MC-CFR + 启发值   → 求解"近似游戏"的 Nash，exploit 有下界
  本文件 (精确)     → 求解原始游戏的 Nash，exploit 趋向 0
"""

import numpy as np
import json, os, time
from scipy.optimize import linprog

from cfr_selfplay_train import (
    NUM_SKILLS, MAX_HP, MAX_ENERGY, SKILL_NAMES,
    game_step, legal_actions, terminal_value, _winner,
    INIT_HP, INIT_EN,
)

GAMMA     = 0.99
SAVE_PATH = "cfr_output/nash_exact.json"


# ── Nash 矩阵博弈求解 ─────────────────────────────────────────────────────

def solve_matrix_nash(A):
    """
    零和矩阵博弈 Nash 均衡。
    A[i,j] = P1 收益（P1 最大化，P2 最小化）。
    返回 (value, sigma1[r], sigma2[c])。
    """
    r, c = A.shape

    # 纯策略快速路径：maximin == minimax 时存在纯策略 Nash
    maximin = float(A.min(axis=1).max())
    minimax = float(A.max(axis=0).min())
    if abs(maximin - minimax) < 1e-10:
        s1 = np.zeros(r); s1[int(A.min(axis=1).argmax())] = 1.0
        s2 = np.zeros(c); s2[int(A.max(axis=0).argmin())] = 1.0
        return maximin, s1, s2

    # P1 LP: max v  s.t.  A^T σ1 ≥ v·1,  σ1 ≥ 0,  Σσ1 = 1
    obj1 = np.zeros(r + 1); obj1[-1] = -1.0
    Aub1 = np.zeros((c, r + 1)); Aub1[:, :r] = -A.T; Aub1[:, r] = 1.0
    Aeq1 = np.zeros((1, r + 1)); Aeq1[0, :r] = 1.0
    res1 = linprog(obj1, A_ub=Aub1, b_ub=np.zeros(c),
                   A_eq=Aeq1, b_eq=[1.0],
                   bounds=[(0, None)] * r + [(None, None)],
                   method='highs')

    # P2 LP: min v  s.t.  A σ2 ≤ v·1,  σ2 ≥ 0,  Σσ2 = 1
    obj2 = np.zeros(c + 1); obj2[-1] = 1.0
    Aub2 = np.zeros((r, c + 1)); Aub2[:, :c] = A; Aub2[:, c] = -1.0
    Aeq2 = np.zeros((1, c + 1)); Aeq2[0, :c] = 1.0
    res2 = linprog(obj2, A_ub=Aub2, b_ub=np.zeros(r),
                   A_eq=Aeq2, b_eq=[1.0],
                   bounds=[(0, None)] * c + [(None, None)],
                   method='highs')

    v  = float(-res1.fun) if res1.success else (maximin + minimax) / 2
    s1 = np.maximum(res1.x[:r], 0) if res1.success else np.ones(r) / r
    s2 = np.maximum(res2.x[:c], 0) if res2.success else np.ones(c) / c
    s1 /= max(s1.sum(), 1e-12)
    s2 /= max(s2.sum(), 1e-12)
    return v, s1, s2


# ── 值迭代 ────────────────────────────────────────────────────────────────

_NON_TERMINAL = [
    (php, pen, ahp, aen)
    for php in range(1, MAX_HP + 1)
    for pen in range(MAX_ENERGY + 1)
    for ahp in range(1, MAX_HP + 1)
    for aen in range(MAX_ENERGY + 1)
]


def nash_value_iter(gamma=GAMMA, max_iter=2000, tol=1e-7):
    """
    零和随机博弈精确 Nash 值迭代。

    每轮对所有非终局状态:
      1. 用当前 V 构造收益矩阵 A[a1,a2] = r(s,a1,a2) + γ·V[s']
      2. 求 Nash 均衡 → 更新 V[s] 和策略 S1/S2[s]
    迭代至 max|V_new - V_old| < tol。

    gamma < 1 保证映射是压缩映射（Banach 不动点定理），即使游戏有循环。
    """
    V  = np.zeros((MAX_HP+1, MAX_ENERGY+1, MAX_HP+1, MAX_ENERGY+1))
    S1 = np.zeros((MAX_HP+1, MAX_ENERGY+1, MAX_HP+1, MAX_ENERGY+1, NUM_SKILLS))
    S2 = np.zeros_like(S1)

    for php in range(MAX_HP + 1):
        for ahp in range(MAX_HP + 1):
            if php <= 0 or ahp <= 0:
                V[php, :, ahp, :] = terminal_value(_winner(php, ahp), php, ahp)

    print(f"\n{'='*56}")
    print(f"  精确 Nash 值迭代")
    print(f"  gamma={gamma}  tol={tol}  非终局状态={len(_NON_TERMINAL)}")
    print(f"{'='*56}")

    t_start = time.perf_counter()
    for it in range(1, max_iter + 1):
        delta = 0.0
        for php, pen, ahp, aen in _NON_TERMINAL:
            l1 = legal_actions(pen)
            l2 = legal_actions(aen)
            A  = np.empty((len(l1), len(l2)))
            for i, a1 in enumerate(l1):
                for j, a2 in enumerate(l2):
                    nhp, nen, nahp, naen, done, w = game_step(
                        php, pen, ahp, aen, a1, a2)
                    A[i, j] = (terminal_value(w, nhp, nahp) if done
                               else gamma * V[nhp, nen, nahp, naen])
            v, s1, s2 = solve_matrix_nash(A)
            delta = max(delta, abs(v - V[php, pen, ahp, aen]))
            V[php, pen, ahp, aen] = v
            S1[php, pen, ahp, aen, :] = 0.0
            S2[php, pen, ahp, aen, :] = 0.0
            for k, a in enumerate(l1): S1[php, pen, ahp, aen, a] = s1[k]
            for k, a in enumerate(l2): S2[php, pen, ahp, aen, a] = s2[k]

        elapsed = time.perf_counter() - t_start
        if it % 10 == 0 or delta < tol:
            print(f"  迭代 {it:4d}: Δ={delta:.2e}  {elapsed:.0f}s")
        if delta < tol:
            print(f"  [收敛] {it} 轮  总用时 {elapsed:.0f}s")
            break
    else:
        print(f"  [未收敛] {max_iter} 轮后 Δ={delta:.2e}")

    # ── 对称化：只修正 V，再从对称 V 重新推导策略 ────────────
    # 零和游戏满足 V[s] = -V[swap(s)]，直接平均策略会破坏 Nash 性质，
    # 正确做法：先对称化 V，再用对称 V 重跑一遍策略求解。
    print("  [对称化] 修正值函数...")
    for php, pen, ahp, aen in _NON_TERMINAL:
        v  = V[php, pen, ahp, aen]
        vs = V[ahp, aen, php, pen]
        avg = (v - vs) / 2
        V[php, pen, ahp, aen] = avg
        V[ahp, aen, php, pen] = -avg

    # 从对称 V 重新推导所有状态的 Nash 策略（只做一次 LP，不更新 V）
    for php, pen, ahp, aen in _NON_TERMINAL:
        l1 = legal_actions(pen)
        l2 = legal_actions(aen)
        A  = np.empty((len(l1), len(l2)))
        for i, a1 in enumerate(l1):
            for j, a2 in enumerate(l2):
                nhp, nen, nahp, naen, done, w = game_step(
                    php, pen, ahp, aen, a1, a2)
                A[i, j] = (terminal_value(w, nhp, nahp) if done
                           else gamma * V[nhp, nen, nahp, naen])
        _, s1, s2 = solve_matrix_nash(A)
        S1[php, pen, ahp, aen, :] = 0.0
        S2[php, pen, ahp, aen, :] = 0.0
        for k, a in enumerate(l1): S1[php, pen, ahp, aen, a] = s1[k]
        for k, a in enumerate(l2): S2[php, pen, ahp, aen, a] = s2[k]

    max_err = max(abs(V[php,pen,ahp,aen] + V[ahp,aen,php,pen])
                  for php,pen,ahp,aen in _NON_TERMINAL)
    print(f"  [对称化完成] V 最大残差={max_err:.2e}")

    return V, S1, S2


# ── 保存 / 加载 ───────────────────────────────────────────────────────────

def save_nash(V, S1, S2, path=SAVE_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {"V": V.tolist(), "S1": S1.tolist(), "S2": S2.tolist(),
            "gamma": GAMMA, "shape": list(V.shape)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[保存] Nash 策略 → {path}")


def load_nash(path=SAVE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    V  = np.array(d["V"])
    S1 = np.array(d["S1"])
    S2 = np.array(d["S2"])
    print(f"[加载] Nash 策略 ← {path}")
    return V, S1, S2


# ── 策略分析 ──────────────────────────────────────────────────────────────

def print_nash_strategy(S1, S2, pHP, pEn, aHP, aEn, label=""):
    l1 = legal_actions(pEn)
    l2 = legal_actions(aEn)
    tag = label or f"HP={pHP} En={pEn} vs HP={aHP} En={aEn}"
    print(f"  [{tag}]")
    s1 = S1[pHP, pEn, aHP, aEn]
    s2 = S2[pHP, pEn, aHP, aEn]
    p1_top = [(SKILL_NAMES[a], s1[a]) for a in l1 if s1[a] > 0.04]
    p2_top = [(SKILL_NAMES[a], s2[a]) for a in l2 if s2[a] > 0.04]
    p1_str = "  ".join(f"{n}={p:.0%}" for n, p in sorted(p1_top, key=lambda x: -x[1]))
    p2_str = "  ".join(f"{n}={p:.0%}" for n, p in sorted(p2_top, key=lambda x: -x[1]))
    print(f"    P1: {p1_str or '(均匀)'}")
    print(f"    P2: {p2_str or '(均匀)'}")


def analyze_nash(V, S1, S2, init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN)):
    print("\n[精确 Nash 均衡策略分析]")

    php, pen, ahp, aen = init
    v0 = V[php, pen, ahp, aen]
    print(f"\n  起始状态 Nash 均衡价值 (P1视角): {v0:+.4f}")
    print(f"  （接近0表示双方势均力敌）\n")

    states = [
        (*init,      f"起始 {init[0]}HP {init[1]}能(对称)"),
        (3, 2, 3, 2, "3HP 2能 对称"),
        (2, 2, 2, 2, "2HP 2能 对称"),
        (2, 1, 2, 2, "P1 1能 vs P2 2能"),
        (4, 3, 4, 3, "高能 3能 满血"),
        (1, 3, 3, 1, "P1濒死高能 vs P2健康低能"),
        (2, 0, 2, 2, "P1空城 P2有能量"),
        (php, 0, ahp, 0, "双方空城"),
    ]
    for s in states:
        print_nash_strategy(S1, S2, *s[:4], s[4])

    print("\n[各状态 Nash 价值（P1视角，正=P1优势）]")
    print(f"  起始 {init[0]}HP {init[1]}能: {V[init[0],init[1],init[2],init[3]]:+.4f}")
    print(f"  P1满血3能 vs P2满血3能: {V[4,3,4,3]:+.4f}")
    print(f"  P1 1血 vs P2 4血:       {V[1,2,4,2]:+.4f}")
    print(f"  P1 4血 vs P2 1血:       {V[4,2,1,2]:+.4f}")


# ── 可利用度评估 ──────────────────────────────────────────────────────────

def exploitability(V, S1, S2, init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN),
                   n_eval=500, gamma=GAMMA):
    """
    评估精确 Nash 策略的可利用度上界。
    BR 方对固定方的平均策略做贪心最佳响应，统计胜率。
    """
    from cfr_selfplay_train import overtime_winner, MAX_ROUNDS

    def _ev_alt(php, pen, ahp, aen, a1, a2):
        nhp, nen, nahp, naen, done, w = game_step(php, pen, ahp, aen, a1, a2)
        if done:
            return terminal_value(w, nhp, nahp)
        return gamma * V[nhp, nen, nahp, naen]

    def br_winrate(br_player):
        wins = 0
        for _ in range(n_eval):
            php, pen, ahp, aen = init
            done = False; rnd = 0
            while not done and rnd < MAX_ROUNDS:
                rnd += 1
                l1, l2 = legal_actions(pen), legal_actions(aen)
                if br_player == 2:
                    fixed_sg = S1[php, pen, ahp, aen, :]
                    best_a, best_v = l2[0], float('-inf')
                    for alt in l2:
                        ev = sum(fixed_sg[a1] * (-_ev_alt(php, pen, ahp, aen, a1, alt))
                                 for a1 in l1)
                        if ev > best_v:
                            best_v = ev; best_a = alt
                    fixed_a = int(np.random.choice(NUM_SKILLS, p=fixed_sg))
                    php, pen, ahp, aen, done, winner = game_step(
                        php, pen, ahp, aen, fixed_a, best_a)
                else:
                    fixed_sg = S2[php, pen, ahp, aen, :]
                    best_a, best_v = l1[0], float('-inf')
                    for alt in l1:
                        ev = sum(fixed_sg[a2] * _ev_alt(php, pen, ahp, aen, alt, a2)
                                 for a2 in l2)
                        if ev > best_v:
                            best_v = ev; best_a = alt
                    fixed_a = int(np.random.choice(NUM_SKILLS, p=fixed_sg))
                    php, pen, ahp, aen, done, winner = game_step(
                        php, pen, ahp, aen, best_a, fixed_a)
            if not done:
                winner = overtime_winner(php, ahp)
            if winner == br_player:
                wins += 1
        return wins / n_eval

    br1 = br_winrate(1)
    br2 = br_winrate(2)
    print(f"\n[可利用度]  BR_P1={br1:.1%}  BR_P2={br2:.1%}  "
          f"avg_exploit={(br1+br2)/2:.1%}")
    print(f"  （Nash 均衡理论值: 两者均应≈50%，平均≈50%）")
    return br1, br2


# ── 人机对战 ──────────────────────────────────────────────────────────────

def human_vs_nash(S1, S2, human_player=1,
                  init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN),
                  n_games=1):
    """
    Nash AI 人机对战。AI 使用精确 Nash 均衡混合策略，不做在线更新。
    返回 {"人类": wins, "AI": wins, "平局": draws}。
    """
    from cfr_selfplay_train import (
        MAX_ROUNDS, overtime_winner,
        _print_state, _print_skills, _ask_skill, _print_outcome,
    )

    human_is_p1 = (human_player == 1)
    wins = {"人类": 0, "AI": 0, "平局": 0}

    try:
        for game_idx in range(1, (n_games or 999999) + 1):
            php, pen, ahp, aen = init
            done = False; rnd = 0

            while not done and rnd < MAX_ROUNDS:
                rnd += 1
                l1 = legal_actions(pen)
                l2 = legal_actions(aen)
                h_legal  = l1 if human_is_p1 else l2
                ai_legal = l2 if human_is_p1 else l1

                # AI 使用 Nash 混合策略
                ai_sg = (S2[php, pen, ahp, aen] if human_is_p1
                         else S1[php, pen, ahp, aen]).copy()
                for a in range(NUM_SKILLS):
                    if a not in ai_legal:
                        ai_sg[a] = 0.0
                total = ai_sg.sum()
                if total < 1e-12:
                    ai_sg[ai_legal] = 1.0 / len(ai_legal)
                else:
                    ai_sg /= total
                ai_action = int(np.random.choice(NUM_SKILLS, p=ai_sg))

                _print_state(php, pen, ahp, aen, rnd, human_is_p1)
                _print_skills(h_legal)
                h_action = _ask_skill(h_legal)

                a1 = h_action if human_is_p1 else ai_action
                a2 = ai_action if human_is_p1 else h_action

                pre_php, pre_ahp = php, ahp
                php, pen, ahp, aen, done, winner = game_step(
                    php, pen, ahp, aen, a1, a2)
                _print_outcome(h_action, ai_action,
                               pre_php, php, pre_ahp, ahp, human_is_p1)

            if not done:
                winner = overtime_winner(php, ahp)

            if winner == 1:
                result = "人类获胜" if human_is_p1 else "AI获胜"
            elif winner == 2:
                result = "AI获胜" if human_is_p1 else "人类获胜"
            else:
                result = "平局"
            key = "人类" if "人类" in result else ("AI" if "AI" in result else "平局")
            wins[key] += 1
            print(f"\n  结局: {result}")

    except KeyboardInterrupt:
        pass

    return wins


# ── 主入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "solve"

    if cmd == "analyze":
        V, S1, S2 = load_nash()
        analyze_nash(V, S1, S2)
        exploitability(V, S1, S2)
    else:
        # python cfr_exact.py [solve]
        V, S1, S2 = nash_value_iter()
        save_nash(V, S1, S2)
        analyze_nash(V, S1, S2)
        exploitability(V, S1, S2)
