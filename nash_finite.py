"""
有限视野 Nash 均衡求解器 - 掌心战争

正确建模 30 回合上限与超时血量胜负规则。

有限视野反向归纳 (Backward Induction):
  状态: (pHP, pEn, aHP, aEn, t)
    t = 已打回合数 ∈ {0, …, MAX_ROUNDS}
    t=0:          游戏刚开始（第 1 回合前）
    t=MAX_ROUNDS: 30 回合结束，overtime_winner 结算

  S1/S2[php, pen, ahp, aen, t] = 第 t+1 回合的出招概率分布
  从 t=MAX_ROUNDS-1 反推到 t=0，单次反向扫描，无需收敛迭代。
"""

import numpy as np
import os, time

from cfr_selfplay_train import (
    NUM_SKILLS, MAX_HP, MAX_ENERGY, SKILL_NAMES,
    game_step, legal_actions, terminal_value, _winner,
    overtime_winner, MAX_ROUNDS,
    INIT_HP, INIT_EN,
)
from cfr_exact import solve_matrix_nash

SAVE_PATH = "cfr_output/nash_finite.npz"

_NON_TERMINAL = [
    (php, pen, ahp, aen)
    for php in range(1, MAX_HP + 1)
    for pen in range(MAX_ENERGY + 1)
    for ahp in range(1, MAX_HP + 1)
    for aen in range(MAX_ENERGY + 1)
]


# ── 反向归纳 ──────────────────────────────────────────────────────────────

def nash_finite_backward():
    """
    有限视野 Nash 反向归纳。

    返回 (V, S1, S2):
      V  形状: (MAX_HP+1, MAX_ENERGY+1, MAX_HP+1, MAX_ENERGY+1, MAX_ROUNDS+1)
      S1/S2 形状: 同 V + (NUM_SKILLS,)
               [php, pen, ahp, aen, t] = 第 t+1 回合的出招概率（t 为已打回合数）
    """
    shape5 = (MAX_HP + 1, MAX_ENERGY + 1, MAX_HP + 1, MAX_ENERGY + 1, MAX_ROUNDS + 1)
    shape6 = shape5 + (NUM_SKILLS,)
    V  = np.zeros(shape5)
    S1 = np.zeros(shape6)
    S2 = np.zeros(shape6)

    # 已终局状态（HP≤0）在任意回合价值固定
    for php in range(MAX_HP + 1):
        for ahp in range(MAX_HP + 1):
            if php <= 0 or ahp <= 0:
                V[php, :, ahp, :, :] = terminal_value(_winner(php, ahp), php, ahp)

    # t=MAX_ROUNDS：非终局状态用超时血量胜负结算
    for php, pen, ahp, aen in _NON_TERMINAL:
        w = overtime_winner(php, ahp)
        V[php, pen, ahp, aen, MAX_ROUNDS] = terminal_value(w, php, ahp)

    n = len(_NON_TERMINAL)
    print(f"\n{'='*56}")
    print(f"  有限视野 Nash 反向归纳")
    print(f"  MAX_ROUNDS={MAX_ROUNDS}  非终局状态={n}")
    print(f"  共 {MAX_ROUNDS} × {n} = {MAX_ROUNDS * n} 次 LP 求解")
    print(f"{'='*56}")

    t0 = time.perf_counter()
    for t in range(MAX_ROUNDS - 1, -1, -1):
        for php, pen, ahp, aen in _NON_TERMINAL:
            l1 = legal_actions(pen)
            l2 = legal_actions(aen)
            A  = np.empty((len(l1), len(l2)))
            for i, a1 in enumerate(l1):
                for j, a2 in enumerate(l2):
                    nhp, nen, nahp, naen, done, w = game_step(php, pen, ahp, aen, a1, a2)
                    A[i, j] = (terminal_value(w, nhp, nahp) if done
                               else V[nhp, nen, nahp, naen, t + 1])
            v, s1, s2 = solve_matrix_nash(A)
            V[php, pen, ahp, aen, t] = v
            S1[php, pen, ahp, aen, t, :] = 0.0
            S2[php, pen, ahp, aen, t, :] = 0.0
            for k, a in enumerate(l1): S1[php, pen, ahp, aen, t, a] = s1[k]
            for k, a in enumerate(l2): S2[php, pen, ahp, aen, t, a] = s2[k]

        done_rounds = MAX_ROUNDS - t
        if done_rounds % 5 == 0 or t == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / done_rounds * t if t > 0 else 0
            print(f"  t={t:2d} 完成  ({done_rounds:2d}/{MAX_ROUNDS})  {elapsed:.0f}s"
                  + (f"  ETA≈{eta:.0f}s" if eta > 2 else ""))

    elapsed = time.perf_counter() - t0
    print(f"  [完成] 总用时 {elapsed:.0f}s")

    # 对称化：只修正 V，再从对称 V 重新推导策略
    print("  [对称化] 修正值函数...")
    for t in range(MAX_ROUNDS + 1):
        for php, pen, ahp, aen in _NON_TERMINAL:
            v  = V[php, pen, ahp, aen, t]
            vs = V[ahp, aen, php, pen, t]
            avg = (v - vs) / 2
            V[php, pen, ahp, aen, t] = avg
            V[ahp, aen, php, pen, t] = -avg

    for t in range(MAX_ROUNDS):
        for php, pen, ahp, aen in _NON_TERMINAL:
            l1 = legal_actions(pen)
            l2 = legal_actions(aen)
            A  = np.empty((len(l1), len(l2)))
            for i, a1 in enumerate(l1):
                for j, a2 in enumerate(l2):
                    nhp, nen, nahp, naen, done, w = game_step(php, pen, ahp, aen, a1, a2)
                    A[i, j] = (terminal_value(w, nhp, nahp) if done
                               else V[nhp, nen, nahp, naen, t + 1])
            _, s1, s2 = solve_matrix_nash(A)
            S1[php, pen, ahp, aen, t, :] = 0.0
            S2[php, pen, ahp, aen, t, :] = 0.0
            for k, a in enumerate(l1): S1[php, pen, ahp, aen, t, a] = s1[k]
            for k, a in enumerate(l2): S2[php, pen, ahp, aen, t, a] = s2[k]

    max_err = max(
        abs(V[php, pen, ahp, aen, t] + V[ahp, aen, php, pen, t])
        for t in range(MAX_ROUNDS + 1)
        for php, pen, ahp, aen in _NON_TERMINAL
    )
    print(f"  [对称化完成] V 最大残差={max_err:.2e}")

    return V, S1, S2


# ── 保存 / 加载 ───────────────────────────────────────────────────────────

def save_nash_finite(V, S1, S2, path=SAVE_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, V=V, S1=S1, S2=S2)
    mb = os.path.getsize(path) / 1e6
    print(f"[保存] 有限视野 Nash 策略 → {path}  ({mb:.1f} MB)")


def load_nash_finite(path=SAVE_PATH):
    d = np.load(path)
    V, S1, S2 = d['V'], d['S1'], d['S2']
    print(f"[加载] 有限视野 Nash 策略 ← {path}")
    return V, S1, S2


# ── 策略分析 ──────────────────────────────────────────────────────────────

def _print_strategy_at(S1, S2, php, pen, ahp, aen, t, label=""):
    l1 = legal_actions(pen)
    l2 = legal_actions(aen)
    tag = label or f"HP={php} En={pen} vs HP={ahp} En={aen}"
    print(f"  [{tag}]")
    s1 = S1[php, pen, ahp, aen, t]
    s2 = S2[php, pen, ahp, aen, t]
    p1_top = [(SKILL_NAMES[a], s1[a]) for a in l1 if s1[a] > 0.04]
    p2_top = [(SKILL_NAMES[a], s2[a]) for a in l2 if s2[a] > 0.04]
    p1_str = "  ".join(f"{n}={p:.0%}" for n, p in sorted(p1_top, key=lambda x: -x[1]))
    p2_str = "  ".join(f"{n}={p:.0%}" for n, p in sorted(p2_top, key=lambda x: -x[1]))
    print(f"    P1: {p1_str or '(均匀)'}")
    print(f"    P2: {p2_str or '(均匀)'}")


def analyze_nash_finite(V, S1, S2, init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN)):
    print("\n[有限视野 Nash 均衡策略分析]")

    php, pen, ahp, aen = init
    v0 = V[php, pen, ahp, aen, 0]
    print(f"\n  起始状态 Nash 均衡价值 (P1视角, t=0): {v0:+.4f}")
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

    print("  ── 第 1 回合策略（t=0）──")
    for s in states:
        _print_strategy_at(S1, S2, *s[:4], 0, s[4])

    print("\n  ── 第 25 回合策略（t=24，接近时间到）──")
    for s in states:
        _print_strategy_at(S1, S2, *s[:4], 24, s[4])

    print("\n[各状态 Nash 价值（P1视角，正=P1优势）]")
    for t in [0, 15, 25, 28]:
        print(f"  t={t}:")
        print(f"    起始 {init[0]}HP {init[1]}能: {V[init[0],init[1],init[2],init[3],t]:+.4f}")
        print(f"    P1满血3能 vs P2满血3能:  {V[4,3,4,3,t]:+.4f}")
        print(f"    P1 1血 vs P2 4血:        {V[1,2,4,2,t]:+.4f}")
        print(f"    P1 3血 vs P2 2血:        {V[3,2,2,2,t]:+.4f}")


# ── 可利用度 ──────────────────────────────────────────────────────────────

def exploitability_finite(V, S1, S2,
                           init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN),
                           n_eval=500):
    def _ev(php, pen, ahp, aen, a1, a2, t):
        nhp, nen, nahp, naen, done, w = game_step(php, pen, ahp, aen, a1, a2)
        return terminal_value(w, nhp, nahp) if done else V[nhp, nen, nahp, naen, t + 1]

    def _make_sg(raw, legal):
        sg = np.zeros(NUM_SKILLS)
        for a in legal:
            sg[a] = raw[a]
        total = sg.sum()
        if total < 1e-12:
            for a in legal: sg[a] = 1.0 / len(legal)
        else:
            sg /= total
        return sg

    def br_winrate(br_player):
        wins = 0
        for _ in range(n_eval):
            php, pen, ahp, aen = init
            done = False; t = 0; winner = 0
            while not done and t < MAX_ROUNDS:
                l1, l2 = legal_actions(pen), legal_actions(aen)
                if br_player == 2:
                    sg1 = _make_sg(S1[php, pen, ahp, aen, t], l1)
                    best_a2, best_v2 = l2[0], float('-inf')
                    for a2 in l2:
                        ev2 = sum(sg1[a1] * (-_ev(php, pen, ahp, aen, a1, a2, t))
                                  for a1 in l1)
                        if ev2 > best_v2:
                            best_v2 = ev2; best_a2 = a2
                    a1 = int(np.random.choice(NUM_SKILLS, p=sg1))
                    a2 = best_a2
                else:
                    sg2 = _make_sg(S2[php, pen, ahp, aen, t], l2)
                    best_a1, best_v1 = l1[0], float('-inf')
                    for a1 in l1:
                        ev1 = sum(sg2[a2] * _ev(php, pen, ahp, aen, a1, a2, t)
                                  for a2 in l2)
                        if ev1 > best_v1:
                            best_v1 = ev1; best_a1 = a1
                    a1 = best_a1
                    a2 = int(np.random.choice(NUM_SKILLS, p=sg2))
                php, pen, ahp, aen, done, winner = game_step(php, pen, ahp, aen, a1, a2)
                t += 1
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


# ── 主入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "solve"

    if cmd == "analyze":
        V, S1, S2 = load_nash_finite()
        analyze_nash_finite(V, S1, S2)
        exploitability_finite(V, S1, S2)
    else:
        V, S1, S2 = nash_finite_backward()
        save_nash_finite(V, S1, S2)
        analyze_nash_finite(V, S1, S2)
        exploitability_finite(V, S1, S2)
