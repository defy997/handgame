"""掌心战争 Nash AI 对战 - 打包专用入口（完全自包含，无 torch/scipy 依赖）

优先使用有限视野 Nash (nash_finite.npz)；若不存在则回退到精确 Nash (nash_exact.json)。
"""
import sys
import os
import json
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

if getattr(sys, 'frozen', False):
    _BASE = os.path.dirname(sys.executable)
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))

_FINITE_PATH = os.path.join(_BASE, 'cfr_output', 'nash_finite.npz')
_EXACT_PATH  = os.path.join(_BASE, 'cfr_output', 'nash_exact.json')


# ── 游戏常量（内联自 dqn_train.py，无 torch 依赖）────────────────────────

NUM_SKILLS = 9
MAX_HP     = 4
MAX_ENERGY = 6
MAX_ROUNDS = 30
INIT_HP    = 4
INIT_EN    = 2

SKILL_NAMES = ['单枪', '双枪', '三枪', '大招', '小防', '大防', '能量', '反弹', '清零']
SKILL_COST  = np.array([1, 2, 2, 3, 0, 1, 0, 1, 2], dtype=np.int32)

HP_MATRIX = np.array([
    [  0,  -1,  -1,   0,   0,   0,  -1,   0,  -2],
    [  0,   0,   0,  -2,   0,   0,  -2,   0,  -3],
    [  0,  -1,   0,  -2,  -1,   0,  -3,  -1,   1],
    [ -3,   0,   0,   0,  -3,   0,  -3,  -3,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],
    [ -1,  -2,   0,   0,   0,   0,   0,   0,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],
], dtype=np.int32)

HP_MATRIX_empty = np.array([
    [ -2,  -3,  -3,  -1,   0,   0,  -1,   0,  -2],
    [ -1,  -2,  -1,  -4,   0,   0,  -2,   0,  -3],
    [ -1,  -3,  -2,  -4,  -1,   0,  -3,  -1,   1],
    [ -5,  -1,  -1,  -2,  -3,   0,  -3,  -3,   1],
    [ -2,  -2,  -1,  -1,   0,   0,   0,   0,   1],
    [ -2,  -2,  -2,  -2,   0,   0,   0,   0,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],
    [ -3,  -4,  -1,  -1,   0,   0,   0,   0,   1],
    [ -1,  -1,  -2,  -2,   0,   0,   0,   0,   1],
], dtype=np.int32)

ZERO_TABLE = np.array([
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1],
], dtype=np.int32)


# ── 游戏逻辑 ─────────────────────────────────────────────────────────────

def legal_actions(en):
    acts = [s for s in range(NUM_SKILLS) if SKILL_COST[s] <= en]
    return acts if acts else [6]


def _winner(pHP, aHP):
    if pHP <= 0 and aHP <= 0: return 3
    if aHP <= 0: return 1
    return 2


def overtime_winner(pHP, aHP):
    if pHP > aHP: return 1
    if aHP > pHP: return 2
    return 3


def game_step(pHP, pEn, aHP, aEn, pSk, aSk):
    pEn2 = min(MAX_ENERGY, max(0, pEn - int(SKILL_COST[pSk])) + (1 if pSk == 6 else 0))
    aEn2 = min(MAX_ENERGY, max(0, aEn - int(SKILL_COST[aSk])) + (1 if aSk == 6 else 0))

    if ZERO_TABLE[pSk][aSk]: pEn2 = aEn2 = 0
    if ZERO_TABLE[aSk][pSk]: pEn2 = aEn2 = 0

    p_dmg  = int(HP_MATRIX[pSk][aSk])
    ai_dmg = int(HP_MATRIX[aSk][pSk])

    aHP2 = max(0, min(MAX_HP, aHP + p_dmg))
    pHP2 = max(0, min(MAX_HP, pHP + ai_dmg))
    if pHP2 <= 0 or aHP2 <= 0:
        return pHP2, pEn2, aHP2, aEn2, True, _winner(pHP2, aHP2)

    if aEn2 == 0 and pSk in (0, 1, 2, 3) and aHP + p_dmg > 0:
        ai_dmg = int(HP_MATRIX_empty[aSk][pSk])
    if pEn2 == 0 and aSk in (0, 1, 2, 3) and pHP + ai_dmg > 0:
        p_dmg  = int(HP_MATRIX_empty[pSk][aSk])

    aHP2 = max(0, min(MAX_HP, aHP + p_dmg))
    pHP2 = max(0, min(MAX_HP, pHP + ai_dmg))

    done = pHP2 <= 0 or aHP2 <= 0
    return pHP2, pEn2, aHP2, aEn2, done, _winner(pHP2, aHP2) if done else 0


# ── UI ───────────────────────────────────────────────────────────────────

def _print_state(pHP, pEn, aHP, aEn, rnd, human_is_p1):
    h_hp, h_en  = (pHP, pEn) if human_is_p1 else (aHP, aEn)
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
            if raw.lower() == 'q':
                raise KeyboardInterrupt
            sk = int(raw)
            if sk in legal:
                return sk
            print(f"  无效，可用编号: {legal}")
        except ValueError:
            print("  请输入整数编号")


def _print_outcome(h_sk, ai_sk, pre_pHP, pHP, pre_aHP, aHP, human_is_p1):
    h_pre,  h_post  = (pre_pHP, pHP) if human_is_p1 else (pre_aHP, aHP)
    ai_pre, ai_post = (pre_aHP, aHP) if human_is_p1 else (pre_pHP, pHP)
    print(f"\n  你出: {SKILL_NAMES[h_sk]:<4}  |  AI出: {SKILL_NAMES[ai_sk]}")
    h_d  = h_post  - h_pre
    ai_d = ai_post - ai_pre
    if h_d  < 0: print(f"  你  -{-h_d} HP  (剩{h_post}/{MAX_HP})")
    elif h_d  > 0: print(f"  你  +{h_d} HP  (剩{h_post}/{MAX_HP})")
    if ai_d < 0: print(f"  AI  -{-ai_d} HP  (剩{ai_post}/{MAX_HP})")
    elif ai_d > 0: print(f"  AI  +{ai_d} HP  (剩{ai_post}/{MAX_HP})")
    if h_d == 0 and ai_d == 0:
        print(f"  双方无伤")


# ── 策略加载 ─────────────────────────────────────────────────────────────

def load_strategy():
    if os.path.exists(_FINITE_PATH):
        d = np.load(_FINITE_PATH)
        print(f"  [有限视野 Nash] 策略加载完毕（正确建模 {MAX_ROUNDS} 回合上限）")
        return d['S1'], d['S2'], True

    if os.path.exists(_EXACT_PATH):
        with open(_EXACT_PATH, 'r', encoding='utf-8') as f:
            d = json.load(f)
        print(f"  [精确 Nash] 策略加载完毕（折现无限游戏近似）")
        return np.array(d['S1']), np.array(d['S2']), False

    return None, None, None


# ── 对战逻辑 ─────────────────────────────────────────────────────────────

def play_one_game(S1, S2, finite, human_player, wins):
    human_is_p1 = (human_player == 1)
    php, pen, ahp, aen = INIT_HP, INIT_EN, INIT_HP, INIT_EN
    done = False
    rnd  = 0

    while not done and rnd < MAX_ROUNDS:
        t = rnd   # 已打回合数（有限 Nash 策略索引）
        rnd += 1

        l1 = legal_actions(pen)
        l2 = legal_actions(aen)
        h_legal  = l1 if human_is_p1 else l2
        ai_legal = l2 if human_is_p1 else l1

        # AI 先决定（双盲）
        if finite:
            ai_raw = (S2[php, pen, ahp, aen, t] if human_is_p1
                      else S1[php, pen, ahp, aen, t]).copy()
        else:
            ai_raw = (S2[php, pen, ahp, aen] if human_is_p1
                      else S1[php, pen, ahp, aen]).copy()

        ai_sg = np.zeros(NUM_SKILLS)
        for a in ai_legal:
            ai_sg[a] = ai_raw[a]
        total = ai_sg.sum()
        if total < 1e-12:
            for a in ai_legal: ai_sg[a] = 1.0 / len(ai_legal)
        else:
            ai_sg /= total
        ai_action = int(np.random.choice(NUM_SKILLS, p=ai_sg))

        _print_state(php, pen, ahp, aen, rnd, human_is_p1)
        _print_skills(h_legal)
        h_action = _ask_skill(h_legal)

        a1 = h_action if human_is_p1 else ai_action
        a2 = ai_action if human_is_p1 else h_action

        pre_php, pre_ahp = php, ahp
        php, pen, ahp, aen, done, winner = game_step(php, pen, ahp, aen, a1, a2)
        _print_outcome(h_action, ai_action, pre_php, php, pre_ahp, ahp, human_is_p1)

    if not done:
        winner = overtime_winner(php, ahp)
        print(f"\n  [超时] {MAX_ROUNDS} 回合到！血量：你={php if human_is_p1 else ahp}  AI={ahp if human_is_p1 else php}")

    if winner == 1:
        result = "人类获胜" if human_is_p1 else "AI获胜"
    elif winner == 2:
        result = "AI获胜" if human_is_p1 else "人类获胜"
    else:
        result = "平局"

    key = "人类" if "人类" in result else ("AI" if "AI" in result else "平局")
    wins[key] += 1
    print(f"\n  结局: {result}")


def print_stats(wins):
    tot = sum(wins.values())
    if tot == 0:
        return
    print(f"\n  ── 总战绩 {'─'*34}")
    print(f"  人类 {wins['人类']}胜  AI {wins['AI']}胜  平局 {wins['平局']}  (共{tot}局)")
    print(f"  胜率  人类={wins['人类']/tot:.1%}  "
          f"AI={wins['AI']/tot:.1%}  平={wins['平局']/tot:.1%}")
    print(f"  {'─'*42}")


# ── 主入口 ───────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════╗")
    print("║    掌心战争  -  Nash 精确均衡 AI 对战    ║")
    print("╚══════════════════════════════════════════╝\n")

    S1, S2, finite = load_strategy()
    if S1 is None:
        print(f"  [错误] 未找到策略文件。")
        print(f"  请将 cfr_output 文件夹放在程序同目录下，")
        print(f"  其中包含 nash_finite.npz 或 nash_exact.json。")
        input("  按回车退出...")
        return

    wins = {"人类": 0, "AI": 0, "平局": 0}
    game_idx = 0

    try:
        while True:
            game_idx += 1
            hp = 1 if game_idx % 2 == 1 else 2
            print(f"\n  ===== 第 {game_idx} 局  你是 P{hp} =====")
            play_one_game(S1, S2, finite, hp, wins)
            print_stats(wins)
            if input("\n  再来一局？(回车=是 / n=退出): ").strip().lower() == 'n':
                break
    except KeyboardInterrupt:
        pass

    print_stats(wins)
    input("\n  按回车退出...")


if __name__ == '__main__':
    main()
