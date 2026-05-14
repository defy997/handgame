"""
将 Nash 精确均衡策略和 MC-CFR 策略导出为可读 txt 文件
"""
import os
import numpy as np

from cfr_selfplay_train import (
    legal_actions, SKILL_NAMES, MAX_HP, MAX_ENERGY, NUM_SKILLS,
    INIT_HP, INIT_EN,
)


def fmt_strategy(probs, legal, threshold=0.02):
    """将概率数组格式化为可读字符串，过滤低概率技能"""
    items = [(SKILL_NAMES[a], probs[a]) for a in legal if probs[a] >= threshold]
    items.sort(key=lambda x: -x[1])
    if not items:
        # fallback: 显示概率最高的
        a = legal[int(np.argmax([probs[x] for x in legal]))]
        return f"{SKILL_NAMES[a]}=100%"
    return "  ".join(f"{n}={p:.0%}" for n, p in items)


def export_nash(save_dir="cfr_output", out_path="cfr_output/nash_strategy.txt"):
    from cfr_exact import load_nash
    nash_path = os.path.join(save_dir, "nash_exact.json")
    if not os.path.exists(nash_path):
        print(f"[错误] 找不到 {nash_path}，请先运行 python cfr_exact.py")
        return

    _, S1, S2 = load_nash(nash_path)

    lines = []
    lines.append("=" * 64)
    lines.append("  Nash 精确均衡策略表 - 掌心战争")
    lines.append("  格式: P1视角 (我方HP/能量 vs 对方HP/能量)")
    lines.append("  纯策略=只有一个选项  混合策略=按概率随机出招")
    lines.append("=" * 64)

    pure_count = mixed_count = 0

    for php in range(1, MAX_HP + 1):
        for pen in range(MAX_ENERGY + 1):
            for ahp in range(1, MAX_HP + 1):
                for aen in range(MAX_ENERGY + 1):
                    l1 = legal_actions(pen)
                    l2 = legal_actions(aen)
                    s1 = S1[php, pen, ahp, aen]
                    s2 = S2[php, pen, ahp, aen]

                    tag = "纯" if s1[l1].max() > 0.99 else "混"
                    if tag == "纯":
                        pure_count += 1
                    else:
                        mixed_count += 1

                    header = (f"[{tag}] 我方 {php}HP {pen}能  vs  对方 {ahp}HP {aen}能")
                    p1_str = fmt_strategy(s1, l1)
                    p2_str = fmt_strategy(s2, l2)
                    lines.append(f"\n{header}")
                    lines.append(f"  P1策略: {p1_str}")
                    lines.append(f"  P2策略: {p2_str}")

    lines.append("\n" + "=" * 64)
    lines.append(f"  共 {pure_count + mixed_count} 个状态")
    lines.append(f"  纯策略: {pure_count}  混合策略: {mixed_count}")
    lines.append("=" * 64)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[保存] Nash 策略表 → {out_path}  ({pure_count+mixed_count} 个状态)")


def export_cfr(save_dir="cfr_output", out_path="cfr_output/cfr_strategy_table.txt"):
    from cfr_selfplay_train import CFRSolver, load_strategy
    cfr_path = os.path.join(save_dir, "cfr_strategy.json")
    if not os.path.exists(cfr_path):
        print(f"[错误] 找不到 {cfr_path}，请先运行训练")
        return

    solver = CFRSolver()
    load_strategy(solver, cfr_path)

    from cfr_selfplay_train import state_id
    lines = []
    lines.append("=" * 64)
    lines.append("  MC-CFR 平均策略表 - 掌心战争")
    lines.append(f"  训练局数: {solver.episode_count}  总步数: {solver.total_steps}")
    lines.append("  格式: P1视角 (我方HP/能量 vs 对方HP/能量)")
    lines.append("=" * 64)

    for php in range(1, MAX_HP + 1):
        for pen in range(MAX_ENERGY + 1):
            for ahp in range(1, MAX_HP + 1):
                for aen in range(MAX_ENERGY + 1):
                    l1 = legal_actions(pen)
                    l2 = legal_actions(aen)
                    sid = state_id(php, pen, ahp, aen)
                    s1 = solver.average_strategy(0, sid, l1)
                    s2 = solver.average_strategy(1, sid, l2)

                    tag = "纯" if s1[l1].max() > 0.99 else "混"
                    header = f"[{tag}] 我方 {php}HP {pen}能  vs  对方 {ahp}HP {aen}能"
                    p1_str = fmt_strategy(s1, l1)
                    p2_str = fmt_strategy(s2, l2)
                    lines.append(f"\n{header}")
                    lines.append(f"  P1策略: {p1_str}")
                    lines.append(f"  P2策略: {p2_str}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[保存] MC-CFR 策略表 → {out_path}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "both"

    if cmd in ("nash", "both"):
        export_nash()
    if cmd in ("cfr", "both"):
        export_cfr()
