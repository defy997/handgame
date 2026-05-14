"""掌心战争 CFR版 - 人机对战入口（打包专用）"""
import sys
import os

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 打包后 _BASE = exe 所在目录；开发时 = 脚本所在目录
if getattr(sys, 'frozen', False):
    _BASE = os.path.dirname(sys.executable)
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))

CFR_PATH  = os.path.join(_BASE, 'cfr_output', 'cfr_strategy.json')
NASH_PATH = os.path.join(_BASE, 'cfr_output', 'nash_exact.json')


def _print_stats(wins, mode):
    tot = sum(wins.values())
    if tot == 0:
        return
    print(f"\n  ── 总战绩 [{mode}] {'─'*28}")
    print(f"  人类 {wins['人类']}胜  AI {wins['AI']}胜  平局 {wins['平局']}  (共{tot}局)")
    print(f"  胜率  人类={wins['人类']/tot:.1%}  "
          f"AI={wins['AI']/tot:.1%}  平={wins['平局']/tot:.1%}")
    print(f"  {'─'*42}")


def main():
    print("╔══════════════════════════════════════════╗")
    print("║         掌心战争  -  AI 对战模式         ║")
    print("╠══════════════════════════════════════════╣")
    print("║  1. MC-CFR AI（自博弈训练，会从对局学习）║")
    print("║  2. Nash 精确均衡 AI（理论最优，不更新） ║")
    print("╚══════════════════════════════════════════╝\n")

    mode_input = input("  选择模式 (1/2，默认1): ").strip()
    use_nash = (mode_input == "2")
    mode_name = "Nash 精确均衡" if use_nash else "MC-CFR"

    if use_nash:
        if not os.path.exists(NASH_PATH):
            print(f"  [错误] 未找到 Nash 策略文件：{NASH_PATH}")
            print("  请先运行: python cfr_exact.py")
            input("  按回车退出..."); return
        from cfr_exact import load_nash, human_vs_nash
        _, S1, S2 = load_nash(NASH_PATH)
        solver = None
    else:
        if not os.path.exists(CFR_PATH):
            print(f"  [错误] 未找到策略文件：{CFR_PATH}")
            print("  请将 cfr_output 文件夹放在程序同目录下。")
            input("  按回车退出..."); return
        from cfr_selfplay_train import CFRSolver, load_strategy, human_vs_cfr
        solver = CFRSolver()
        load_strategy(solver, CFR_PATH)
        S1 = S2 = None

    wins = {"人类": 0, "AI": 0, "平局": 0}
    game_idx = 0

    print(f"\n  模式: {mode_name}")
    print(f"  双方自动交替先后手（保证双侧策略均等）\n")

    try:
        while True:
            game_idx += 1
            # 奇局人类P1，偶局人类P2，均衡覆盖双侧策略
            hp = 1 if game_idx % 2 == 1 else 2
            print(f"\n  ===== 第 {game_idx} 局  你是 P{hp} =====")

            if use_nash:
                result = human_vs_nash(S1, S2, human_player=hp, n_games=1)
            else:
                result = human_vs_cfr(solver, human_player=hp,
                                      update_cfr=True, save_path=CFR_PATH,
                                      n_games=1)
            for k, v in result.items():
                wins[k] += v

            _print_stats(wins, mode_name)

            if input("\n  再来一局？(回车=是 / n=退出): ").strip().lower() == "n":
                break
    except KeyboardInterrupt:
        pass

    _print_stats(wins, mode_name)
    input("\n  按回车退出...")


if __name__ == '__main__':
    main()
