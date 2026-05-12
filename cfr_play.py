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

SAVE_PATH = os.path.join(_BASE, 'cfr_output', 'cfr_strategy.npz')

from cfr_selfplay_train import CFRSolver, load_strategy, human_vs_cfr


def main():
    print("╔══════════════════════════════════════════╗")
    print("║       掌心战争  -  CFR AI 对战模式       ║")
    print("║  输入 1 你先手(P1)  输入 2 你后手(P2)    ║")
    print("╚══════════════════════════════════════════╝\n")

    if not os.path.exists(SAVE_PATH):
        print(f"  [错误] 未找到策略文件：{SAVE_PATH}")
        print("  请将 cfr_output 文件夹放在程序同目录下。")
        input("  按回车退出...")
        return

    solver = CFRSolver()
    load_strategy(solver, SAVE_PATH)

    game_idx = 0
    while True:
        game_idx += 1
        # 奇局 P1、偶局 P2，让两张策略表都能从人类对局更新
        hp = 1 if game_idx % 2 == 1 else 2
        print(f"\n  第 {game_idx} 局  你是 P{hp}（自动交替以均衡训练双侧策略）")
        human_vs_cfr(solver, human_player=hp, update_cfr=True, save_path=SAVE_PATH,
                     n_games=1)
        if input("\n  再来一局？(回车=是 / n=退出): ").strip().lower() == 'n':
            break
    input("\n  按回车退出...")


if __name__ == '__main__':
    main()
