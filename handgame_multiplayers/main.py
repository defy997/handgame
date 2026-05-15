from game import Game
import argparse
import os
import re
import heros.basic
import heros.hunter


def _parse_ai_indices(text: str, max_players: int) -> list[int]:
    indices: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= max_players:
            raise ValueError(f"Invalid ai index: {idx}")
        indices.append(idx)
    return indices

def main() -> None:
    parser = argparse.ArgumentParser(description="Console battle")
    parser.add_argument("--ai", type=str, default="", help="AI player indices, e.g. 0,1")
    parser.add_argument("--ai-model", type=str, default="", help="AI model path")
    args = parser.parse_args()

    print("双人控制台对战：手牌小游戏")
    print("按提示选择行动，两个玩家轮流在同一终端操作。")
    print()

    while True:
        verbose_choice = input("是否启用详细输出模式？(y/n，默认n): ").strip().lower()
        if verbose_choice in ["y", "yes"]:
            verbose = True
            print("已启用详细输出模式。")
            break
        if verbose_choice in ["n", "no", ""]:
            verbose = False
            print("已禁用详细输出模式。")
            break
        print("请输入 y 或 n。")

    p1 = heros.basic.BasicHero("玩家1")
    p2 = heros.basic.BasicHero("玩家2")
    # p3 = heros.hunter.Hunter("玩家3")
    game = Game(p1, p2, verbose=verbose)

    context = f"{p1.__class__.__name__.lower()}_vs_{p2.__class__.__name__.lower()}"
    model_dir = os.path.join("ai", context)

    def find_latest_model(path: str, context_tag: str) -> str:
        if not os.path.isdir(path):
            return ""
        pattern = re.compile(rf"^{re.escape(context_tag)}_(\d{{8}})_model\.pt$")
        candidates: list[tuple[str, str]] = []
        for name in os.listdir(path):
            match = pattern.match(name)
            if match:
                candidates.append((match.group(1), name))
        if not candidates:
            return ""
        candidates.sort()
        return os.path.join(path, candidates[-1][1])

    for idx, player in enumerate(game.players):
        while True:
            answer = input(f"玩家{idx + 1}({player.name}) 是否使用 AI？(y/n): ").strip().lower()
            if answer in ("y", "yes"):
                game.set_player_ai(idx, enabled=True)
                model_path = args.ai_model or find_latest_model(model_dir, context)
                if model_path:
                    game.load_player_ai_model(idx, model_path)
                    print(f"已加载模型: {model_path}")
                else:
                    print("未找到匹配模型，将使用默认参数。")
                break
            if answer in ("n", "no"):
                break
            print("请输入 y 或 n。")

    game.run()


if __name__ == "__main__":
    main()
