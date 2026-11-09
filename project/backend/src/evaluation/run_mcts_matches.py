"""Play MCTS against the other agents and export the results to CSV.

From project/backend:
    python -m src.evaluation.run_mcts_matches --games 4 --time-limit 1.0

Each game alternates the MCTS colour. Both sides get the same per-move
time limit. Reading the YAML config needs PyYAML (pip install pyyaml).
"""

import argparse
import csv
from pathlib import Path

from src.agents import AlphaBetaAgent, HeuristicAgent, MCTSAgent, RandomAgent, SearchBudget
from src.environment import Side, XiangqiEnv
from src.evaluation.match import MatchRunner, avg_stats

OPPONENTS = ("random", "heuristic", "alphabeta")
CSV_COLUMNS = [
    "opponent",
    "mcts_color",
    "result",
    "plies",
    "avg_simulations_per_move",
    "avg_time_per_move",
]


def load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML is required to read the config: pip install pyyaml")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_opponent(name: str, engine, time_limit: float, seed: int):
    if name == "random":
        return RandomAgent(seed)
    if name == "heuristic":
        return HeuristicAgent(engine, time_limit=time_limit)
    if name == "alphabeta":
        return AlphaBetaAgent(engine, max_depth=2)
    raise ValueError(f"unknown opponent: {name}")


def run_matches(args: argparse.Namespace) -> list[dict]:
    config = load_config(args.config)
    time_limit = args.time_limit if args.time_limit is not None else config["time_limit"]
    env = XiangqiEnv()
    engine = env.engine
    runner = MatchRunner(
        env,
        max_plies=args.max_plies,
        budget=SearchBudget(time_limit=time_limit),
    )

    rows = []
    for name in args.opponents.split(","):
        for game in range(args.games):
            game_seed = args.seed + game
            mcts = MCTSAgent.from_config(engine, {**config, "seed": game_seed})
            opponent = make_opponent(name, engine, time_limit, game_seed)
            # Even games: MCTS plays Red; odd games: MCTS plays Black.
            mcts_side = Side.RED if game % 2 == 0 else Side.BLACK
            if mcts_side is Side.RED:
                result = runner.play(mcts, opponent)
            else:
                result = runner.play(opponent, mcts)

            if result.winner is None:
                outcome = "draw"
            elif result.winner is mcts_side:
                outcome = "win"
            else:
                outcome = "loss"
            avg_simulations, avg_time = avg_stats(result, mcts_side)
            color = "red" if mcts_side is Side.RED else "black"
            print(
                f"{name} game {game + 1}/{args.games}: MCTS={color} {outcome} "
                f"({result.reason}) plies={result.plies}",
                flush=True,
            )
            rows.append({
                "opponent": name,
                "mcts_color": color,
                "result": outcome,
                "plies": result.plies,
                "avg_simulations_per_move": round(avg_simulations, 2),
                "avg_time_per_move": round(avg_time, 4),
            })
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print("\nopponent    W  D  L")
    for name in dict.fromkeys(row["opponent"] for row in rows):
        results = [row["result"] for row in rows if row["opponent"] == name]
        print(
            f"{name:<10}{results.count('win'):>3}{results.count('draw'):>3}"
            f"{results.count('loss'):>3}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Play MCTS against other Xiangqi agents")
    parser.add_argument("--config", default="configs/mcts.yaml")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--opponents", default=",".join(OPPONENTS))
    parser.add_argument("--out", default="results/mcts_matches.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = run_matches(args)
    write_csv(rows, args.out)
    print_summary(rows)
    print(f"\nSaved {len(rows)} games to {args.out}")


if __name__ == "__main__":
    main()
