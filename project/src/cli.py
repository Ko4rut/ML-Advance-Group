"""Terminal commands for inspecting the environment and running random games."""

import argparse
import time
from collections.abc import Sequence

from .agents import RandomAgent
from .environment import Move, Side, XiangqiEnv
from .environment.state import row_col
from .visualization.board import BoardRenderer


def format_move(move: Move) -> str:
    source_row, source_col = row_col(move.source)
    target_row, target_col = row_col(move.target)
    return f"({source_row},{source_col}) -> ({target_row},{target_col})"


def show_initial() -> int:
    env = XiangqiEnv()
    print(BoardRenderer().render(env.state))
    print(f"Legal moves: {len(env.get_legal_actions())}")
    return 0


def play_random(seed: int, max_plies: int, delay: float) -> int:
    env = XiangqiEnv()
    agents = {
        Side.RED: RandomAgent(seed),
        Side.BLACK: RandomAgent(seed + 1),
    }
    renderer = BoardRenderer()

    print(renderer.render(env.state))
    while not env.is_terminal() and env.state.ply_count < max_plies:
        side = env.state.side_to_move
        decision = agents[side].select_action(env.state)
        result = env.step(decision.move)
        print(f"\nPly {env.state.ply_count}: {side.name} {format_move(decision.move)}")
        print(renderer.render(result.state))
        if delay:
            time.sleep(delay)

    result = env.engine.outcome(env.state)
    if result is None:
        print(f"\nStopped after {max_plies} plies (truncated).")
    else:
        winner = "DRAW" if result.winner is None else result.winner.name
        print(f"\nGame over: {result.reason}; winner: {winner}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and run the Xiangqi environment")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show", help="render the initial state")

    random_parser = commands.add_parser("random", help="watch two random agents play")
    random_parser.add_argument("--seed", type=int, default=42)
    random_parser.add_argument("--max-plies", type=int, default=20)
    random_parser.add_argument("--delay", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "show":
        return show_initial()
    if args.max_plies < 1:
        raise SystemExit("--max-plies must be at least 1")
    if args.delay < 0:
        raise SystemExit("--delay cannot be negative")
    return play_random(args.seed, args.max_plies, args.delay)
