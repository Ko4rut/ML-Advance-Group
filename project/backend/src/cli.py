"""Terminal commands for inspecting the environment and generating Xiangqi games."""

import argparse
import json
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

    batch_parser = commands.add_parser(
        "batch",
        help="generate agent-vs-agent games into a JSONL dataset without API or UI",
    )
    agent_names = ("random", "rules", "minimax", "alphabeta", "astar")
    batch_parser.add_argument("--red", choices=agent_names, default="alphabeta")
    batch_parser.add_argument("--black", choices=agent_names, default="alphabeta")
    batch_parser.add_argument("--games", type=int, default=10)
    batch_parser.add_argument("--red-depth", type=int, default=2)
    batch_parser.add_argument("--black-depth", type=int, default=2)
    batch_parser.add_argument("--max-plies", type=int, default=160)
    batch_parser.add_argument(
        "--opening-plies",
        type=int,
        default=0,
        help="number of uniformly random legal opening plies; labelled separately",
    )
    batch_parser.add_argument(
        "--no-progress-limit",
        type=int,
        default=0,
        help="optional early truncation after N plies without capture (0 disables)",
    )
    batch_parser.add_argument("--seed", type=int, default=42)
    batch_parser.add_argument("--output", required=True, help="new JSONL file path")
    batch_parser.add_argument("--progress-every", type=int, default=1)
    batch_parser.add_argument("--workers", type=int, default=1, help="parallel games; one writer")
    batch_parser.add_argument(
        "--resume", action="store_true",
        help="resume a fixed JSONL batch, validating saved game indices and configuration",
    )
    batch_parser.add_argument(
        "--ply-progress-every", type=int, default=0,
        help="print live per-game progress every N plies (0 disables)",
    )
    batch_parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "show":
        return show_initial()
    if args.command == "batch":
        from .training.batch_games import generate_batch

        result = generate_batch(
            args.red,
            args.black,
            games=args.games,
            output=args.output,
            seed=args.seed,
            red_depth=args.red_depth,
            black_depth=args.black_depth,
            max_plies=args.max_plies,
            opening_plies=args.opening_plies,
            no_progress_limit=args.no_progress_limit,
            progress_every=args.progress_every,
            workers=args.workers,
            ply_progress_every=args.ply_progress_every,
            resume=args.resume,
            quiet=args.quiet,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.max_plies < 1:
        raise SystemExit("--max-plies must be at least 1")
    if args.delay < 0:
        raise SystemExit("--delay cannot be negative")
    return play_random(args.seed, args.max_plies, args.delay)
