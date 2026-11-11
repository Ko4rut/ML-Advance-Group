"""Terminal commands for inspecting the environment and generating Xiangqi games."""

import argparse
import json
import sys
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
        "--opening-mode", choices=("none", "random", "book"), default="random",
        help="none=Alpha-Beta from start, random=legacy random prefix, book=curated legal lines",
    )
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
    batch_parser.add_argument(
        "--deduplicate-games", action="store_true",
        help="skip identical game action sequences; retry with a different deterministic sampling seed",
    )
    batch_parser.add_argument(
        "--max-duplicate-retries", type=int, default=8,
        help="maximum number of retries for a duplicate game before stopping safely",
    )
    batch_parser.add_argument(
        "--exploration-top-k", type=int, default=1,
        help="1 disables exploration; 3 samples among top three sufficiently good Alpha-Beta moves",
    )
    batch_parser.add_argument(
        "--exploration-score-margin", type=int, default=12,
        help="maximum evaluation point difference from best score for Top-K candidates",
    )
    batch_parser.add_argument(
        "--exploration-temperature", type=float, default=8.0,
        help="softmax temperature for sampled Top-K candidate scores",
    )
    batch_parser.add_argument(
        "--exploration-rate", type=float, default=0.35,
        help="chance to explore at each eligible post-opening ply",
    )
    batch_parser.add_argument(
        "--exploration-window", type=int, default=20,
        help="number of plies after opening eligible for exploration",
    )
    batch_parser.add_argument(
        "--exploration-max-moves", type=int, default=2,
        help="maximum exploratory moves per game",
    )
    batch_parser.add_argument("--quiet", action="store_true")

    tournament_parser = commands.add_parser(
        "tournament",
        help="round-robin tournament between agents (25 openings x 2 colors per pair)",
    )
    tournament_parser.add_argument(
        "--config", default=None,
        help="tournament YAML (default configs/tournament.yaml; with --resume: the run's own)",
    )
    mode = tournament_parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="quick check: 2 openings x 2 colors, max 40 plies, separate directory")
    mode.add_argument("--pilot", action="store_true",
                      help="1 game per pair at the real config; prints a total-time estimate")
    tournament_parser.add_argument("--workers", default=None,
                                   help="parallel games: integer or 'auto' (default: from config)")
    tournament_parser.add_argument("--output-root", default=None,
                                   help="parent directory for run folders (default results/tournaments)")
    tournament_parser.add_argument("--resume", default=None, metavar="RUN_DIR",
                                   help="continue an interrupted run directory")
    tournament_parser.add_argument("--allow-code-change", action="store_true",
                                   help="with --resume: continue even if the git commit changed")
    tournament_parser.add_argument("--retry-errors", action="store_true",
                                   help="with --resume: re-run games that ended in an agent error")
    report_parser = commands.add_parser(
        "tournament-report",
        help="regenerate summary/ (CSV, plots, report.md) of a tournament run from its raw log",
    )
    report_parser.add_argument("run_dir", help="run directory containing manifest.json and games.jsonl")
    report_parser.add_argument("--no-plots", action="store_true", help="skip PNG charts")
    return parser


def run_tournament_report_command(args: argparse.Namespace) -> int:
    """Lazy-import the report stack, like the `tournament` subcommand."""
    from .evaluation import report

    try:
        result = report.generate_report(args.run_dir, plots=not args.no_plots)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    print(f"Report regenerated in {result['summary_dir']} ({result['completed']}/{result['total']} games)")
    if result["completed"] < result["total"]:
        print(f"WARNING: run is incomplete ({result['completed']}/{result['total']} games)")
    if result["errors"]:
        print(f"WARNING: {result['errors']} game(s) ended in an agent error (see errors.csv)")
    for path in result["files"]:
        print(f"  {path}")
    return 0


def run_tournament_command(args: argparse.Namespace, argv: Sequence[str] | None) -> int:
    """Lazy-import the evaluation stack, like the `batch` subcommand."""
    from pathlib import Path

    from .evaluation import tournament as t

    workers = args.workers
    if workers is not None and workers != "auto":
        try:
            workers = int(workers)
        except ValueError:
            raise SystemExit("--workers must be an integer or 'auto'") from None
    try:
        if args.resume:
            config, config_path = t.load_resume_config(args.resume, args.config)
        else:
            if args.allow_code_change or args.retry_errors:
                raise SystemExit("--allow-code-change and --retry-errors require --resume")
            config_path = args.config or "configs/tournament.yaml"
            path = Path(config_path)
            if not path.is_absolute() and not path.exists():
                path = t.BACKEND_ROOT / path
            mode = "smoke" if args.smoke else "pilot" if args.pilot else "full"
            config = t.resolve_config(t.load_raw_config(path), mode=mode)
        if workers is None:
            workers = config.workers
        summary = t.run_tournament(
            config,
            config_path=config_path,
            output_root=args.output_root,
            workers=workers,
            resume_dir=args.resume,
            allow_code_change=args.allow_code_change,
            retry_errors=args.retry_errors,
            argv=["python", "-m", "src", *(argv if argv is not None else sys.argv[1:])],
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    if config.mode == "pilot":
        t.print_pilot_estimate(summary["run_dir"], config.full_games_per_pair, t.resolve_workers(workers))
    return 130 if summary["interrupted"] else 0


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
            opening_mode=args.opening_mode,
            no_progress_limit=args.no_progress_limit,
            progress_every=args.progress_every,
            workers=args.workers,
            ply_progress_every=args.ply_progress_every,
            resume=args.resume,
            quiet=args.quiet,
            deduplicate_games=args.deduplicate_games,
            max_duplicate_retries=args.max_duplicate_retries,
            exploration_top_k=args.exploration_top_k,
            exploration_score_margin=args.exploration_score_margin,
            exploration_temperature=args.exploration_temperature,
            exploration_rate=args.exploration_rate,
            exploration_window=args.exploration_window,
            exploration_max_moves=args.exploration_max_moves,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "tournament":
        return run_tournament_command(args, argv)
    if args.command == "tournament-report":
        return run_tournament_report_command(args)

    if args.max_plies < 1:
        raise SystemExit("--max-plies must be at least 1")
    if args.delay < 0:
        raise SystemExit("--delay cannot be negative")
    return play_random(args.seed, args.max_plies, args.delay)
