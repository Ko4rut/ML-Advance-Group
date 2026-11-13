"""Real-network Neural MCTS demos: a full game, and a per-move time-budget measurement.

    # a complete game, neural (Red) vs classic UCT+rollout MCTS (Black), equal simulations per move
    python experiments/neural_mcts/phase4/play_game.py game --model distill-v4-large \
        --neural-side red --simulations 100 --output game.json
    # neural self-play game
    python experiments/neural_mcts/phase4/play_game.py game --opponent neural --simulations 60
    # what does a 3 s move buy? (positions from a short argmax-free self-play prefix)
    python experiments/neural_mcts/phase4/play_game.py budget --model distill-v4-large --seconds 3 --moves 6

Every move is applied with ``GameEngine.apply_move`` (raises on an illegal move) and recorded.
This is a plumbing/legality/latency check, NOT a strength benchmark (Phase 6).
Needs torch and the provisioned checkpoint (models/neural/README.md).
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

BACKEND = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND))

from src.agents import MCTSAgent, NeuralMCTSAgent, SearchBudget  # noqa: E402
from src.environment import GameEngine, GameState, Move, Side  # noqa: E402
from src.networks.pretrained import NeuralModelError, NeuralModelSettings, load_adapter  # noqa: E402
from src.search.mcts import MCTSSearch, RolloutEvaluator, UCTSelection  # noqa: E402


def ucci(move: Move) -> str:
    sq = lambda i: f"{'abcdefghi'[i % 9]}{9 - i // 9}"
    return sq(move.source) + sq(move.target)


def build_neural(engine, adapter, c_puct):
    return NeuralMCTSAgent.from_config(engine, {"c_puct": c_puct, "seed": 0}, adapter)


def build_classic(engine):
    evaluator = RolloutEvaluator(engine, policy="capture_first", max_depth=2, seed=0)
    return MCTSAgent(MCTSSearch(engine, UCTSelection(1.4), evaluator, seed=0))


def load(model):
    try:
        return load_adapter(NeuralModelSettings(model=model))
    except NeuralModelError as exc:
        print(f"CANNOT LOAD ({type(exc).__name__}): {exc}", file=sys.stderr)
        raise SystemExit(2)


def game(args) -> int:
    engine, adapter = GameEngine(), load(args.model)
    neural_side = Side.RED if args.neural_side == "red" else Side.BLACK
    players = {neural_side: ("neural", build_neural(engine, adapter, args.c_puct))}
    players[neural_side.opponent] = (("neural", build_neural(engine, adapter, args.c_puct))
                                     if args.opponent == "neural" else ("classic", build_classic(engine)))
    budget = SearchBudget(max_simulations=args.simulations) if args.simulations else SearchBudget(time_limit=args.seconds)
    state, record = GameState.initial(), []
    started = time.perf_counter()
    while engine.outcome(state) is None and len(record) < args.max_plies:
        kind, agent = players[state.side_to_move]
        t = time.perf_counter()
        decision = agent.select_action(state, budget)
        elapsed = time.perf_counter() - t
        state = engine.apply_move(state, decision.move)  # raises ValueError on an illegal move
        record.append({"ply": len(record) + 1, "side": Side(-state.side_to_move).name, "agent": kind,
                       "move": ucci(decision.move), "simulations": decision.simulations,
                       "nodes": decision.nodes, "seconds": round(elapsed, 4)})
    result = engine.outcome(state)
    summary = {"model": adapter.info.name, "sha256": adapter.info.sha256, "plies": len(record),
               "neural_side": neural_side.name, "opponent": args.opponent,
               "budget": {"max_simulations": args.simulations, "seconds": None if args.simulations else args.seconds},
               "c_puct": args.c_puct,
               "result": None if result is None else {"winner": None if result.winner is None else result.winner.name,
                                                      "reason": result.reason},
               "all_moves_legal": True, "network_forward_passes": adapter.inference_calls,
               "wall_seconds": round(time.perf_counter() - started, 2),
               "neural_move_seconds_p50": round(statistics.median(r["seconds"] for r in record if r["agent"] == "neural"), 4)}
    print(json.dumps(summary, indent=1))
    print(" ".join(f"{r['move']}" for r in record))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"summary": summary, "moves": record}, indent=1) + "\n")
    return 0


def budget(args) -> int:
    engine, adapter = GameEngine(), load(args.model)
    agent = build_neural(engine, adapter, args.c_puct)
    state, rows = GameState.initial(), []
    for index in range(args.moves):
        if engine.outcome(state) is not None:
            break
        t = time.perf_counter()
        decision = agent.select_action(state, SearchBudget(time_limit=args.seconds))
        elapsed = time.perf_counter() - t
        search = agent.search
        rows.append({"move_index": index, "side": state.side_to_move.name, "move": ucci(decision.move),
                     "seconds": round(elapsed, 3), "simulations": search.last_simulations,
                     "network_evaluations": search.last_evaluations,
                     "terminal_leaves": search.last_terminal_leaves, "nodes": search.last_nodes,
                     "simulations_per_second": round(search.last_simulations / elapsed, 1)})
        state = engine.apply_move(state, decision.move)
    summary = {"model": adapter.info.name, "time_limit": args.seconds, "moves": rows,
               "median_simulations": statistics.median(r["simulations"] for r in rows),
               "max_seconds": max(r["seconds"] for r in rows),
               "overshoot_max_seconds": round(max(r["seconds"] for r in rows) - args.seconds, 3)}
    print(json.dumps(summary, indent=1))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=1) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("game", "budget"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="distill-v4-large")
        p.add_argument("--c-puct", type=float, default=1.5)
        p.add_argument("--output")
    g = sub.choices["game"]
    g.add_argument("--neural-side", choices=("red", "black"), default="red")
    g.add_argument("--opponent", choices=("classic", "neural"), default="classic")
    g.add_argument("--simulations", type=int, default=100, help="per move; 0 = use --seconds")
    g.add_argument("--seconds", type=float, default=1.0)
    g.add_argument("--max-plies", type=int, default=240)
    b = sub.choices["budget"]
    b.add_argument("--seconds", type=float, default=3.0)
    b.add_argument("--moves", type=int, default=6)
    args = parser.parse_args()
    return game(args) if args.command == "game" else budget(args)


if __name__ == "__main__":
    raise SystemExit(main())
