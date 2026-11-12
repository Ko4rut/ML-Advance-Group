"""Semantic sanity probes for the pinned pretrained cc-lite checkpoint (no MCTS, no training).

Complements qualify.py (structural/source parity). Here we ask whether the real
network's outputs are *meaningful* for our engine:

  A. Opening line: top policy moves for a well-known opening, printed in UCCI notation.
  B. Value sign: Red/Black up a rook (or a cannon) -> value for the side to move.
  C. Policy-only play (argmax over legal priors, no search) vs a seeded uniform-random mover.

Not a strength benchmark. Results are evidence of semantics only.
"""

import argparse
import json
from pathlib import Path
import random

from contracts import color_swap, predict
from qualify import CheckpointProbe
from src.environment import GameEngine, GameState, Move, Side

FILES = "abcdefghi"


def ucci(move: Move) -> str:
    def cell(index):
        row, col = divmod(index, 9)
        return f"{FILES[col]}{9 - row}"
    return cell(move.source) + cell(move.target)


def parse_ucci(text: str) -> Move:
    def idx(cell):
        return (9 - int(cell[1])) * 9 + FILES.index(cell[0])
    return Move(idx(text[:2]), idx(text[2:]))


def top(priors, n=3):
    ranked = sorted(priors.items(), key=lambda item: -item[1])[:n]
    return [(ucci(move), round(p, 4)) for move, p in ranked]


def opening_line(probe, engine):
    state, rows = GameState.initial(), []
    # Central cannon vs screen horse, then development: all checked legal by the engine.
    for text in ("h2e2", "h9g7", "h0g2", "i9h9", "i0h0", "b9c7"):
        priors, value = predict(state, engine, probe.forward)
        rows.append({"side": state.side_to_move.name, "value": round(value, 4),
                     "top3": top(priors), "played": text,
                     "played_prior": round(priors[parse_ucci(text)], 4)})
        state = state.moved(parse_ucci(text))
    priors, value = predict(state, engine, probe.forward)
    rows.append({"side": state.side_to_move.name, "value": round(value, 4),
                 "top3": top(priors), "played": None})
    return rows


def without(state, *squares, side=None):
    board = list(state.board)
    for square in squares:
        board[square] = 0
    return GameState(tuple(board), side or state.side_to_move)


def value_sign(probe, engine):
    initial = GameState.initial()
    # Initial squares: black rooks 0, 8; black cannons 19, 25; red rooks 81, 89; red cannons 64, 70.
    cases = {"black_loses_rook": (0,), "red_loses_rook": (81,),
             "black_loses_both_rooks": (0, 8), "red_loses_both_rooks": (81, 89),
             "black_loses_cannon": (19,), "red_loses_cannon": (64,)}
    rows = {}
    for name, squares in cases.items():
        entry = {}
        for side in (Side.RED, Side.BLACK):
            _, value = predict(without(initial, *squares, side=side), engine, probe.forward)
            entry[f"value_{side.name.lower()}_to_move"] = round(value, 4)
        rows[name] = entry
    return rows


def policy_game(probe, engine, policy_side, seed, max_plies=200):
    rng, state = random.Random(seed), GameState.initial()
    for ply in range(max_plies):
        moves = engine.legal_moves(state)
        result = engine.outcome(state, legal_moves=moves)
        if result is not None:
            return ply, result
        if state.side_to_move is policy_side:
            priors, _ = predict(state, engine, probe.forward)
            move = max(moves, key=lambda m: (priors[m], -m.source, -m.target))
        else:
            move = rng.choice(moves)
        state = state.moved(move)
    return max_plies, None


def policy_vs_random(probe, engine, games_per_side):
    summary = {"games": 0, "policy_wins": 0, "draws": 0, "policy_losses": 0,
               "unfinished_at_cap": 0, "per_game": []}
    for side in (Side.RED, Side.BLACK):
        for seed in range(games_per_side):
            plies, result = policy_game(probe, engine, side, seed)
            summary["games"] += 1
            if result is None:
                outcome = "unfinished"; summary["unfinished_at_cap"] += 1
            elif result.winner is None:
                outcome = f"draw:{result.reason}"; summary["draws"] += 1
            elif result.winner is side:
                outcome = "policy_win"; summary["policy_wins"] += 1
            else:
                outcome = "policy_loss"; summary["policy_losses"] += 1
            summary["per_game"].append({"policy_side": side.name, "seed": seed,
                                        "plies": plies, "outcome": outcome})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-side", type=int, default=10)
    parser.add_argument("--variant", default="distill-v4")
    args = parser.parse_args()
    engine = GameEngine()
    probe = CheckpointProbe(args.source, args.checkpoint, args.variant)
    result = {"opening_line": opening_line(probe, engine),
              "value_sign": value_sign(probe, engine),
              "policy_only_vs_random": policy_vs_random(probe, engine, args.games_per_side),
              "forward_calls": probe.forward_calls, "mcts_executed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "policy_only_vs_random"}, indent=1))
    games = result["policy_only_vs_random"]
    print({k: v for k, v in games.items() if k != "per_game"})


if __name__ == "__main__":
    main()
