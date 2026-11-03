"""Run the original and optimized engines against identical Alpha-Beta positions.

From project/backend:
    python -m src.evaluation.compare_alphabeta --depth 2 --positions 3

Original classes retain the old candidate-state allocation and root move order;
optimized classes are imported from the current source. Times are per-position
medians, not guarantees across different hardware.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

from ..agents import AlphaBetaAgent
from ..agents._search_utils import (
    INFINITY, SearchLimitReached, create_search_context,
    evaluate_position, order_moves, terminal_score, validate_search_root,
)
from ..environment import GameEngine, GameState
from ..environment.movegen import MoveGenerator
from ..environment.types import Move


class OriginalMoveGenerator(MoveGenerator):
    def legal_moves(self, state: GameState) -> list[Move]:
        legal = []
        current_side = state.side_to_move
        for move in self.pseudo_legal_moves(state):
            child = self._apply_unchecked(state, move)
            if not self.rules.is_in_check(child, current_side):
                legal.append(move)
        return legal


class OriginalGameEngine(GameEngine):
    def __init__(self) -> None:
        super().__init__()
        self.move_generator = OriginalMoveGenerator(self.validator)

    def outcome(self, state, legal_moves=None):
        # Restore the expensive full move list construction at every leaf.
        moves = self.legal_moves(state) if legal_moves is None else legal_moves
        return super().outcome(state, legal_moves=moves)


class OriginalAlphaBeta:
    """Frozen original root ordering and recursive Alpha-Beta for comparison."""

    def __init__(self, engine, max_depth):
        self.engine = engine
        self.max_depth = max_depth

    def select_action(self, state):
        moves = order_moves(state, validate_search_root(self.engine, state))
        context = create_search_context(state, None)
        best_move = moves[0]
        for depth in range(1, self.max_depth + 1):
            iteration_move = best_move
            iteration_score = None
            alpha = -INFINITY
            try:
                for move in moves:
                    score = self._search(
                        state.moved(move), depth - 1, context, 1, alpha, INFINITY,
                    )
                    if iteration_score is None or score > iteration_score:
                        iteration_move, iteration_score = move, score
                    if iteration_score is not None:
                        alpha = max(alpha, iteration_score)
            except SearchLimitReached:
                break
            best_move = iteration_move
        return best_move, context.nodes

    def _search(self, state, depth, context, ply, alpha, beta):
        context.visit()
        moves = self.engine.legal_moves(state) if depth > 0 else None
        result = self.engine.outcome(state, legal_moves=moves)
        if result is not None:
            return terminal_score(result.reward(context.root_side), ply)
        if depth == 0:
            return evaluate_position(state, context.root_side)
        maximizing = state.side_to_move is context.root_side
        best_score = -INFINITY if maximizing else INFINITY
        for move in order_moves(state, moves):
            score = self._search(
                state.moved(move), depth - 1, context, ply + 1, alpha, beta,
            )
            if maximizing:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score


def sample_positions(count: int, seed: int, spacing: int):
    rng = random.Random(seed)
    engine = GameEngine()
    state = GameState.initial()
    result = []
    for _ in range(count):
        if engine.outcome(state) is not None:
            state = GameState.initial()
        result.append(state)
        for _ in range(spacing):
            moves = engine.legal_moves(state)
            if engine.outcome(state, legal_moves=moves) is not None:
                state = GameState.initial()
                break
            state = state.moved(rng.choice(moves))
    return result


def run_benchmark(
    depth: int = 2,
    positions: int = 3,
    repeats: int = 1,
    seed: int = 42,
    spacing: int = 5,
) -> dict:
    if min(depth, positions, repeats, spacing) < 1:
        raise ValueError("All benchmark sizes must be positive")
    reference = OriginalAlphaBeta(OriginalGameEngine(), depth)
    optimized = AlphaBetaAgent(GameEngine(), depth)
    rows = []
    for index, state in enumerate(sample_positions(positions, seed, spacing)):
        results = {}
        for name, agent in (("original", reference), ("optimized", optimized)):
            timings = []
            node_counts = []
            chosen = None
            for _ in range(repeats):
                start = time.perf_counter()
                decision = agent.select_action(state)
                timings.append(time.perf_counter() - start)
                if name == "original":
                    chosen, nodes = decision
                else:
                    chosen, nodes = decision.move, decision.nodes
                node_counts.append(nodes)
            results[name] = {
                "seconds": round(statistics.median(timings), 6),
                "nodes": int(statistics.median(node_counts)),
                "action": [chosen.source, chosen.target],
            }
        same = results["original"]["action"] == results["optimized"]["action"]
        speedup = results["original"]["seconds"] / max(
            results["optimized"]["seconds"], 1e-12
        )
        row = {
            "position": index,
            "ply": state.ply_count,
            "original": results["original"],
            "optimized": results["optimized"],
            "same_move": same,
            "speedup": round(speedup, 3),
        }
        rows.append(row)
        print(
            f"[{index + 1}/{positions}] ply={state.ply_count} "
            f"old={results['original']['seconds']:.3f}s "
            f"new={results['optimized']['seconds']:.3f}s "
            f"speedup={speedup:.2f}x same_move={same}",
            flush=True,
        )
    old = sum(row["original"]["seconds"] for row in rows)
    new = sum(row["optimized"]["seconds"] for row in rows)
    return {
        "depth": depth,
        "positions": positions,
        "repeats": repeats,
        "original_seconds": round(old, 6),
        "optimized_seconds": round(new, 6),
        "overall_speedup": round(old / max(new, 1e-12), 3),
        "same_move_count": sum(row["same_move"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare original vs optimized Xiangqi engine/Alpha-Beta"
    )
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--positions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacing", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
