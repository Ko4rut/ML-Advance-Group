"""Shared helpers for tree-search agents."""

import time
from dataclasses import dataclass

from ..environment import GameEngine, GameState, Move, Side
from .base import SearchBudget
from .position_evaluator import PIECE_VALUES, material_score, evaluate_position, order_moves

MATE_SCORE = 1_000_000
INFINITY = 2 * MATE_SCORE


class SearchLimitReached(Exception):
    pass


@dataclass(slots=True)
class SearchContext:
    root_side: Side
    max_nodes: int | None
    deadline: float | None
    nodes: int = 1

    def visit(self) -> None:
        if self.max_nodes is not None and self.nodes >= self.max_nodes:
            raise SearchLimitReached
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise SearchLimitReached
        self.nodes += 1


def create_search_context(
    state: GameState,
    budget: SearchBudget | None,
) -> SearchContext:
    deadline = (
        time.monotonic() + budget.time_limit
        if budget is not None and budget.time_limit is not None
        else None
    )
    return SearchContext(
        root_side=state.side_to_move,
        max_nodes=budget.max_nodes if budget is not None else None,
        deadline=deadline,
    )


def search_depth(max_depth: int, budget: SearchBudget | None) -> int:
    if budget is not None and budget.max_depth is not None:
        return min(max_depth, budget.max_depth)
    return max_depth


def terminal_score(reward: int, ply: int) -> int:
    if reward > 0:
        return MATE_SCORE - ply
    if reward < 0:
        return -MATE_SCORE + ply
    return 0


def validate_search_root(engine: GameEngine, state: GameState) -> list[Move]:
    moves = engine.legal_moves(state)
    result = engine.outcome(state, legal_moves=moves)
    if result is not None:
        if result.reason in {"checkmate", "stalemate"}:
            raise ValueError("cannot select an action from a state with no legal moves")
        raise ValueError("cannot select an action from a terminal state")
    return moves
