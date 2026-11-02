"""Depth-limited minimax agent for Xiangqi."""

from ..environment import GameEngine, GameState, Side
from .base import Agent, AgentDecision, SearchBudget
from ._search_utils import (
    SearchContext,
    SearchLimitReached,
    create_search_context,
    material_score,
    search_depth,
    terminal_score,
    validate_search_root,
)

class MinimaxAgent(Agent):
    """Choose a move by exploring every branch of a minimax tree."""

    def __init__(self, engine: GameEngine, max_depth: int = 2) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        self.engine = engine
        self.max_depth = max_depth

    def select_action(
        self,
        state: GameState,
        budget: SearchBudget | None = None,
    ) -> AgentDecision:
        moves = validate_search_root(self.engine, state)
        depth_limit = search_depth(self.max_depth, budget)
        context = create_search_context(state, budget)

        best_move = moves[0]
        for depth in range(1, depth_limit + 1):
            iteration_move = best_move
            iteration_score: int | None = None
            try:
                for move in moves:
                    score = self._minimax(
                        state.moved(move),
                        depth - 1,
                        context,
                        ply=1,
                    )
                    if iteration_score is None or score > iteration_score:
                        iteration_move = move
                        iteration_score = score
            except SearchLimitReached:
                break
            best_move = iteration_move

        return AgentDecision(move=best_move, nodes=context.nodes)

    def _minimax(
        self,
        state: GameState,
        depth: int,
        context: SearchContext,
        ply: int,
    ) -> int:
        context.visit()
        moves = self.engine.legal_moves(state) if depth > 0 else None
        result = self.engine.outcome(state, legal_moves=moves)
        if result is not None:
            return terminal_score(result.reward(context.root_side), ply)
        if depth == 0:
            return material_score(state, context.root_side)

        scores = (
            self._minimax(state.moved(move), depth - 1, context, ply + 1)
            for move in moves
        )
        if state.side_to_move is context.root_side:
            return max(scores)
        return min(scores)