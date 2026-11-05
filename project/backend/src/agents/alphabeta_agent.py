"""Alpha-beta pruning agent for Xiangqi."""

from ..environment import GameEngine, GameState, Move
from .base import Agent, AgentDecision, SearchBudget
from ._search_utils import (
    INFINITY,
    SearchContext,
    SearchLimitReached,
    create_search_context,
    evaluate_position,
    order_moves,
    search_depth,
    terminal_score,
    validate_search_root,
)


class AlphaBetaAgent(Agent):
    """Minimax with alpha-beta pruning and inexpensive tactical ordering."""

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
        moves = order_moves(state, validate_search_root(self.engine, state))
        depth_limit = search_depth(self.max_depth, budget)
        context = create_search_context(state, budget)

        best_move = moves[0]
        move_ranks = {move: rank for rank, move in enumerate(moves)}
        for depth in range(1, depth_limit + 1):
            iteration_move = best_move
            iteration_score: int | None = None
            alpha = -INFINITY
            # Principal-variation ordering improves pruning. Original ordering
            # still decides ties, so equal-scoring move choice stays stable.
            ordered = moves if depth == 1 else [best_move] + [
                move for move in moves if move != best_move
            ]
            try:
                for move in ordered:
                    score = self._alphabeta(
                        state.moved(move),
                        depth - 1,
                        context,
                        ply=1,
                        alpha=alpha,
                        beta=INFINITY,
                    )
                    if (
                        iteration_score is None
                        or score > iteration_score
                        or (
                            score == iteration_score
                            and move_ranks[move] < move_ranks[iteration_move]
                        )
                    ):
                        iteration_move = move
                        iteration_score = score
                    if iteration_score is not None:
                        alpha = max(alpha, iteration_score)
            except SearchLimitReached:
                break
            best_move = iteration_move

        return AgentDecision(move=best_move, nodes=context.nodes)

    def rank_actions(
        self,
        state: GameState,
    ) -> tuple[list[tuple["Move", int]], int]:
        """Score all legal root moves for controlled training exploration.

        Each root child uses the *full* alpha-beta window, yielding comparable
        minimax scores. Reusing the running root alpha would give only bounds
        for inferior moves and is unsafe for Top-K probabilities.

        This intentionally costs more than select_action and must be called
        only on occasional exploration plies, not on every normal move.
        """
        moves = order_moves(state, validate_search_root(self.engine, state))
        context = create_search_context(state, None)
        scores: list[tuple[Move, int]] = []
        for move in moves:
            score = self._alphabeta(
                state.moved(move),
                self.max_depth - 1,
                context,
                ply=1,
                alpha=-INFINITY,
                beta=INFINITY,
            )
            scores.append((move, score))
        # Python stable sort preserves the original deterministic move order
        # for equal scores; score is always from the root player's perspective.
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores, context.nodes

    def _alphabeta(
        self,
        state: GameState,
        depth: int,
        context: SearchContext,
        ply: int,
        alpha: int,
        beta: int,
    ) -> int:
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
            score = self._alphabeta(
                state.moved(move),
                depth - 1,
                context,
                ply + 1,
                alpha,
                beta,
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
