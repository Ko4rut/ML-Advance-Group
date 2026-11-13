"""Rule-priority agent for the Xiangqi environment."""

from ..environment import GameEngine, GameState, Move
from .base import Agent, AgentDecision, SearchBudget
from .position_evaluator import PIECE_VALUES, move_priority


class RuleBasedAgent(Agent):
    """Choose a legal move using lexicographically ordered tactical rules."""

    def __init__(self, engine: GameEngine) -> None:
        self.engine = engine

    def select_action(
        self,
        state: GameState,
        budget: SearchBudget | None = None,
    ) -> AgentDecision:
        del budget
        moves = self.engine.legal_moves(state)
        if not moves:
            raise ValueError("cannot select an action from a state with no legal moves")

        move = max(moves, key=lambda candidate: self._priority(state, candidate))
        return AgentDecision(move=move)

    def _priority(self, state: GameState, move: Move) -> tuple[int, int, int]:
        moving_side = state.side_to_move
        child = state.moved(move)
        result = self.engine.outcome(child)
        wins_now = int(result is not None and result.winner is moving_side)
        if wins_now:
            return 1, 0, 0

        opponent_moves = self.engine.legal_moves(child)
        moved_piece_value = PIECE_VALUES[abs(state.board[move.source])]
        captured_piece = abs(state.board[move.target])
        captured_piece_value = PIECE_VALUES[captured_piece] if captured_piece else 0
        exchange_score = captured_piece_value
        opponent_winning_replies = 0

        for reply in opponent_moves:
            reply_state = child.moved(reply)
            reply_result = self.engine.outcome(reply_state)
            if reply_result is not None and reply_result.winner is moving_side.opponent:
                opponent_winning_replies += 1

            if reply.target != move.target:
                continue

            continuation_moves = self.engine.legal_moves(reply_state)
            exchange_result = captured_piece_value - moved_piece_value
            if any(move.target == reply.target for move in continuation_moves):
                attacker_value = PIECE_VALUES[abs(child.board[reply.source])]
                exchange_result += attacker_value
            exchange_score = min(exchange_score, exchange_result)

        position_score = move_priority(state, move)[1]
        return -opponent_winning_replies, exchange_score, position_score
