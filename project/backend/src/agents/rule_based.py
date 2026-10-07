"""Rule-priority agent for the Xiangqi environment."""

from ..environment import GameEngine, GameState, Move
from .base import Agent, AgentDecision, SearchBudget

"""from top to bottom corresponds to: king, advisor, elephant, horse, rook, cannon, pawn"""
_PIECE_VALUES = {
    1: 10_000,
    2: 20,
    3: 20,
    4: 40,
    5: 90,
    6: 45,
    7: 10,
}


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

        moved_piece_value = _PIECE_VALUES[abs(state.board[move.source])]
        opponent_can_capture_moved_piece = any(
            reply.target == move.target
            for reply in self.engine.legal_moves(child)
        )
        safety_score = -moved_piece_value if opponent_can_capture_moved_piece else 0

        captured_piece = abs(state.board[move.target])
        captured_piece_value = _PIECE_VALUES[captured_piece] if captured_piece else 0
        return wins_now, safety_score, captured_piece_value
