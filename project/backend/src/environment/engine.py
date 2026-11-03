'''Apply validated moves and determine game outcomes.'''

from .movegen import MoveGenerator
from .rules import RuleValidator, RulesConfig
from .state import GameState
from .types import GameResult, Move

class GameEngine:
    '''Rules-aware, agent-independent Xiangqi game engine.'''

    def __init__(self, rules: RulesConfig | None = None) -> None:
        self.rules = rules or RulesConfig()
        self.validator = RuleValidator()
        self.move_generator = MoveGenerator(self.validator)

    def legal_moves(self, state: GameState) -> list[Move]:
        return self.move_generator.legal_moves(state)

    def apply_move(self, state: GameState, move: Move) -> GameState:
        if not isinstance(move, Move):
            raise TypeError("move must be a Move")
        if self.outcome(state) is not None:
            raise ValueError("cannot apply a move to a terminal state")
        if move not in self.legal_moves(state):
            raise ValueError(f"illegal move: {move.source}->{move.target}")
        return state.moved(move)

    def outcome(
        self,
        state: GameState,
        legal_moves: list[Move] | None = None,
    ) -> GameResult | None:
        has_moves = (
            self.move_generator.has_legal_move(state)
            if legal_moves is None
            else bool(legal_moves)
        )
        if not has_moves:
            winner = state.side_to_move.opponent
            reason = "checkmate" if self.validator.is_in_check(state, state.side_to_move) else "stalemate"
            return GameResult(winner, reason)

        if state.history.count(state.position_key()) >= self.rules.repetition_count:
            return GameResult(None, "repetition")

        if state.no_capture_count >= self.rules.no_capture_limit:
            return GameResult(None, "no_capture_limit")

        return None

_DEFAULT_ENGINE = GameEngine()

def apply_move(state: GameState, move: Move, rules: RulesConfig | None = None) -> GameState:
    return GameEngine(rules).apply_move(state, move)


def outcome(state: GameState, rules: RulesConfig | None = None) -> GameResult | None:
    return GameEngine(rules).outcome(state)


def legal_moves(state: GameState) -> list[Move]:
    return _DEFAULT_ENGINE.legal_moves(state)