'''Generate pseudo-legal and legal Xiangqi moves.'''

from .rules import RuleValidator
from .state import GameState
from .types import Move, Side


class MoveGenerator:
    '''Generate geometry first, then filter moves that expose the moving king.'''

    def __init__(self, rules: RuleValidator | None = None) -> None:
        self.rules = rules or RuleValidator()

    def pseudo_legal_moves(self, state: GameState) -> list[Move]:
        moves: list[Move] = []
        side = state.side_to_move
        for source, piece in enumerate(state.board):
            if piece == 0:
                continue
            piece_side = Side.RED if piece > 0 else Side.BLACK
            if piece_side is not side:
                continue
            moves.extend(Move(source, target) for target in self.rules.piece_targets(state, source))
        return moves

    def legal_moves(self, state: GameState) -> list[Move]:
        """Validate candidate moves on a reusable board without constructing states."""
        legal: list[Move] = []
        board = list(state.board)
        current_side = state.side_to_move
        for move in self.pseudo_legal_moves(state):
            piece, victim = board[move.source], board[move.target]
            board[move.source] = 0
            board[move.target] = piece
            allowed = not self.rules.is_in_check(board, current_side)
            board[move.source], board[move.target] = piece, victim
            if allowed:
                legal.append(move)
        return legal

    def has_legal_move(self, state: GameState) -> bool:
        """Short-circuit terminal checks as soon as any legal reply exists."""
        board = list(state.board)
        current_side = state.side_to_move
        for move in self.pseudo_legal_moves(state):
            piece, victim = board[move.source], board[move.target]
            board[move.source] = 0
            board[move.target] = piece
            allowed = not self.rules.is_in_check(board, current_side)
            board[move.source], board[move.target] = piece, victim
            if allowed:
                return True
        return False

    @staticmethod
    def _apply_unchecked(state: GameState, move: Move) -> GameState:
        board = list(state.board)
        board[move.target] = board[move.source]
        board[move.source] = 0
        next_board = tuple(board)
        next_side = state.side_to_move.opponent
        return GameState(
            next_board,
            next_side,
            ply_count=state.ply_count + 1,
            no_capture_count=0 if state.board[move.target] != 0 else state.no_capture_count + 1,
            history=state.history + ((next_board, next_side),),
        )


_DEFAULT_MOVE_GENERATOR = MoveGenerator()


def legal_moves(state: GameState) -> list[Move]:
    return _DEFAULT_MOVE_GENERATOR.legal_moves(state)
