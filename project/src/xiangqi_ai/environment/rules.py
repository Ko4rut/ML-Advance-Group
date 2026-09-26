'''Xiangqi movement geometry and check detection.'''

from dataclasses import dataclass

from .state import BOARD_COLS, BOARD_ROWS, GameState, row_col, square
from .types import Piece, Side


@dataclass(frozen=True, slots=True)
class RulesConfig:
    '''Rule thresholds for the simplified project rule set.'''

    repetition_count: int = 3
    no_capture_limit: int = 120

    def __post_init__(self) -> None:
        if self.repetition_count < 2:
            raise ValueError("repetition_count must be at least 2")
        if self.no_capture_limit < 1:
            raise ValueError("no_capture_limit must be at least 1")


def _board(value: GameState | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, GameState):
        return value.board
    board = tuple(value)
    if len(board) != BOARD_ROWS * BOARD_COLS:
        raise ValueError("board must contain exactly 90 squares")
    return board


def _side_of(piece: int) -> Side:
    return Side.RED if piece > 0 else Side.BLACK


class RuleValidator:
    '''Generate geometry and detect checks.'''

    def in_palace(self, row: int, col: int, side: Side) -> bool:
        side = Side(side)
        if not (3 <= col <= 5):
            return False
        return (7 <= row <= 9) if side is Side.RED else (0 <= row <= 2)

    def piece_targets(self, state: GameState | tuple[int, ...] | list[int], source: int) -> tuple[int, ...]:
        board = _board(state)
        row, col = row_col(source)
        piece = board[source]
        if piece == 0:
            return ()
        side = _side_of(piece)
        kind = Piece(abs(piece))
        targets_out: list[int] = []

        def add(r: int, c: int) -> None:
            if not (0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS):
                return
            target = square(r, c)
            if board[target] == 0 or _side_of(board[target]) is not side:
                targets_out.append(target)

        if kind is Piece.KING:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if self.in_palace(nr, nc, side):
                    add(nr, nc)

        elif kind is Piece.ADVISOR:
            for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                nr, nc = row + dr, col + dc
                if self.in_palace(nr, nc, side):
                    add(nr, nc)

        elif kind is Piece.ELEPHANT:
            for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS):
                    continue
                if side is Side.RED and nr < 5:
                    continue
                if side is Side.BLACK and nr > 4:
                    continue
                eye = square(row + dr // 2, col + dc // 2)
                if board[eye] == 0:
                    add(nr, nc)

        elif kind is Piece.HORSE:
            horse_moves = (
                (-2, -1, -1, 0), (-2, 1, -1, 0),
                (2, -1, 1, 0), (2, 1, 1, 0),
                (-1, -2, 0, -1), (1, -2, 0, -1),
                (-1, 2, 0, 1), (1, 2, 0, 1),
            )
            for dr, dc, leg_dr, leg_dc in horse_moves:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS):
                    continue
                if board[square(row + leg_dr, col + leg_dc)] != 0:
                    continue
                add(nr, nc)

        elif kind in (Piece.ROOK, Piece.CANNON):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                jumped = False
                while 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS:
                    target = square(nr, nc)
                    occupant = board[target]
                    if kind is Piece.ROOK:
                        if occupant == 0:
                            targets_out.append(target)
                        else:
                            if _side_of(occupant) is not side:
                                targets_out.append(target)
                            break
                    elif not jumped:
                        if occupant == 0:
                            targets_out.append(target)
                        else:
                            jumped = True
                    else:
                        if occupant != 0:
                            if _side_of(occupant) is not side:
                                targets_out.append(target)
                            break
                    nr += dr
                    nc += dc

        elif kind is Piece.PAWN:
            forward = -1 if side is Side.RED else 1
            add(row + forward, col)
            crossed = row <= 4 if side is Side.RED else row >= 5
            if crossed:
                add(row, col - 1)
                add(row, col + 1)

        return tuple(targets_out)

    def is_in_check(self, state: GameState | tuple[int, ...] | list[int], side: Side) -> bool:
        board = _board(state)
        side = Side(side)
        king_piece = int(Piece.KING) * int(side)
        try:
            king_square = board.index(king_piece)
        except ValueError:
            return True

        king_row, king_col = row_col(king_square)
        other_king = -king_piece
        try:
            other_square = board.index(other_king)
        except ValueError:
            other_square = None
        if other_square is not None:
            other_row, other_col = row_col(other_square)
            if other_col == king_col:
                step = 1 if other_row > king_row else -1
                if all(board[square(r, king_col)] == 0
                       for r in range(king_row + step, other_row, step)):
                    return True

        enemy = side.opponent
        for source, piece in enumerate(board):
            if piece == 0 or _side_of(piece) is not enemy:
                continue
            if king_square in self.piece_targets(board, source):
                return True
        return False


_DEFAULT_RULE_VALIDATOR = RuleValidator()


def targets(board: GameState | tuple[int, ...] | list[int], source: int) -> tuple[int, ...]:
    return _DEFAULT_RULE_VALIDATOR.piece_targets(board, source)


def is_in_check(state: GameState | tuple[int, ...] | list[int], side: Side) -> bool:
    return _DEFAULT_RULE_VALIDATOR.is_in_check(state, side)
