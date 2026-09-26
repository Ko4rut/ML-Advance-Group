'''Immutable board state and rule-related history.'''

from dataclasses import dataclass

from .types import Move, Side


BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

PositionKey = tuple[tuple[int, ...], Side]


def square(row: int, col: int) -> int:
    if not (0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS):
        raise ValueError("row/col out of board bounds")
    return row * BOARD_COLS + col


def row_col(index: int) -> tuple[int, int]:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < BOARD_SIZE:
        raise ValueError("square index must be in [0, 90)")
    return divmod(index, BOARD_COLS)


_INITIAL_BOARD = (
    -5, -4, -3, -2, -1, -2, -3, -4, -5,
     0,  0, -6,  0,  0,  0, -6,  0,  0,
    -7,  0, -7,  0, -7,  0, -7,  0, -7,
     0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0, 0,  0,  0,
     0,  0,  0,  0,  0,  0, 0,  0,  0,
     7,  0,  7,  0,  7,  0, 7,  0,  7,
     0,  0,  6,  0,  0,  0,  6,  0,  0,
     0,  0,  0,  0,  0,  0, 0,  0,  0,
     5,  4,  3,  2,  1,  2, 3,  4,  5,
)


@dataclass(frozen=True, slots=True)
class GameState:
    '''Immutable game state.

    History stores position keys (board + side to move), including current.
    '''

    board: tuple[int, ...]
    side_to_move: Side
    ply_count: int = 0
    no_capture_count: int = 0
    history: tuple[PositionKey, ...] = ()

    def __post_init__(self) -> None:
        board = tuple(self.board)
        if len(board) != BOARD_SIZE:
            raise ValueError(f"board must contain exactly {BOARD_SIZE} squares")
        if any(not isinstance(piece, int) or isinstance(piece, bool) or abs(piece) > 7
               for piece in board):
            raise ValueError("board contains an invalid piece value")
        if not isinstance(self.side_to_move, Side):
            object.__setattr__(self, "side_to_move", Side(self.side_to_move))
        if self.ply_count < 0 or self.no_capture_count < 0:
            raise ValueError("counters cannot be negative")

        history = tuple(self.history)
        if not history:
            history = ((board, self.side_to_move),)
        else:
            normalized: list[PositionKey] = []
            for key_board, key_side in history:
                key_board = tuple(key_board)
                if len(key_board) != BOARD_SIZE:
                    raise ValueError("history contains an invalid board")
                normalized.append((key_board, Side(key_side)))
            history = tuple(normalized)
            if history[-1] != (board, self.side_to_move):
                raise ValueError("history must end with the current position")
        object.__setattr__(self, "board", board)
        object.__setattr__(self, "history", history)

    @classmethod
    def initial(cls) -> "GameState":
        return cls(_INITIAL_BOARD, Side.RED)

    def clone(self) -> "GameState":
        '''Return a safe clone; immutability makes sharing safe.'''
        return self

    def position_key(self) -> PositionKey:
        return self.board, self.side_to_move

    def moved(self, move: Move) -> "GameState":
        '''Apply a move after the caller has validated it.'''
        board = list(self.board)
        captured = board[move.target] != 0
        board[move.target] = board[move.source]
        board[move.source] = 0
        next_side = self.side_to_move.opponent
        next_board = tuple(board)
        return GameState(
            next_board,
            next_side,
            ply_count=self.ply_count + 1,
            no_capture_count=0 if captured else self.no_capture_count + 1,
            history=self.history + ((next_board, next_side),),
        )
