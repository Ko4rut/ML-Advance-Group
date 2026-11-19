"""Core state and rules for the Connect 4 assignment.

A board is a 2D integer NumPy array (0 empty, 1 / -1 the two players). The functions
``actions``, ``result``, ``terminal`` and ``utility`` take such a board, check it once and
answer. Code that asks many questions about the same board (a search) should use
``Position`` instead: it is checked once, remembers who has won, and every ``drop`` gives
a new ``Position`` that is valid by construction, so nothing is checked again.
"""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Board: TypeAlias = NDArray[np.integer]
Player: TypeAlias = int

_WIN_LENGTH = 4
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _validate_board(state: Board) -> Board:
    if not isinstance(state, np.ndarray):
        raise TypeError("state must be a NumPy ndarray")
    if state.ndim != 2 or any(size == 0 for size in state.shape):
        raise ValueError("state must be a non-empty 2D board")
    if not np.issubdtype(state.dtype, np.integer):
        raise TypeError("state must have an integer dtype")
    if state.min() < -1 or state.max() > 1:
        raise ValueError("board cells must be -1, 0, or 1")
    return state


def _validate_player(player: Player) -> None:
    if isinstance(player, bool) or not isinstance(player, Integral) or player not in (-1, 1):
        raise ValueError("player must be 1 or -1")


def _winning_players(state: Board) -> frozenset[int]:
    """Scan the whole board for four in a row; used once when a board enters ``Position``."""
    rows, columns = state.shape
    winners = set()
    for row in range(rows):
        for column in range(columns):
            player = int(state[row, column])
            if player == 0:
                continue
            for row_step, column_step in _DIRECTIONS:
                end_row = row + (_WIN_LENGTH - 1) * row_step
                end_column = column + (_WIN_LENGTH - 1) * column_step
                if not (0 <= end_row < rows and 0 <= end_column < columns):
                    continue
                if all(
                    state[row + offset * row_step, column + offset * column_step] == player
                    for offset in range(1, _WIN_LENGTH)
                ):
                    winners.add(player)
    return frozenset(winners)


def _completes_four(board: Board, row: int, column: int) -> bool:
    """Whether the disc just placed at (row, column) is part of four in a row."""
    rows, columns = board.shape
    player = board[row, column]
    for row_step, column_step in _DIRECTIONS:
        length = 1
        for sign in (1, -1):
            r, c = row + sign * row_step, column + sign * column_step
            while 0 <= r < rows and 0 <= c < columns and board[r, c] == player:
                length += 1
                r, c = r + sign * row_step, c + sign * column_step
        if length >= _WIN_LENGTH:
            return True
    return False


class Position:
    """A checked board together with who has won.

    Build one from outside data with ``Position.from_board`` (the only place a board is
    checked); get the next one with ``drop``. ``board`` must not be modified.
    """

    __slots__ = ("board", "winners", "empty_cells")

    def __init__(self, board: Board, winners: frozenset[int], empty_cells: int) -> None:
        self.board = board
        self.winners = winners
        self.empty_cells = empty_cells

    @classmethod
    def from_board(cls, state: Board) -> Position:
        """Check ``state`` and scan it once. A board where both players have four in a
        row is accepted here; ``utility`` reports it as invalid."""
        board = _validate_board(state)
        return cls(board, _winning_players(board), int(np.count_nonzero(board == 0)))

    @property
    def terminal(self) -> bool:
        """Whether a player has four in a row or the board is full."""
        return bool(self.winners) or self.empty_cells == 0

    def legal_columns(self) -> list[int]:
        """Columns that still have an empty top cell; none once the game is over."""
        if self.terminal:
            return []
        return np.flatnonzero(self.board[0] == 0).tolist()

    def utility(self, player: Player) -> int | None:
        """Terminal utility from ``player``'s view (1 win, 0 draw, -1 loss); None if ongoing."""
        _validate_player(player)
        if len(self.winners) > 1:
            raise ValueError("state is invalid because both players have winning lines")
        if self.winners:
            return 1 if player in self.winners else -1
        return 0 if self.empty_cells == 0 else None

    def drop(self, player: Player, column: int) -> Position:
        """Return the position after ``player`` drops a disc into ``column``."""
        _validate_player(player)
        if self.terminal:
            raise ValueError("cannot make a move from a terminal state")
        if isinstance(column, bool) or not isinstance(column, Integral):
            raise TypeError("action must be an integer column index")
        if column < 0 or column >= self.board.shape[1]:
            raise ValueError("action is outside the board")
        if self.board[0, column] != 0:
            raise ValueError(f"column {column} is full")

        board = self.board.copy()
        row = int(np.flatnonzero(board[:, column] == 0)[-1])
        board[row, column] = player
        # The position had no winner (it was not terminal), so only a line through the
        # new disc can win.
        winners = frozenset((int(player),)) if _completes_four(board, row, column) else frozenset()
        return Position(board, winners, self.empty_cells - 1)


def empty_board(shape: tuple[int, int] = (6, 7)) -> Board:
    """Create an empty integer board; rows and columns must both be positive."""
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("shape must be a tuple of (rows, columns)")
    if any(
        isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
        for size in shape
    ):
        raise ValueError("board dimensions must be positive integers")
    return np.zeros(shape, dtype=np.int8)


def actions(state: Board) -> list[int]:
    """Return the indices of columns that still have an empty cell."""
    return Position.from_board(state).legal_columns()


def result(state: Board, player: Player, action: int) -> Board:
    """Return a copied board after dropping ``player``'s disc into ``action``."""
    return Position.from_board(state).drop(player, action).board


def terminal(state: Board) -> bool:
    """Return whether a player has four in a row or the board is full."""
    return Position.from_board(state).terminal


def utility(state: Board, player: Player) -> int | None:
    """Return terminal utility from ``player``'s view; return None if ongoing."""
    return Position.from_board(state).utility(player)


def random_player(board: Board, player: Player = 1) -> int:
    """Choose uniformly from legal columns; fail explicitly if no move is possible."""
    position = Position.from_board(board)
    _validate_player(player)
    legal_actions = position.legal_columns()
    if not legal_actions:
        raise ValueError("random_player cannot move from a terminal state")
    return int(np.random.choice(legal_actions))
