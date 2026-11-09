"""Core state and rules for the Connect 4 assignment."""

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
    if not np.isin(state, (-1, 0, 1)).all():
        raise ValueError("board cells must be -1, 0, or 1")
    return state


def _validate_player(player: Player) -> None:
    if isinstance(player, bool) or not isinstance(player, Integral) or player not in (-1, 1):
        raise ValueError("player must be 1 or -1")


def _winning_players(state: Board) -> set[int]:
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
    return winners


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
    board = _validate_board(state)
    if terminal(board):
        return []
    return [column for column in range(board.shape[1]) if board[0, column] == 0]


def result(state: Board, player: Player, action: int) -> Board:
    """Return a copied board after dropping ``player``'s disc into ``action``."""
    board = _validate_board(state)
    _validate_player(player)
    if terminal(board):
        raise ValueError("cannot make a move from a terminal state")
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise TypeError("action must be an integer column index")
    if action < 0 or action >= board.shape[1]:
        raise ValueError("action is outside the board")
    if board[0, action] != 0:
        raise ValueError(f"column {action} is full")

    next_board = board.copy()
    empty_rows = np.flatnonzero(next_board[:, action] == 0)
    next_board[empty_rows[-1], action] = player
    return next_board


def terminal(state: Board) -> bool:
    """Return whether a player has four in a row or the board is full."""
    board = _validate_board(state)
    return bool(_winning_players(board)) or not bool(np.any(board == 0))


def utility(state: Board, player: Player) -> int | None:
    """Return terminal utility from ``player``'s view; return None if ongoing."""
    board = _validate_board(state)
    _validate_player(player)
    winners = _winning_players(board)
    if len(winners) > 1:
        raise ValueError("state is invalid because both players have winning lines")
    if winners:
        return 1 if next(iter(winners)) == player else -1
    if not np.any(board == 0):
        return 0
    return None


def random_player(board: Board, player: Player = 1) -> int:
    """Choose uniformly from legal columns; fail explicitly if no move is possible."""
    _validate_board(board)
    _validate_player(player)
    legal_actions = actions(board)
    if not legal_actions:
        raise ValueError("random_player cannot move from a terminal state")
    return int(np.random.choice(legal_actions))
