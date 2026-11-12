"""Minimal WXF (World Xiangqi Federation) move notation helpers.

Only used to *define* and *audit* opening-book lines; never on a hot path.

Conventions (verified against the v1 book in tests):

* Piece letters: K king, A advisor, E elephant, H horse, R rook, C cannon, P pawn.
* Files are numbered 1-9 from each player's own right-hand side.
  Red: ``col = 9 - file``. Black: ``col = file - 1``.
* Red advances toward row 0 (``+`` = row decreases); Black toward row 9.
* ``+``/``-`` = forward/backward, ``=`` = along the rank. After ``+``/``-``
  a rook, cannon, pawn or king takes a step count; a horse, elephant or
  advisor takes the destination file.
* Only the form ``<piece><file><op><arg>`` is supported. Two like pieces on
  the same file (``+C``/``-C`` prefix forms) raise ``NotImplementedError``.
"""

from __future__ import annotations

import re

from ..environment import GameState, Move, Piece, Side

_LETTER_TO_PIECE = {
    "K": Piece.KING, "A": Piece.ADVISOR, "E": Piece.ELEPHANT, "B": Piece.ELEPHANT,
    "H": Piece.HORSE, "N": Piece.HORSE, "R": Piece.ROOK, "C": Piece.CANNON,
    "P": Piece.PAWN,
}
_PIECE_TO_LETTER = {
    Piece.KING: "K", Piece.ADVISOR: "A", Piece.ELEPHANT: "E", Piece.HORSE: "H",
    Piece.ROOK: "R", Piece.CANNON: "C", Piece.PAWN: "P",
}
_STRAIGHT = {Piece.KING, Piece.ROOK, Piece.CANNON, Piece.PAWN}
_PATTERN = re.compile(r"^([KABEHNRCP])([1-9])([+\-=])([1-9])$")


def file_to_col(side: Side, file: int) -> int:
    return 9 - file if side is Side.RED else file - 1


def col_to_file(side: Side, col: int) -> int:
    return 9 - col if side is Side.RED else col + 1


def _forward(side: Side) -> int:
    """Row delta of one step forward."""
    return -1 if side is Side.RED else 1


def _pieces_on_file(state: GameState, side: Side, kind: Piece, file: int) -> list[int]:
    col = file_to_col(side, file)
    found = []
    for row in range(10):
        value = state.board[row * 9 + col]
        if value * side > 0 and abs(value) == kind:
            found.append(row * 9 + col)
    return found


def wxf(state: GameState, notation: str) -> Move:
    """Convert WXF notation to a :class:`Move` for the side to move."""
    match = _PATTERN.match(notation.strip())
    if not match:
        raise ValueError(f"Unsupported WXF notation: {notation!r}")
    letter, file_text, op, arg_text = match.groups()
    kind = _LETTER_TO_PIECE[letter]
    side = state.side_to_move
    candidates = _pieces_on_file(state, side, kind, int(file_text))
    if not candidates:
        raise ValueError(f"No {letter} on file {file_text} for {side.name}: {notation!r}")
    if len(candidates) > 1:
        raise NotImplementedError(f"Ambiguous like pieces on one file: {notation!r}")
    source = candidates[0]
    row, col = divmod(source, 9)
    arg = int(arg_text)
    if op == "=":
        if kind not in _STRAIGHT:
            raise ValueError(f"'=' is only valid for straight movers: {notation!r}")
        target_row, target_col = row, file_to_col(side, arg)
    else:
        direction = _forward(side) * (1 if op == "+" else -1)
        if kind in _STRAIGHT:
            target_row, target_col = row + direction * arg, col
        else:
            target_col = file_to_col(side, arg)
            span = abs(target_col - col)
            if kind is Piece.HORSE:
                drow = {1: 2, 2: 1}.get(span)
            elif kind is Piece.ELEPHANT:
                drow = 2 if span == 2 else None
            else:  # advisor
                drow = 1 if span == 1 else None
            if drow is None:
                raise ValueError(f"Impossible {letter} destination: {notation!r}")
            target_row = row + direction * drow
    if not (0 <= target_row < 10 and 0 <= target_col < 9):
        raise ValueError(f"Move leaves the board: {notation!r}")
    return Move(source, target_row * 9 + target_col)


def to_wxf(state: GameState, move: Move) -> str:
    """Convert a :class:`Move` to WXF notation (inverse of :func:`wxf`)."""
    side = state.side_to_move
    value = state.board[move.source]
    if value == 0 or value * side <= 0:
        raise ValueError("Source square does not hold a piece of the side to move")
    kind = Piece(abs(value))
    row, col = divmod(move.source, 9)
    target_row, target_col = divmod(move.target, 9)
    file = col_to_file(side, col)
    if len(_pieces_on_file(state, side, kind, file)) > 1:
        raise NotImplementedError("Ambiguous like pieces on one file")
    letter = _PIECE_TO_LETTER[kind]
    if target_row == row:
        return f"{letter}{file}={col_to_file(side, target_col)}"
    forward = (target_row - row) * _forward(side) > 0
    op = "+" if forward else "-"
    if kind in _STRAIGHT:
        if target_col != col:
            raise ValueError("Straight mover cannot change file and rank together")
        return f"{letter}{file}{op}{abs(target_row - row)}"
    return f"{letter}{file}{op}{col_to_file(side, target_col)}"


def replay_notation(notation: tuple[str, ...]) -> tuple[Move, ...]:
    """Convert a whole line from the initial position (no legality check)."""
    state = GameState.initial()
    moves = []
    for text in notation:
        action = wxf(state, text)
        moves.append(action)
        state = state.moved(action)
    return tuple(moves)


def notate_line(moves: tuple[Move, ...]) -> tuple[str, ...]:
    """Convert a whole line from the initial position to WXF notation."""
    state = GameState.initial()
    text = []
    for action in moves:
        text.append(to_wxf(state, action))
        state = state.moved(action)
    return tuple(text)
