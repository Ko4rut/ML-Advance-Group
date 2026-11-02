"""Lightweight, symmetrical Xiangqi evaluation for Minimax and Alpha-Beta.

The evaluator deliberately avoids legal move generation at leaf nodes.  Its
weights are hand-tuned starting points and must be validated empirically.
"""

from __future__ import annotations

from ..environment import GameState, Side

PIECE_VALUES = {
    1: 10_000,
    2: 20,
    3: 20,
    4: 40,
    5: 90,
    6: 45,
    7: 10,
}

# Only minor bonuses: the value of a piece remains much larger than the
# incentive to develop it. Count each *living* rook, horse, or cannon separately.
# This rewards bringing out the second horse instead of moving the first again.
DEVELOPMENT_BONUS = {
    4: 10,  # Horse: especially important to activate in the opening.
    5: 5,   # Rook: useful when files/lanes become available.
    6: 4,   # Cannon: starts in an already active rank in Xiangqi.
}

STARTING_SQUARES = {
    1: {
        4: frozenset((82, 88)),
        5: frozenset((81, 89)),
        6: frozenset((64, 70)),
    },
    -1: {
        4: frozenset((1, 7)),
        5: frozenset((0, 8)),
        6: frozenset((19, 25)),
    },
}


def material_score(state: GameState, side: Side) -> int:
    """Material-only baseline retained for controlled experiments."""
    difference = 0
    for piece in state.board:
        if piece:
            difference += PIECE_VALUES[abs(piece)] * (1 if piece > 0 else -1)
    return difference * int(side)


def _piece_position(piece: int, square: int) -> int:
    """Simple board-position bonus, symmetric between the two sides."""
    row, col = divmod(square, 9)
    forward = 9 - row if piece > 0 else row
    central = 4 - abs(col - 4)
    kind = abs(piece)

    if kind == 7:
        # Crossing the river matters, but a pawn step should not dominate
        # developing a horse from its initial square.
        return forward + (4 + central if forward >= 5 else 0)
    if kind == 4:
        return central * 2 + min(forward, 5)
    if kind == 5:
        return central + min(forward, 5)
    if kind == 6:
        return central + min(forward, 5)
    return 0


def _development_score(piece: int, square: int) -> int:
    kind = abs(piece)
    if kind not in DEVELOPMENT_BONUS:
        return 0
    color = 1 if piece > 0 else -1
    if square in STARTING_SQUARES[color][kind]:
        return 0
    return DEVELOPMENT_BONUS[kind]


def _king_safety(board: tuple[int, ...]) -> int:
    """Cheap palace guard feature, not a tactical attack detector."""
    score = 0
    for color in (1, -1):
        try:
            king_square = board.index(color)
        except ValueError:
            continue
        king_row, king_col = divmod(king_square, 9)
        guards = 0
        for sq, piece in enumerate(board):
            if piece * color <= 0 or abs(piece) not in (2, 3):
                continue
            row, col = divmod(sq, 9)
            if abs(row - king_row) <= 2 and abs(col - king_col) <= 2:
                guards += 1
        score += color * min(guards, 4) * 3
    return score


def evaluate_position(state: GameState, side: Side) -> int:
    """Material + positioning + per-piece activation + king guard.

    The engine decides terminal results *before* this function is called.
    Slightly discourage a repeat if ahead; if behind, allow the prospect of a
    draw. We never forbid an otherwise legal repeating move.
    """
    red_total = 0
    for square, piece in enumerate(state.board):
        if not piece:
            continue
        color = 1 if piece > 0 else -1
        red_total += color * (
            PIECE_VALUES[abs(piece)]
            + _piece_position(piece, square)
            + _development_score(piece, square)
        )
    red_total += _king_safety(state.board)
    score = red_total * int(side)

    if state.history.count(state.position_key()) >= 2:
        if score >= 0:
            score -= 8
        else:
            score += 4
    return score


def move_priority(state: GameState, move) -> tuple[int, int, int]:
    """Order captures first, then moves that improve activity."""
    victim = abs(state.board[move.target])
    attacker_piece = state.board[move.source]
    attacker = abs(attacker_piece)
    capture = PIECE_VALUES[victim] if victim else 0
    development = (
        _piece_position(attacker_piece, move.target)
        + _development_score(attacker_piece, move.target)
        - _piece_position(attacker_piece, move.source)
        - _development_score(attacker_piece, move.source)
    )
    capture_priority = capture * 10 - PIECE_VALUES[attacker] // 10 if capture else 0
    return capture_priority, development, -move.target


def order_moves(state: GameState, moves: list) -> list:
    """Sort deterministically for consistent tie-breaks and pruning."""
    return sorted(moves, key=lambda move: move_priority(state, move), reverse=True)
