"""Fast, deterministic Xiangqi static evaluation for tree-search agents.

Scores are integer project units, always from the supplied side's perspective.
No legal-move generation is performed at search leaves.
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


def material_score(state: GameState, side: Side) -> int:
    """Preserve the original material-only baseline for benchmarking."""
    difference = 0
    for piece in state.board:
        if piece:
            difference += PIECE_VALUES[abs(piece)] * (1 if piece > 0 else -1)
    return difference * int(side)


def _piece_position(piece: int, square: int) -> int:
    """Small position rewards, symmetrical across red and black."""
    row, col = divmod(square, 9)
    forward = 9 - row if piece > 0 else row
    central = 4 - abs(col - 4)
    kind = abs(piece)

    if kind == 7:  # Pawn advances and crosses river.
        return forward * 2 + (5 + central * 2 if forward >= 5 else 0)
    if kind == 4:  # Horse activity and development.
        return central * 2 + min(forward, 5)
    if kind == 5:  # Rook activity.
        return central + min(forward, 5)
    if kind == 6:  # Cannon activity.
        return central + min(forward, 5)
    return 0


def _king_safety(board: tuple[int, ...]) -> int:
    """Approximate nearby palace guards without expensive move generation.

    This is a cheap defensive feature, not tactical threat detection.
    """
    score = 0
    for side in (1, -1):
        try:
            king_index = board.index(side)
        except ValueError:
            continue
        king_row, king_col = divmod(king_index, 9)
        guards = 0
        for sq, piece in enumerate(board):
            if piece * side <= 0 or abs(piece) not in (2, 3):
                continue
            row, col = divmod(sq, 9)
            if abs(row - king_row) <= 2 and abs(col - king_col) <= 2:
                guards += 1
        score += side * min(guards, 4) * 3
    return score


def evaluate_position(state: GameState, side: Side) -> int:
    """Combine material, development, king guards, and nonterminal repetition.

    Search must still check official terminal wins/draws before calling this.
    From an advantageous position, repeating is mildly unattractive; a losing
    side is allowed to value repetition as a possible route toward a draw.
    """
    red_material = 0
    red_position = 0
    for square, piece in enumerate(state.board):
        if not piece:
            continue
        sign = 1 if piece > 0 else -1
        red_material += sign * PIECE_VALUES[abs(piece)]
        red_position += sign * _piece_position(piece, square)

    red_total = red_material + red_position + _king_safety(state.board)
    score = red_total * int(side)
    if state.history.count(state.position_key()) >= 2:
        if score >= 0:
            score -= 8
        else:
            score += 4
    return score


def move_priority(state: GameState, move) -> tuple[int, int, int]:
    """Cheap ordering: valuable captures first, then positive development."""
    victim = abs(state.board[move.target])
    attacker = abs(state.board[move.source])
    capture = PIECE_VALUES[victim] if victim else 0
    development = _piece_position(state.board[move.source], move.target) - _piece_position(
        state.board[move.source], move.source
    )
    return (capture * 10 - PIECE_VALUES[attacker] // 10) if capture else 0, development, -move.target


def order_moves(state: GameState, moves: list) -> list:
    """Sort moves for tie breaks and pruning without removing legal moves."""
    return sorted(moves, key=lambda move: move_priority(state, move), reverse=True)
