"""Inexpensive Xiangqi hanging-piece / unfavorable-exchange heuristic.

This is *static pseudo-attack* assessment, not full static exchange evaluation:
pins, legal recaptures and multi-capture sequences are left to Alpha-Beta.
Do not call legal_moves() or create GameState inside a leaf evaluator.
"""

from __future__ import annotations

from collections.abc import Sequence

# Piece values are deliberately shared with the positional evaluator.
# Import lazily inside the scorer to avoid tying attack geometry to the engine.
THREATENED_KINDS = frozenset((4, 5, 6))  # horse, rook, cannon
MATERIAL = (0, 10_000, 20, 20, 40, 90, 45, 10)


def _in_palace(row: int, col: int, color: int) -> bool:
    if col < 3 or col > 5:
        return False
    if color > 0:
        return 7 <= row <= 9
    return 0 <= row <= 2


def attacks_square(
    board: Sequence[int],
    source: int,
    target: int,
) -> bool:
    """Whether a piece geometrically attacks a target, even an allied piece.

    Considering allied pieces at the destination is essential to identify
    defenders: after an opponent captures that square, an allied piece may
    recapture. This follows the engine's horse-leg, elephant-eye, cannon-screen,
    palace, river and pawn-forward geometry.
    """
    if source == target:
        return False
    piece = board[source]
    if piece == 0:
        return False

    kind = abs(piece)
    color = 1 if piece > 0 else -1
    row, col = divmod(source, 9)
    target_row, target_col = divmod(target, 9)
    dr = target_row - row
    dc = target_col - col

    if kind == 1:
        return (
            abs(dr) + abs(dc) == 1
            and _in_palace(target_row, target_col, color)
        )

    if kind == 2:
        return (
            abs(dr) == 1
            and abs(dc) == 1
            and _in_palace(target_row, target_col, color)
        )

    if kind == 3:
        if abs(dr) != 2 or abs(dc) != 2:
            return False
        if color > 0 and target_row < 5:
            return False
        if color < 0 and target_row > 4:
            return False
        eye = (row + dr // 2) * 9 + col + dc // 2
        return board[eye] == 0

    if kind == 4:
        if abs(dr) == 2 and abs(dc) == 1:
            leg = source + (9 if dr > 0 else -9)
            return board[leg] == 0
        if abs(dr) == 1 and abs(dc) == 2:
            leg = source + (1 if dc > 0 else -1)
            return board[leg] == 0
        return False

    if kind in (5, 6):
        if dr != 0 and dc != 0:
            return False
        if dr == 0:
            step = 1 if dc > 0 else -1
            distance = abs(dc)
        else:
            step = 9 if dr > 0 else -9
            distance = abs(dr)
        screens = 0
        for n in range(1, distance):
            if board[source + n * step] != 0:
                screens += 1
                if screens > 1:
                    return False
        if kind == 5:
            return screens == 0
        return screens == 1

    if kind == 7:
        forward = -1 if color > 0 else 1
        if dr == forward and dc == 0:
            return True
        crossed = row <= 4 if color > 0 else row >= 5
        return crossed and dr == 0 and abs(dc) == 1

    return False


def threat_score(board: Sequence[int], side_to_move: int) -> int:
    """Signed RED-perspective score for threatened major/minor pieces.

    An attacked undefended piece incurs a larger penalty, particularly if the
    attacker moves next. A defended piece incurs a smaller penalty only when
    the cheapest attacking piece is worth *less* than its target. This avoids
    penalizing a rook that can trade a cannon but is itself recapturable, etc.

    Scoring only horses, rooks and cannons keeps the leaf evaluation bounded.
    This term is a risk estimate, not an instruction to refuse all exchanges.
    """
    pieces = [(square, piece) for square, piece in enumerate(board) if piece]
    red_score = 0

    for target, victim in pieces:
        kind = abs(victim)
        if kind not in THREATENED_KINDS:
            continue

        victim_color = 1 if victim > 0 else -1
        victim_value = MATERIAL[kind]
        least_attacker = 0

        for source, piece in pieces:
            if piece * victim >= 0:
                continue
            if attacks_square(board, source, target):
                value = MATERIAL[abs(piece)]
                if least_attacker == 0 or value < least_attacker:
                    least_attacker = value
                if least_attacker == 10:
                    break

        if least_attacker == 0:
            continue

        defended = False
        for source, piece in pieces:
            if source == target or piece * victim <= 0:
                continue
            if attacks_square(board, source, target):
                defended = True
                break

        urgent = side_to_move != victim_color

        if defended:
            # A low-value attacker may win an unfavorable exchange despite
            # nominal recapture; equal / more expensive attackers are safe
            # enough that we don't punish the protected piece.
            disadvantage = max(0, victim_value - least_attacker)
            penalty = disadvantage * (55 if urgent else 20) // 100
        else:
            # A completely loose cannon, horse or rook can be taken without
            # immediate recapture. It can still escape if its owner moves next.
            penalty = victim_value * (65 if urgent else 25) // 100

        red_score -= victim_color * penalty

    return red_score
