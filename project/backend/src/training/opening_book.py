"""Small, curated and replay-validated Xiangqi opening book.

Both red and black moves are recorded. These are *opening templates*, not
learned policies or guarantees of best play. The game engine validates every
line with the project's own rules; Alpha-Beta resumes after the selected prefix.

Keep BOOK_VERSION updated when changing line moves or selection policy: the
version is recorded for reproducible batch checkpoint/resume.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..environment import GameEngine, GameState, Move


BOOK_VERSION = "xiangqi-curated-v1"


@dataclass(frozen=True, slots=True)
class OpeningLine:
    opening_id: str
    family: str
    name: str
    moves: tuple[Move, ...]


@dataclass(frozen=True, slots=True)
class OpeningSelection:
    line: OpeningLine
    moves: tuple[Move, ...]


def move(source_row: int, source_col: int, target_row: int, target_col: int) -> Move:
    """Build a zero-based (row, column) move; row 0 is Black's back rank."""
    return Move(source_row * 9 + source_col, target_row * 9 + target_col)


# Six-ply starting templates. Families are sampled uniformly before selecting
# one of their variants, preventing families with more variations from
# accidentally dominating the dataset. All moves are verified in tests.
OPENING_LINES: tuple[OpeningLine, ...] = (
    OpeningLine(
        "central_cannon_screen_horses", "central_cannon",
        "Pháo đầu - Bình phong mã",
        (
            move(7, 1, 7, 4), move(0, 1, 2, 2),
            move(9, 1, 7, 2), move(0, 7, 2, 6),
            move(9, 0, 9, 1), move(0, 0, 0, 1),
        ),
    ),
    OpeningLine(
        "central_cannon_counter_cannon", "central_cannon",
        "Pháo đầu - Đối pháo",
        (
            move(7, 1, 7, 4), move(2, 1, 2, 4),
            move(9, 1, 7, 2), move(0, 1, 2, 2),
            move(9, 0, 9, 1), move(0, 0, 0, 1),
        ),
    ),
    OpeningLine(
        "central_cannon_right", "central_cannon",
        "Pháo đầu cánh phải",
        (
            move(7, 7, 7, 4), move(0, 7, 2, 6),
            move(9, 7, 7, 6), move(0, 1, 2, 2),
            move(9, 8, 9, 7), move(0, 8, 0, 7),
        ),
    ),
    OpeningLine(
        "pawn_left", "advance_pawn",
        "Tiên nhân chỉ lộ - Chốt cánh trái",
        (
            move(6, 2, 5, 2), move(3, 2, 4, 2),
            move(9, 1, 7, 2), move(0, 1, 2, 2),
            move(9, 0, 9, 1), move(0, 0, 0, 1),
        ),
    ),
    OpeningLine(
        "pawn_center", "advance_pawn",
        "Tiến chốt trung lộ",
        (
            move(6, 4, 5, 4), move(3, 4, 4, 4),
            move(9, 1, 7, 2), move(0, 7, 2, 6),
            move(9, 2, 7, 4), move(0, 6, 2, 4),
        ),
    ),
    OpeningLine(
        "pawn_right", "advance_pawn",
        "Tiến chốt cánh phải",
        (
            move(6, 6, 5, 6), move(3, 6, 4, 6),
            move(9, 7, 7, 6), move(0, 7, 2, 6),
            move(9, 8, 9, 7), move(0, 8, 0, 7),
        ),
    ),
    OpeningLine(
        "elephant_left", "flying_elephant",
        "Phi tượng cánh trái",
        (
            move(9, 2, 7, 4), move(0, 2, 2, 4),
            move(9, 1, 7, 2), move(0, 1, 2, 2),
            move(9, 0, 9, 1), move(0, 0, 0, 1),
        ),
    ),
    OpeningLine(
        "elephant_right", "flying_elephant",
        "Phi tượng cánh phải",
        (
            move(9, 6, 7, 4), move(0, 6, 2, 4),
            move(9, 7, 7, 6), move(0, 7, 2, 6),
            move(9, 8, 9, 7), move(0, 8, 0, 7),
        ),
    ),
    OpeningLine(
        "horse_left", "develop_horse",
        "Khởi mã cánh trái",
        (
            move(9, 1, 7, 2), move(0, 1, 2, 2),
            move(9, 2, 7, 4), move(0, 2, 2, 4),
            move(9, 0, 9, 1), move(0, 0, 0, 1),
        ),
    ),
    OpeningLine(
        "horse_right", "develop_horse",
        "Khởi mã cánh phải",
        (
            move(9, 7, 7, 6), move(0, 7, 2, 6),
            move(9, 6, 7, 4), move(0, 6, 2, 4),
            move(9, 8, 9, 7), move(0, 8, 0, 7),
        ),
    ),
    OpeningLine(
        "palace_cannon_left", "palace_cannon",
        "Quá cung pháo cánh trái",
        (
            move(7, 1, 7, 3), move(0, 1, 2, 2),
            move(9, 1, 7, 2), move(0, 7, 2, 6),
            move(9, 0, 9, 1), move(0, 0, 0, 1),
        ),
    ),
    OpeningLine(
        "palace_cannon_right", "palace_cannon",
        "Quá cung pháo cánh phải",
        (
            move(7, 7, 7, 5), move(0, 7, 2, 6),
            move(9, 7, 7, 6), move(0, 1, 2, 2),
            move(9, 8, 9, 7), move(0, 8, 0, 7),
        ),
    ),
)


def sample_opening(game_seed: int, max_plies: int) -> OpeningSelection:
    """Choose one family then one line, deterministically for a game seed.

    `max_plies` limits only scripted plies. If the selected line is shorter,
    Alpha-Beta takes over immediately; it never fills the rest randomly.
    """
    if max_plies < 1:
        raise ValueError("book opening requires opening_plies >= 1")
    rng = random.Random(game_seed + 7919)
    families = sorted({line.family for line in OPENING_LINES})
    family = rng.choice(families)
    options = [line for line in OPENING_LINES if line.family == family]
    line = rng.choice(options)
    return OpeningSelection(line=line, moves=line.moves[:max_plies])


def validate_opening_book(engine: GameEngine | None = None) -> int:
    """Replay every line through legal_moves and terminal checks.

    Raise immediately for malformed, duplicate or illegal book sequences.
    Used by unit tests and can be run before generating a large batch.
    """
    engine = engine or GameEngine()
    ids: set[str] = set()
    for line in OPENING_LINES:
        if not line.opening_id or not line.family or not line.moves:
            raise ValueError(f"Incomplete opening line: {line!r}")
        if line.opening_id in ids:
            raise ValueError(f"Duplicate opening id: {line.opening_id}")
        ids.add(line.opening_id)
        state = GameState.initial()
        for ply, action in enumerate(line.moves, start=1):
            choices = engine.legal_moves(state)
            if engine.outcome(state, legal_moves=choices) is not None:
                raise ValueError(f"Terminal book line: {line.opening_id}, ply={ply}")
            if action not in choices:
                raise ValueError(
                    f"Illegal book move: {line.opening_id}, ply={ply}, "
                    f"{action.source}->{action.target}"
                )
            state = state.moved(action)
    return len(OPENING_LINES)
