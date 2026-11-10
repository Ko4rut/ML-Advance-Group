"""Small, curated and replay-validated Xiangqi opening book.

Both red and black moves are recorded. These are *opening templates*, not
learned policies or guarantees of best play. The game engine validates every
line with the project's own rules; Alpha-Beta resumes after the selected prefix.

Keep BOOK_VERSION updated when changing line moves or selection policy: the
version is recorded for reproducible batch checkpoint/resume.

Books are immutable once published. ``xiangqi-curated-v1`` is frozen (a unit
test pins its SHA-256) so old datasets stay interpretable; ``v2`` extends it
with longer lines for the round-robin tournament. Never edit a published
version in place: add a new one instead.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..environment import GameEngine, GameState, Move
from .wxf import replay_notation


@dataclass(frozen=True, slots=True)
class OpeningLine:
    opening_id: str
    family: str
    name: str
    moves: tuple[Move, ...]
    # Optional human-readable WXF notation (v2 lines); empty for frozen v1.
    notation: tuple[str, ...] = ()
    # Optional reviewer remark shown in docs/OPENING_BOOK_V2.md.
    note: str = ""


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
# FROZEN: do not edit (see module docstring); tests pin the SHA-256.
OPENING_LINES_V1: tuple[OpeningLine, ...] = (
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


def _wxf_line(opening_id: str, family: str, name: str, text: str, note: str = "") -> OpeningLine:
    """Build a line from space-separated WXF notation (Red first, alternating)."""
    notation = tuple(text.split())
    return OpeningLine(opening_id, family, name, replay_notation(notation), notation, note)


# v2 additions: 8-10 ply lines written in WXF. Chinese-style notation maps
# directly (e.g. 炮二平五 = C2=5, 马八进七 = H8+7). Theory is best-effort:
# every line is legal and quiet (no early sacrifices), but names/move orders
# should be reviewed by a Xiangqi player, see docs/OPENING_BOOK_V2.md.
_V2_NEW_LINES: tuple[OpeningLine, ...] = (
    # --- Central cannon vs screen horses (中炮对屏风马) -------------------------
    _wxf_line("screen_horses_c57", "screen_horses", "Pháo đầu - Bình phong mã, ngũ thất pháo",
              "C2=5 H2+3 H2+3 H8+7 R1=2 R9=8 C8=7 R1+1 H8+9 P7+1"),
    _wxf_line("screen_horses_c56", "screen_horses", "Pháo đầu - Bình phong mã, ngũ lục pháo",
              "C2=5 H2+3 H2+3 H8+7 R1=2 R9=8 C8=6 C8=9 H8+7 P3+1"),
    _wxf_line("screen_horses_c58", "screen_horses", "Pháo đầu - Bình phong mã, ngũ bát pháo",
              "C2=5 H2+3 H2+3 H8+7 R1=2 R9=8 H8+7 C8=9 R9=8 R8+4"),
    _wxf_line("screen_horses_c59", "screen_horses", "Pháo đầu - Bình phong mã, ngũ cửu pháo",
              "C2=5 H2+3 H2+3 H8+7 R1=2 R9=8 C8=9 P7+1 H8+7 P3+1"),
    _wxf_line("screen_horses_pawn7", "screen_horses", "Pháo đầu - Bình phong mã, tiến binh 7 xe tuần hà",
              "C2=5 H2+3 H2+3 H8+7 R1=2 R9=8 P7+1 P7+1 R2+4 C8=9"),
    # --- Central cannon vs reverse palace horse (中炮对反宫马) ------------------
    _wxf_line("reverse_palace_p7", "reverse_palace_horse", "Pháo đầu - Phản cung mã, tiến binh 7",
              "C2=5 C8=6 H2+3 H8+7 R1=2 H2+3 P7+1 R9=8 H8+7 P7+1"),
    _wxf_line("reverse_palace_p7b", "reverse_palace_horse", "Pháo đầu - Phản cung mã, xe cửu ra trước",
              "C2=5 C8=6 H2+3 H8+7 R1=2 H2+3 H8+7 R9=8 P7+1 P3+1"),
    _wxf_line("reverse_palace_mirror", "reverse_palace_horse", "Pháo đầu - Phản cung mã, pháo đen lộ 4",
              "C2=5 C2=4 H2+3 H2+3 R1=2 H8+7 H8+7 R9=8 P7+1 P7+1"),
    # --- Same-direction cannons (顺炮) ---------------------------------------
    _wxf_line("shun_cannon_straight_rook", "same_direction_cannon", "Thuận pháo - Xe thẳng đối xe ngang",
              "C2=5 C2=5 H2+3 H2+3 R1=2 R1=2 R2+6 H8+7 H8+7 R9=8"),
    _wxf_line("shun_cannon_rook_race", "same_direction_cannon", "Thuận pháo - Hai bên xuất xe sớm",
              "C2=5 C2=5 H2+3 H2+3 R1=2 H8+7 R2+4 R9=8 H8+7 R1=2"),
    # --- Opposed cannons (列炮) ----------------------------------------------
    _wxf_line("lie_cannon_p7", "opposed_cannon", "Liệt pháo - Tiến binh 7",
              "C2=5 C8=5 H2+3 H8+7 R1=2 R9=8 P7+1 H2+3 H8+7 R1=2"),
    _wxf_line("lie_cannon_rook_ride", "opposed_cannon", "Liệt pháo - Xe tuần hà",
              "C2=5 C8=5 H2+3 H8+7 R1=2 R9=8 R2+4 H2+3 H8+7 R1=2"),
    # --- Advance pawn vs pawn-bottom cannon (仙人指路对卒底炮) ------------------
    _wxf_line("pawn_vs_bottom_cannon_central", "pawn_vs_bottom_cannon", "Tiên nhân chỉ lộ - Tốt để pháo, pháo đầu",
              "P7+1 C2=3 C2=5 H8+7 H2+3 R9=8 R1=2 P7+1 H8+7 R1+1"),
    _wxf_line("pawn_vs_bottom_cannon_horses", "pawn_vs_bottom_cannon", "Tiên nhân chỉ lộ - Tốt để pháo, khởi mã",
              "P7+1 C2=3 H2+3 H8+7 R1=2 R9=8 C8=5 P7+1 H8+7 R1+1"),
    # --- Advance pawn vs pawn (仙人指路对兵) ---------------------------------
    _wxf_line("pawn_vs_pawn_c3", "pawn_vs_pawn", "Tiên nhân chỉ lộ - Đối binh, pháo lộ 3",
              "P7+1 P7+1 C2=3 C8=5 H8+7 H2+3 R9=8 R1=2 P3+1 P3+1"),
    _wxf_line("pawn_vs_pawn_horses", "pawn_vs_pawn", "Tiên nhân chỉ lộ - Đối binh, khởi mã",
              "P7+1 P7+1 H2+3 H8+7 C8=7 H2+3 R1=2 R9=8 H8+9 R1=2"),
    # --- Flying elephant (飞相局) ---------------------------------------------
    _wxf_line("elephant_vs_central_cannon", "flying_elephant", "Phi tượng cục - Đối pháo đầu",
              "E3+5 C8=5 H2+3 H8+7 R1=2 R9=8 P7+1 P7+1 H8+7 H2+3"),
    _wxf_line("elephant_vs_screen_setup", "flying_elephant", "Phi tượng cục - Đối bình phong mã",
              "E3+5 H8+7 H2+3 H2+3 R1=2 R9=8 C8=6 P7+1 H8+7 R1+1"),
    # --- Horse opening (起马局) -----------------------------------------------
    _wxf_line("horse_vs_central_cannon", "develop_horse", "Khởi mã cục - Đối pháo đầu",
              "H2+3 C8=5 H8+7 H8+7 R9=8 R9=8 P7+1 H2+3 R1+1 R1=2"),
    _wxf_line("horse_vs_cannon_left", "develop_horse", "Khởi mã cục - Mã trái, đối pháo đầu lộ 2",
              "H8+7 C2=5 H2+3 H2+3 R1=2 R1=2 P7+1 H8+7 R9=8 R9=8"),
    # --- Corner cannon (士角炮) -----------------------------------------------
    _wxf_line("corner_cannon_vs_central", "corner_cannon", "Sĩ giác pháo - Đối pháo đầu",
              "C2=4 C8=5 H8+7 H8+7 R9=8 R9=8 H2+3 H2+3 R1=2 R1=2",
              "Cùng dạng pháo sĩ giác với v1 palace_cannon_* (v1 đặt tên 'Quá cung pháo')."),
    _wxf_line("corner_cannon_left_vs_central", "corner_cannon", "Sĩ giác pháo cánh trái - Đối pháo đầu",
              "C8=6 C2=5 H2+3 H2+3 R1=2 R1=2 H8+7 H8+7 R9=8 R9=8"),
    # --- Cross-palace cannon (过宫炮) -----------------------------------------
    _wxf_line("cross_palace_right", "cross_palace_cannon", "Quá cung pháo - Pháo hữu qua cung",
              "C2=6 H8+7 H2+3 H2+3 R1=2 R9=8 P7+1 P7+1 H8+7 R1=2"),
    _wxf_line("cross_palace_left", "cross_palace_cannon", "Quá cung pháo - Pháo tả qua cung",
              "C8=4 H2+3 H2+3 H8+7 R1=2 R9=8 H8+7 R1+1 R9=8 P3+1"),
)

OPENING_LINES_V2: tuple[OpeningLine, ...] = OPENING_LINES_V1 + _V2_NEW_LINES

OPENING_BOOKS: dict[str, tuple[OpeningLine, ...]] = {
    "xiangqi-curated-v1": OPENING_LINES_V1,
    "xiangqi-curated-v2": OPENING_LINES_V2,
}
BOOK_VERSION = "xiangqi-curated-v2"
OPENING_LINES: tuple[OpeningLine, ...] = OPENING_BOOKS[BOOK_VERSION]


def opening_lines(version: str = BOOK_VERSION) -> tuple[OpeningLine, ...]:
    """Return the lines of a published book version."""
    try:
        return OPENING_BOOKS[version]
    except KeyError:
        known = ", ".join(sorted(OPENING_BOOKS))
        raise ValueError(f"Unknown opening book version {version!r}; known: {known}") from None


def get_opening(opening_id: str, version: str = BOOK_VERSION) -> OpeningLine:
    """Look up one line by id within a book version."""
    for line in opening_lines(version):
        if line.opening_id == opening_id:
            return line
    raise KeyError(f"Opening {opening_id!r} not found in {version}")


def sample_opening(game_seed: int, max_plies: int, version: str = BOOK_VERSION) -> OpeningSelection:
    """Choose one family then one line, deterministically for a game seed.

    `max_plies` limits only scripted plies. If the selected line is shorter,
    Alpha-Beta takes over immediately; it never fills the rest randomly.
    """
    if max_plies < 1:
        raise ValueError("book opening requires opening_plies >= 1")
    lines = opening_lines(version)
    rng = random.Random(game_seed + 7919)
    families = sorted({line.family for line in lines})
    family = rng.choice(families)
    options = [line for line in lines if line.family == family]
    line = rng.choice(options)
    return OpeningSelection(line=line, moves=line.moves[:max_plies])


def validate_opening_book(engine: GameEngine | None = None, version: str = BOOK_VERSION) -> int:
    """Replay every line through legal_moves and terminal checks.

    Raise immediately for malformed, duplicate or illegal book sequences.
    Used by unit tests and can be run before generating a large batch.
    """
    engine = engine or GameEngine()
    lines = opening_lines(version)
    ids: set[str] = set()
    for line in lines:
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
    return len(lines)
