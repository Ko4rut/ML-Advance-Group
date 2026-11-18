"""Deterministic recipe behind opening book ``xiangqi-curated-v3`` (run from project/backend).

    python -m scripts.generate_opening_book_v3            # print the frozen table + selection report
    python -m scripts.generate_opening_book_v3 --check    # exit 1 if opening_book.py differs
    python -m scripts.generate_opening_book_v3 --doc      # (re)write docs/OPENING_BOOK_V3.md

Why v3 exists: the round-robin needs >= 50 distinct openings (50 x 2 colors = 100
games per pair), v2 has only 36 lines (34 distinct final positions), and published
books are never edited in place. The recipe, with no randomness anywhere:

1. **Base lines** = the v2 lines, in book order, minus
   * lines whose final position repeats an earlier line (v1 ``horse_left`` /
     ``horse_right`` transpose to ``elephant_left`` / ``elephant_right``), and
   * lines that are not balanced: material difference != 0 or
     ``|evaluate_position(final, RED)| > BALANCE_THRESHOLD`` (Red to move).
2. **Extension** of each base line by ``EXTENSION_PLIES`` = 2 plies (one Red and
   one Black move, so Red is on move again exactly like in v2). Each move is the
   best-ranked move of a depth-2 alpha-beta search (``AlphaBetaAgent.rank_actions``,
   i.e. the project's own ``evaluate_position``) that is
   quiet (no capture), not a king move, not a retreat, not a sideways move onto an
   edge file, not the undo of the side's previous move, and expressible in WXF.
   The extended line is kept only if
   its final position is balanced (same rule as above), not a duplicate of any
   other candidate, and material stays equal.
3. **Selection** of exactly ``TARGET_LINES`` = 50 lines: all base lines, plus the
   extended lines taken family by family in round-robin order (families in
   alphabetical order, lines inside a family in book order) until 50 are reached.

The result is frozen as data in ``src/training/opening_book.py``
(``_V3_BASE_IDS`` and ``_V3_EXTENSIONS``); this script only documents and audits it.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from src.agents.alphabeta_agent import AlphaBetaAgent
from src.agents.position_evaluator import evaluate_position, material_score
from src.environment import GameEngine, GameState, Move, Side
from src.training.opening_book import OPENING_LINES_V1, OPENING_LINES_V2, OpeningLine
from src.training.wxf import notate_line, to_wxf

BALANCE_THRESHOLD = 30      # |static eval, Red's view| in evaluate_position units (horse = 40, pawn = 10)
EXTENSION_PLIES = 2
TARGET_LINES = 50
SEARCH_DEPTH = 2

DOC_PATH = Path("docs/OPENING_BOOK_V3.md")


def final_state(moves) -> GameState:
    state = GameState.initial()
    for move in moves:
        state = state.moved(move)
    return state


def balance(state: GameState) -> tuple[int, int]:
    """(material difference, static evaluation), both from Red's point of view."""
    return material_score(state, Side.RED), evaluate_position(state, Side.RED)


def is_balanced(state: GameState) -> bool:
    material, score = balance(state)
    return material == 0 and abs(score) <= BALANCE_THRESHOLD


def base_lines() -> list[OpeningLine]:
    """v2 lines in book order, without transposition duplicates and unbalanced finals."""
    seen: set = set()
    kept: list[OpeningLine] = []
    for line in OPENING_LINES_V2:
        state = final_state(line.moves)
        key = state.position_key()
        if key in seen or not is_balanced(state):
            continue
        seen.add(key)
        kept.append(line)
    return kept


def _notatable(state: GameState, move: Move) -> str | None:
    try:
        return to_wxf(state, move)
    except (NotImplementedError, ValueError):
        return None


def pick_move(agent: AlphaBetaAgent, state: GameState, previous_own: Move | None) -> tuple[Move, str] | None:
    """Best-ranked admissible move for the side to move (see module docstring, step 2)."""
    ranked, _ = agent.rank_actions(state)
    side = state.side_to_move
    for move, _score in ranked:
        if state.board[move.target] != 0:
            continue  # quiet moves only
        if abs(state.board[move.source]) == 1:
            continue  # no king moves
        if previous_own is not None and move.source == previous_own.target and move.target == previous_own.source:
            continue  # never undo the previous move
        source_row, target_row = move.source // 9, move.target // 9
        if (target_row > source_row) if side is Side.RED else (target_row < source_row):
            continue  # no retreats (Red advances toward row 0, Black toward row 9)
        source_col, target_col = move.source % 9, move.target % 9
        if target_col in (0, 8) and source_col != target_col:
            continue  # no sideways drift onto the edge files (the evaluator likes it, players do not)
        text = _notatable(state, move)
        if text is None:
            continue
        return move, text
    return None


def extension(agent: AlphaBetaAgent, line: OpeningLine) -> tuple[str, ...] | None:
    """The two WXF moves appended to ``line``, or None if no admissible balanced pair exists."""
    state = final_state(line.moves)
    previous_red, previous_black = line.moves[-2], line.moves[-1]
    red = pick_move(agent, state, previous_red)
    if red is None:
        return None
    after_red = state.moved(red[0])
    black = pick_move(agent, after_red, previous_black)
    if black is None:
        return None
    result = after_red.moved(black[0])
    if not is_balanced(result):
        return None
    return red[1], black[1]


def build_selection() -> tuple[list[OpeningLine], list[tuple[str, tuple[str, ...]]], dict[str, str]]:
    """Return (base lines, [(base id, extension WXF)] in book order, skipped {base id: reason})."""
    from src.training.wxf import replay_notation

    engine = GameEngine()
    agent = AlphaBetaAgent(engine, max_depth=SEARCH_DEPTH)
    bases = base_lines()
    taken = {final_state(line.moves).position_key() for line in bases}
    candidates: list[tuple[OpeningLine, tuple[str, ...]]] = []
    skipped: dict[str, str] = {}
    for line in bases:
        ext = extension(agent, line)
        if ext is None:
            skipped[line.opening_id] = "no admissible balanced extension"
            continue
        notation = notate_line(line.moves) + ext
        key = final_state(replay_notation(notation)).position_key()
        if key in taken:
            skipped[line.opening_id] = "extension transposes into another line"
            continue
        taken.add(key)
        candidates.append((line, ext))

    # Round-robin over families (alphabetical), book order inside a family.
    by_family: dict[str, list[tuple[OpeningLine, tuple[str, ...]]]] = defaultdict(list)
    for line, ext in candidates:
        by_family[line.family].append((line, ext))
    chosen: list[tuple[OpeningLine, tuple[str, ...]]] = []
    need = TARGET_LINES - len(bases)
    queues = {family: list(items) for family, items in sorted(by_family.items())}
    while len(chosen) < need and any(queues.values()):
        for family in list(queues):
            if queues[family] and len(chosen) < need:
                chosen.append(queues[family].pop(0))
    order = {line.opening_id: index for index, line in enumerate(bases)}
    chosen.sort(key=lambda item: order[item[0].opening_id])
    for line, _ in candidates:
        if line.opening_id not in {c[0].opening_id for c in chosen}:
            skipped[line.opening_id] = "not selected (target size reached)"
    return bases, [(line.opening_id, ext) for line, ext in chosen], skipped


def render_python(bases, extensions) -> str:
    out = ["_V3_BASE_IDS: tuple[str, ...] = ("]
    out += [f'    "{line.opening_id}",' for line in bases]
    out += [")", "", "_V3_EXTENSIONS: tuple[tuple[str, str], ...] = ("]
    out += [f'    ("{base_id}", "{" ".join(ext)}"),' for base_id, ext in extensions]
    out += [")"]
    return "\n".join(out)


def check_against_book(bases, extensions) -> list[str]:
    from src.training import opening_book

    problems = []
    if tuple(line.opening_id for line in bases) != opening_book._V3_BASE_IDS:
        problems.append("_V3_BASE_IDS differs from the recipe")
    expected = tuple((base_id, " ".join(ext)) for base_id, ext in extensions)
    if expected != opening_book._V3_EXTENSIONS:
        problems.append("_V3_EXTENSIONS differs from the recipe")
    return problems


HEADER = """# Opening book v3 (`xiangqi-curated-v3`)

Generated by `python -m scripts.generate_opening_book_v3 --doc`. Do not edit by hand.

Why: the round-robin needs at least **50 distinct openings** (50 x 2 colors = **100 games per
pair**). v2 has 36 lines (34 distinct final positions) and published books are immutable, so v3
is a new book. Repeating openings is not an option: deterministic agents would replay identical
games, which adds no statistical information.

## Recipe (deterministic, no randomness)

1. **Base lines**: every v2 line in book order, except (a) lines whose final position equals an
   earlier line's (v1 `horse_left`/`horse_right` transpose into `elephant_left`/`elephant_right`)
   and (b) lines that are not balanced. v3 therefore has no duplicate final positions at all.
2. **Extension**: each base line is extended by 2 plies (one Red move, one Black move, so Red is on
   move at the end exactly as in v2). Each move is the best-ranked move of a depth-{depth}
   alpha-beta search (`AlphaBetaAgent.rank_actions`, the project's own `evaluate_position`) that is
   quiet (no capture), not a king move, not a retreat, not a sideways move onto an edge file, not the
   undo of that side's previous move and expressible in WXF. The extended line is kept only if its
   final position is balanced and new.
3. **Selection**: all base lines plus extended lines, family by family in round-robin order
   (alphabetical families, book order inside a family), until exactly {target} lines.

## Balance criterion

For every line the final position (Red to move) must satisfy:

* material difference `material_score(final, RED) == 0` (every line is capture-free), and
* `|evaluate_position(final, RED)| <= {threshold}`.

`evaluate_position` values: horse 40, cannon 45, rook 90, pawn 10 (+ small positional terms), so
{threshold} is less than one horse, and it admits the usual small tempo edge of the side to move. The
largest |score| among the {target} selected lines is **{largest}**. The v2 lines that fail this
test (`screen_horses_c58`: 55, `cross_palace_left`: 49) are not used.

The frozen data lives in `src/training/opening_book.py` (`_V3_BASE_IDS`, `_V3_EXTENSIONS`);
`tests/test_opening_book_v3.py` pins the book's SHA-256 and re-checks every criterion, and
`python -m scripts.generate_opening_book_v3 --check` re-runs the recipe and compares it with the data.
"""


def build_doc() -> str:
    from src.training.opening_book import OPENING_LINES_V3

    largest = max(abs(balance(final_state(line.moves))[1]) for line in OPENING_LINES_V3)
    out = [HEADER.format(depth=SEARCH_DEPTH, target=TARGET_LINES, threshold=BALANCE_THRESHOLD, largest=largest)]
    families: dict[str, int] = defaultdict(int)
    for line in OPENING_LINES_V3:
        families[line.family] += 1
    out.append(f"Total lines: **{len(OPENING_LINES_V3)}** "
               f"({sum(1 for line in OPENING_LINES_V3 if not line.opening_id.endswith('_x'))} from v2, "
               f"{sum(1 for line in OPENING_LINES_V3 if line.opening_id.endswith('_x'))} extended).\n")
    out += ["## Families\n", "| family | lines |", "|---|---|"]
    out += [f"| `{family}` | {count} |" for family, count in sorted(families.items())]
    out += ["", "## All lines\n",
            "| opening_id | family | plies | WXF moves | material | static eval (Red) |",
            "|---|---|---|---|---|---|"]
    v1_ids = {line.opening_id for line in OPENING_LINES_V1}
    for line in OPENING_LINES_V3:
        notation = line.notation or notate_line(line.moves)
        material, score = balance(final_state(line.moves))
        marker = " (v1)" if line.opening_id in v1_ids else ""
        out.append(f"| `{line.opening_id}`{marker} | `{line.family}` | {len(line.moves)} | "
                   f"`{' '.join(notation)}` | {material} | {score:+d} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare the recipe with the frozen data")
    parser.add_argument("--doc", action="store_true", help=f"write {DOC_PATH}")
    args = parser.parse_args(argv)
    bases, extensions, skipped = build_selection()
    if args.check:
        problems = check_against_book(bases, extensions)
        for problem in problems:
            print("MISMATCH:", problem)
        print("recipe matches the frozen data" if not problems else "run without --check to see the new table")
        return 1 if problems else 0
    if args.doc:
        DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOC_PATH.write_text(build_doc(), encoding="utf-8")
        print(f"wrote {DOC_PATH}")
        return 0
    print(render_python(bases, extensions))
    print(f"\n# {len(bases)} base + {len(extensions)} extended = {len(bases) + len(extensions)} lines")
    for base_id, reason in skipped.items():
        print(f"# skipped extension of {base_id}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
