"""Opening book v3: >= 50 balanced, distinct, legal openings; v1/v2 untouched; recipe reproducible."""

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from src.agents.alphabeta_agent import AlphaBetaAgent
from src.agents.position_evaluator import evaluate_position, material_score
from src.environment import GameEngine, GameState, Side
from src.evaluation import tournament as t
from src.training.opening_book import (
    BOOK_VERSION,
    OPENING_BOOKS,
    OPENING_LINES_V1,
    OPENING_LINES_V2,
    OPENING_LINES_V3,
    get_opening,
    validate_opening_book,
)
from src.training.wxf import notate_line, replay_notation

V3 = "xiangqi-curated-v3"

# Published books never change. If one of these fails, a book was edited in place: add a new
# version instead. (v1's digest is also pinned in test_opening_book_v2.)
V1_SHA256 = "6e5a3f954f197a979d46ca2e8f426fc1d3d377c6e10b2251cb1d95b8e8bfa94d"
V2_SHA256 = "645a08629a4c1d8f394a50a0258df0a88b91f7cc203eb9519838e1111ce6c547"
V3_SHA256 = "c5b5e97f525a99993dc8cbacb6eccdb270859d53f776f2ee9b0142dba521f7e1"

MIN_OPENINGS = 50
BALANCE_THRESHOLD = 30  # |evaluate_position(final, RED)|, see docs/OPENING_BOOK_V3.md


def digest(lines) -> str:
    payload = [
        [line.opening_id, line.family, line.name, [[move.source, move.target] for move in line.moves]]
        for line in lines
    ]
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def final_state(line) -> GameState:
    state = GameState.initial()
    for move in line.moves:
        state = state.moved(move)
    return state


class PublishedBooksAreFrozenTests(unittest.TestCase):
    def test_v1_v2_and_v3_are_pinned(self):
        self.assertEqual(digest(OPENING_LINES_V1), V1_SHA256)
        self.assertEqual(digest(OPENING_LINES_V2), V2_SHA256)
        self.assertEqual(digest(OPENING_LINES_V3), V3_SHA256)

    def test_registry_and_default_version(self):
        self.assertIs(OPENING_BOOKS[V3], OPENING_LINES_V3)
        self.assertEqual(BOOK_VERSION, "xiangqi-curated-v2")  # batch generation default is unchanged


class BookV3Tests(unittest.TestCase):
    def test_at_least_50_openings_with_unique_ids(self):
        self.assertGreaterEqual(len(OPENING_LINES_V3), MIN_OPENINGS)
        ids = [line.opening_id for line in OPENING_LINES_V3]
        self.assertEqual(len(set(ids)), len(ids))
        for opening_id in ids:
            self.assertEqual(get_opening(opening_id, V3).opening_id, opening_id)

    def test_every_move_is_legal_and_no_line_ends_the_game(self):
        self.assertEqual(validate_opening_book(version=V3), len(OPENING_LINES_V3))

    def test_no_two_openings_share_a_final_position(self):
        finals = Counter(final_state(line).position_key() for line in OPENING_LINES_V3)
        self.assertEqual([key for key, count in finals.items() if count > 1], [])
        self.assertEqual(len(finals), len(OPENING_LINES_V3))

    def test_openings_are_balanced(self):
        """Equal material (quiet lines) and a small static score with Red to move."""
        scores = []
        for line in OPENING_LINES_V3:
            state = final_state(line)
            self.assertIs(state.side_to_move, Side.RED, line.opening_id)  # same tempo for every line
            self.assertEqual(material_score(state, Side.RED), 0, line.opening_id)
            score = evaluate_position(state, Side.RED)
            self.assertLessEqual(abs(score), BALANCE_THRESHOLD, line.opening_id)
            scores.append(score)
        self.assertLessEqual(max(abs(score) for score in scores), BALANCE_THRESHOLD)

    def test_lines_are_quiet_and_have_wxf_notation_that_round_trips(self):
        for line in OPENING_LINES_V3:
            state = GameState.initial()
            for move in line.moves:
                self.assertEqual(state.board[move.target], 0, line.opening_id)  # no capture
                state = state.moved(move)
            notation = line.notation or notate_line(line.moves)
            self.assertEqual(replay_notation(notation), line.moves, line.opening_id)

    def test_v3_reuses_v2_lines_unchanged_and_extends_the_rest(self):
        v2 = {line.opening_id: line for line in OPENING_LINES_V2}
        extended = [line for line in OPENING_LINES_V3 if line.opening_id.endswith("_x")]
        plain = [line for line in OPENING_LINES_V3 if not line.opening_id.endswith("_x")]
        self.assertTrue(extended and plain)
        for line in plain:
            self.assertEqual(line, v2[line.opening_id])
        for line in extended:
            base = v2[line.opening_id[:-2]]
            self.assertEqual(line.moves[:len(base.moves)], base.moves)
            self.assertEqual(len(line.moves), len(base.moves) + 2)  # one Red + one Black move
            self.assertEqual(line.family, base.family)

    def test_families_are_all_represented(self):
        families = {line.family for line in OPENING_LINES_V3}
        self.assertEqual(families, {line.family for line in OPENING_LINES_V2})

    def test_shipped_configs_use_the_whole_book_for_100_games_per_pair(self):
        for name in ("tournament_default.yaml", "tournament_equal_time.yaml"):
            raw = t.load_raw_config(t.BACKEND_ROOT / "configs" / name)
            self.assertEqual(raw["opening_book_version"], V3)
            self.assertEqual(raw["openings"], [line.opening_id for line in OPENING_LINES_V3], name)
            self.assertEqual(len(raw["openings"]) * raw.get("colors_per_opening", 2), 100)


class RecipeIsReproducibleTests(unittest.TestCase):
    """The frozen data must equal what scripts/generate_opening_book_v3.py derives (spot check)."""

    def test_recipe_reproduces_the_frozen_extensions_for_sample_lines(self):
        from scripts import generate_opening_book_v3 as recipe
        from src.training import opening_book

        agent = AlphaBetaAgent(GameEngine(), max_depth=recipe.SEARCH_DEPTH)
        frozen = dict(opening_book._V3_EXTENSIONS)
        bases = {line.opening_id: line for line in recipe.base_lines()}
        self.assertEqual(tuple(bases), opening_book._V3_BASE_IDS)  # same base selection, same order
        for opening_id in ("central_cannon_screen_horses", "pawn_left", "cross_palace_right"):
            self.assertEqual(" ".join(recipe.extension(agent, bases[opening_id])), frozen[opening_id])

    def test_documentation_exists(self):
        doc = (Path(t.BACKEND_ROOT) / "docs" / "OPENING_BOOK_V3.md").read_text(encoding="utf-8")
        self.assertIn(str(BALANCE_THRESHOLD), doc)
        for line in OPENING_LINES_V3:
            self.assertIn(f"`{line.opening_id}`", doc)


if __name__ == "__main__":
    unittest.main()
