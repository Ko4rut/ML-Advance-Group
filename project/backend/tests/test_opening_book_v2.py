"""Opening book v2: frozen v1, new lines, uniqueness, WXF helper and resume safety."""

import hashlib
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from src.environment import GameEngine, GameState, Move
from src.training.batch_games import generate_batch
from src.training.opening_book import (
    BOOK_VERSION,
    OPENING_BOOKS,
    OPENING_LINES,
    OPENING_LINES_V1,
    OPENING_LINES_V2,
    get_opening,
    sample_opening,
    validate_opening_book,
)
from src.training.wxf import notate_line, replay_notation, to_wxf, wxf

V1 = "xiangqi-curated-v1"
V2 = "xiangqi-curated-v2"

# SHA-256 of the v1 lines (id, family, name, [source, target] list). If this
# fails, v1 was edited: never do that, add a new book version instead.
V1_SHA256 = "6e5a3f954f197a979d46ca2e8f426fc1d3d377c6e10b2251cb1d95b8e8bfa94d"

# v1 is known to contain two pairs that transpose to the same final position.
V1_KNOWN_DUPLICATE_PAIRS = {
    frozenset({"elephant_left", "horse_left"}),
    frozenset({"elephant_right", "horse_right"}),
}


def _digest(lines) -> str:
    payload = [
        [line.opening_id, line.family, line.name,
         [[move.source, move.target] for move in line.moves]]
        for line in lines
    ]
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _final_groups(lines) -> dict:
    groups = defaultdict(list)
    for line in lines:
        state = GameState.initial()
        for action in line.moves:
            state = state.moved(action)
        groups[state.position_key()].append(line.opening_id)
    return groups


class FrozenV1Tests(unittest.TestCase):
    def test_v1_is_frozen(self):
        self.assertEqual(len(OPENING_LINES_V1), 12)
        self.assertEqual(_digest(OPENING_LINES_V1), V1_SHA256)

    def test_v1_sampling_is_unchanged_by_the_refactor(self):
        picks = [sample_opening(seed, 6, V1).line.opening_id for seed in range(6)]
        self.assertEqual(picks, [
            "elephant_left", "central_cannon_screen_horses", "palace_cannon_right",
            "palace_cannon_right", "elephant_left", "elephant_left",
        ])

    def test_v1_has_exactly_the_two_known_duplicate_final_positions(self):
        duplicates = {frozenset(ids) for ids in _final_groups(OPENING_LINES_V1).values()
                      if len(ids) > 1}
        self.assertEqual(duplicates, V1_KNOWN_DUPLICATE_PAIRS)
        self.assertEqual(len(_final_groups(OPENING_LINES_V1)), 10)


class BookV2Tests(unittest.TestCase):
    def test_registry_and_default(self):
        self.assertEqual(set(OPENING_BOOKS), {V1, V2})
        self.assertEqual(BOOK_VERSION, V2)
        self.assertIs(OPENING_LINES, OPENING_LINES_V2)

    def test_v2_is_legal_and_extends_v1_unchanged(self):
        self.assertEqual(validate_opening_book(version=V2), len(OPENING_LINES_V2))
        self.assertEqual(OPENING_LINES_V2[:len(OPENING_LINES_V1)], OPENING_LINES_V1)

    def test_v2_size_families_and_lengths(self):
        self.assertTrue(30 <= len(OPENING_LINES_V2) <= 40, len(OPENING_LINES_V2))
        families = Counter(line.family for line in OPENING_LINES_V2)
        self.assertTrue(all(count >= 2 for count in families.values()), families)
        new_lines = OPENING_LINES_V2[len(OPENING_LINES_V1):]
        self.assertTrue(all(8 <= len(line.moves) <= 10 for line in new_lines))
        self.assertTrue(all(len(line.notation) == len(line.moves) for line in new_lines))

    def test_v2_final_positions_are_distinct_apart_from_inherited_v1_pairs(self):
        duplicates = {frozenset(ids) for ids in _final_groups(OPENING_LINES_V2).values()
                      if len(ids) > 1}
        self.assertEqual(duplicates, V1_KNOWN_DUPLICATE_PAIRS)

    def test_new_lines_are_quiet(self):
        """No captures in the scripted plies: no early material is given away."""
        for line in OPENING_LINES_V2[len(OPENING_LINES_V1):]:
            state = GameState.initial()
            for action in line.moves:
                self.assertEqual(state.board[action.target], 0, line.opening_id)
                state = state.moved(action)

    def test_notation_matches_moves_and_round_trips(self):
        for line in OPENING_LINES_V2[len(OPENING_LINES_V1):]:
            self.assertEqual(replay_notation(line.notation), line.moves, line.opening_id)
            self.assertEqual(notate_line(line.moves), line.notation, line.opening_id)

    def test_get_opening_and_unknown_version(self):
        self.assertEqual(get_opening("pawn_left", V1).opening_id, "pawn_left")
        self.assertEqual(get_opening("screen_horses_c57").family, "screen_horses")
        with self.assertRaises(KeyError):
            get_opening("screen_horses_c57", V1)  # only exists in v2
        with self.assertRaisesRegex(ValueError, "Unknown opening book version"):
            validate_opening_book(version="xiangqi-curated-v9")
        with self.assertRaisesRegex(ValueError, "Unknown opening book version"):
            sample_opening(1, 6, "nope")


class WxfHelperTests(unittest.TestCase):
    def test_known_conversions(self):
        initial = GameState.initial()
        self.assertEqual(wxf(initial, "C8=5"), Move(7 * 9 + 1, 7 * 9 + 4))
        after = initial.moved(wxf(initial, "C8=5"))
        self.assertEqual(wxf(after, "H2+3"), Move(0 * 9 + 1, 2 * 9 + 2))

    def test_v1_round_trip(self):
        for line in OPENING_LINES_V1:
            self.assertEqual(replay_notation(notate_line(line.moves)), line.moves)

    def test_to_wxf_is_inverse_of_wxf(self):
        state = GameState.initial()
        for text in ("P7+1", "P7+1", "C2=3", "C8=5", "H8+7", "H2+3"):
            action = wxf(state, text)
            self.assertEqual(to_wxf(state, action), text)
            state = state.moved(action)

    def test_errors_are_explicit(self):
        initial = GameState.initial()
        with self.assertRaisesRegex(ValueError, "Unsupported WXF"):
            wxf(initial, "+C2=5")
        with self.assertRaisesRegex(ValueError, "No R on file"):
            wxf(initial, "R5=4")
        with self.assertRaisesRegex(ValueError, "Impossible H"):
            wxf(initial, "H2+5")
        with self.assertRaisesRegex(ValueError, "straight movers"):
            wxf(initial, "H2=3")

    def test_ambiguous_files_are_refused(self):
        # Two red pawns on file 5 after one advances? Build it by hand instead:
        board = [0] * 90
        board[9 * 9 + 4] = 1    # red king
        board[0 * 9 + 3] = -1   # black king
        board[5 * 9 + 4] = 7    # red pawn
        board[4 * 9 + 4] = 7    # second red pawn on the same file
        state = GameState(tuple(board), GameState.initial().side_to_move)
        with self.assertRaises(NotImplementedError):
            wxf(state, "P5+1")


class ResumeCompatibilityTests(unittest.TestCase):
    def test_resuming_a_v1_dataset_is_refused_not_silently_mixed(self):
        """A dataset recorded with book v1 must not resume against the v2 default."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.jsonl"
            kwargs = dict(
                seed=11, max_plies=6, opening_mode="book", opening_plies=4, quiet=True,
            )
            generate_batch("random", "random", games=2, output=path, **kwargs)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["opening_book_version"] == V2 for row in rows))
            for row in rows:  # pretend the dataset predates v2
                row["opening_book_version"] = V1
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incompatible opening_book_version"):
                generate_batch(
                    "random", "random", games=4, output=path, resume=True, **kwargs,
                )


class ApiFacingBehaviourTests(unittest.TestCase):
    def test_default_sampling_uses_v2_and_stays_legal(self):
        engine = GameEngine()
        for seed in range(60):
            selection = sample_opening(seed, 10)
            state = GameState.initial()
            for action in selection.moves:
                self.assertIn(action, engine.legal_moves(state))
                state = state.moved(action)


if __name__ == "__main__":
    unittest.main()
