"""Book completeness, legal replay, deterministic diversity and batch handoff."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.environment import GameEngine, GameState, Move
from src.training.batch_games import generate_batch, play_game
from src.training.opening_book import (
    BOOK_VERSION,
    OPENING_LINES,
    sample_opening,
    validate_opening_book,
)


class OpeningBookTests(unittest.TestCase):
    def test_every_opening_is_legal_and_all_families_covered(self):
        self.assertEqual(validate_opening_book(), len(OPENING_LINES))
        self.assertGreaterEqual(len(OPENING_LINES), 10)
        self.assertEqual(
            {line.family for line in OPENING_LINES},
            {
                "central_cannon", "advance_pawn", "flying_elephant",
                "develop_horse", "palace_cannon",
            },
        )

    def test_sample_is_deterministic_and_varied(self):
        self.assertEqual(sample_opening(42, 6), sample_opening(42, 6))
        families = Counter(sample_opening(seed, 6).line.family for seed in range(200))
        self.assertEqual(len(families), 5)
        self.assertTrue(all(count >= 20 for count in families.values()))

    def test_valid_prefixes_1_to_6(self):
        engine = GameEngine()
        for line in OPENING_LINES:
            for length in range(1, 7):
                state = GameState.initial()
                for action in line.moves[:length]:
                    self.assertIn(action, engine.legal_moves(state), line.opening_id)
                    state = state.moved(action)

    def test_handoff_and_provenance(self):
        record = play_game(
            "alphabeta", "random", game_index=7, seed=42,
            red_depth=1, max_plies=8, opening_mode="book", opening_plies=6,
        )
        self.assertEqual(record["opening_mode"], "book")
        self.assertEqual(record["opening_book_version"], BOOK_VERSION)
        self.assertIn(record["opening_id"], {line.opening_id for line in OPENING_LINES})
        self.assertEqual(record["opening_applied_plies"], 6)
        self.assertEqual(
            [row["actor"] for row in record["moves"][:6]],
            ["opening_book"] * 6,
        )
        self.assertEqual(record["moves"][6]["actor"], "alphabeta")
        self.assertEqual(record["moves"][7]["actor"], "random")
        engine = GameEngine()
        state = GameState.initial()
        for row in record["moves"]:
            action = Move(**row["action"])
            self.assertIn(action, engine.legal_moves(state))
            self.assertEqual(row["board"], list(state.board))
            state = state.moved(action)
        self.assertEqual(record["final_board"], list(state.board))

    def test_book_mode_does_not_generate_extra_random_moves(self):
        record = play_game(
            "random", "random", max_plies=4,
            opening_mode="book", opening_plies=8,
        )
        self.assertEqual(record["opening_applied_plies"], 4)
        self.assertTrue(all(row["actor"] == "opening_book" for row in record["moves"]))

    def test_resume_book_mode_and_reject_changed_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.jsonl"
            first = generate_batch(
                "random", "random", games=3, output=path, seed=11,
                max_plies=6, opening_mode="book", opening_plies=4, quiet=True,
            )
            self.assertEqual(first["games"], 3)
            second = generate_batch(
                "random", "random", games=5, output=path, seed=11,
                max_plies=6, opening_mode="book", opening_plies=4,
                resume=True, quiet=True,
            )
            self.assertEqual(second["new_games"], 2)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["game_index"] for row in rows}, set(range(5)))
            self.assertTrue(all(row["opening_mode"] == "book" for row in rows))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                generate_batch(
                    "random", "random", games=5, output=path, seed=11,
                    max_plies=6, opening_mode="random", opening_plies=4,
                    resume=True, quiet=True,
                )

    def test_no_book_mode_and_invalid_configuration(self):
        record = play_game(
            "random", "random", max_plies=2,
            opening_mode="none", opening_plies=6,
        )
        self.assertEqual(record["opening_applied_plies"], 0)
        self.assertEqual(record["opening_id"], None)
        self.assertTrue(all(row["actor"] == "random" for row in record["moves"]))
        with self.assertRaises(ValueError):
            play_game("random", "random", opening_mode="book", opening_plies=0)


if __name__ == "__main__":
    unittest.main()
