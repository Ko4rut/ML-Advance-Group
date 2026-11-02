"""Tests for the terminal batch data collector.

Run from project/backend: python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.training.batch_games import generate_batch, play_game


class BatchGamesTests(unittest.TestCase):
    def test_record_contains_replayable_legal_moves(self):
        record = play_game(
            "random", "random",
            seed=7,
            max_plies=6,
            opening_plies=2,
        )
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(len(record["initial_board"]), 90)
        self.assertLessEqual(record["ply_count"], 6)
        self.assertEqual(len(record["moves"]), record["ply_count"])

        from src.environment import XiangqiEnv, Move

        env = XiangqiEnv()
        for row in record["moves"]:
            self.assertEqual(list(env.state.board), row["board"])
            action = row["action"]
            move = Move(action["source"], action["target"])
            self.assertIn(move, env.get_legal_actions())
            env.step(move)
            self.assertEqual(list(env.state.board), row["next_board"])
        self.assertEqual(list(env.state.board), record["final_board"])

    def test_truncated_game_does_not_invent_winner(self):
        record = play_game("random", "random", seed=11, max_plies=1)
        self.assertEqual(record["outcome"]["status"], "truncated")
        self.assertIsNone(record["outcome"]["winner"])
        self.assertEqual(record["outcome"]["reason"], "max_plies")

    def test_jsonl_batch_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "games.jsonl"
            summary = generate_batch(
                "random", "random",
                games=2,
                output=path,
                max_plies=2,
                quiet=True,
            )
            self.assertEqual(summary["games"], 2)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(len(row["moves"]) <= 2 for row in records))
            with self.assertRaises(FileExistsError):
                generate_batch(
                    "random", "random",
                    games=1, output=path,
                    max_plies=2, quiet=True,
                )


if __name__ == "__main__":
    unittest.main()
