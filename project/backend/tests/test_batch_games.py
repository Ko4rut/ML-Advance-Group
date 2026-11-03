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


    def test_resume_skips_saved_indices_and_matches_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "resume.jsonl"
            first = generate_batch(
                "random", "random", games=2, output=path, seed=99,
                max_plies=2, quiet=True,
            )
            self.assertEqual(first["games"], 2)
            continued = generate_batch(
                "random", "random", games=4, output=path, seed=99,
                max_plies=2, resume=True, quiet=True,
            )
            self.assertEqual(continued["games"], 4)
            self.assertEqual(continued["new_games"], 2)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["game_index"] for row in rows], [0, 1, 2, 3])
            self.assertEqual([row["seed"] for row in rows], [99, 1108, 2117, 3126])
            self.assertTrue((Path(str(path) + ".checkpoint.json")).exists())
            no_op = generate_batch(
                "random", "random", games=4, output=path, seed=99,
                max_plies=2, resume=True, quiet=True,
            )
            self.assertEqual(no_op["new_games"], 0)

    def test_resume_rejects_changed_configuration_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "games.jsonl"
            generate_batch("random", "random", games=2, output=path,
                           max_plies=2, quiet=True)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                generate_batch("random", "random", games=2, output=path,
                               max_plies=3, resume=True, quiet=True)
            duplicate = path.read_text(encoding="utf-8").splitlines()[0]
            with path.open("a", encoding="utf-8") as stream:
                stream.write(duplicate + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                generate_batch("random", "random", games=2, output=path,
                               max_plies=2, resume=True, quiet=True)

    def test_resume_repairs_torn_last_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "games.jsonl"
            generate_batch("random", "random", games=1, output=path,
                           max_plies=2, quiet=True)
            with path.open("ab") as stream:
                stream.write(b'{"partial":')
            resumed = generate_batch("random", "random", games=2, output=path,
                                     max_plies=2, resume=True, quiet=True)
            self.assertEqual(resumed["new_games"], 1)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)

    def test_parallel_batch_has_exactly_one_record_per_index(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "parallel.jsonl"
            result = generate_batch(
                "random", "random", games=4, output=path, seed=101,
                max_plies=2, workers=2, quiet=True,
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["games"], 4)
            self.assertEqual({row["game_index"] for row in rows}, set(range(4)))


if __name__ == "__main__":
    unittest.main()
