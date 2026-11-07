"""Integration tests using the actual repository GameEngine."""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.silver.reader import SilverGameReader, SilverTransitionReader
from src.data.silver.data_pipeline import SilverPipeline
from src.environment import GameEngine, GameState, Move


class SilverPipelineTests(unittest.TestCase):
    def sample_game(self, *, actor="opening_book"):
        engine = GameEngine()
        start = GameState.initial()
        action = Move(54, 45)  # Red pawn moves one square forward.
        self.assertIn(action, engine.legal_moves(start))
        after = engine.apply_move(start, action)
        return {
            "schema_version": 1, "game_id": "raw-123",
            "initial_board": list(start.board), "initial_side": "red",
            "red_agent": "alphabeta", "black_agent": "alphabeta",
            "moves": [{
                "ply": 1, "board": list(start.board), "side": "red", "actor": actor,
                "action": {"source": action.source, "target": action.target},
                "next_board": list(after.board), "terminated": False, "reward": 0,
            }],
            "outcome": {"status": "truncated", "winner": None, "reason": "max_plies"},
            "final_board": list(after.board), "ply_count": 1,
        }

    def write_records(self, folder: Path, name: str, records: list[object]) -> Path:
        path = folder / name
        path.write_text(
            "\n".join(r if isinstance(r, str) else json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )
        return path

    def read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


    def test_sample_limit_preserves_full_bronze_provenance(self):
        """Sample rows must retain the SHA of the entire source file."""
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [self.sample_game() for _ in range(3)]
            for index, item in enumerate(items):
                item["game_id"] = f"game-{index}"
                item["moves"][0]["action"] = {"source": 54 + 2 * index, "target": 45 + 2 * index}
                # Keep source snapshots consistent with their newly chosen action.
                engine = GameEngine()
                after = engine.apply_move(GameState.initial(), Move(54 + 2 * index, 45 + 2 * index))
                item["moves"][0]["next_board"] = list(after.board)
                item["final_board"] = list(after.board)
            source = self.write_records(root, "bronze.jsonl", items)
            result = SilverPipeline().run_alphabeta(
                [source], root / "silver", max_records=2
            )
            self.assertEqual(result.source_records, 2)
            self.assertEqual(result.accepted_games, 2)
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            for row in self.read_jsonl(root / "silver", "games.jsonl"):
                self.assertEqual(row["provenance"]["file_sha256"], checksum)

    def test_sample_limit_rejects_invalid_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_records(root, "bronze.jsonl", [self.sample_game()])
            with self.assertRaises(ValueError):
                SilverPipeline().run_alphabeta([source], root / "silver", max_records=0)

    def test_valid_replay_produces_shared_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_records(root, "bronze.jsonl", [self.sample_game()])
            output = root / "silver"
            report = SilverPipeline().run_alphabeta([source], output, source_version="v1")
            self.assertEqual((report.source_records, report.accepted_games, report.transitions), (1, 1, 1))
            game = next(iter(SilverGameReader(output / "games.jsonl")))
            step = next(iter(SilverTransitionReader(output / "transitions.jsonl")))
            self.assertEqual(game.provenance.source_game_id, "raw-123")
            self.assertEqual(game.provenance.source_file, "bronze.jsonl")
            self.assertEqual(len(game.provenance.file_sha256), 64)
            self.assertEqual(step.source_type, "opening_book")
            self.assertTrue(step.truncated)
            self.assertFalse(step.terminated)
            self.assertIsNone(step.recorded_reward)
            self.assertEqual(step.board, list(GameState.initial().board))

    def test_duplicates_are_reported_not_added_to_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = self.sample_game(actor="alphabeta")
            other = {**item, "game_id": "different-id"}
            source = self.write_records(root, "bronze.jsonl", [item, other])
            output = root / "silver"
            report = SilverPipeline().run_alphabeta([source], output)
            self.assertEqual((report.accepted_games, report.duplicate_games), (1, 1))
            self.assertEqual(len(self.read_jsonl(output / "duplicates.jsonl")), 1)
            self.assertEqual(len(self.read_jsonl(output / "transitions.jsonl")), 1)

    def test_malformed_json_and_schema_are_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_records(
                root, "bronze.jsonl", ["{bad", {"game_id": "incomplete"}, self.sample_game()]
            )
            output = root / "silver"
            report = SilverPipeline().run_alphabeta([source], output)
            self.assertEqual(report.quarantined_records, 2)
            self.assertEqual(report.accepted_games, 1)
            errors = self.read_jsonl(output / "quarantine.jsonl")
            self.assertEqual([row["line_number"] for row in errors], [1, 2])
            self.assertEqual([row["error_code"] for row in errors], ["INVALID_JSON", "INVALID_SCHEMA"])

    def test_illegal_move_is_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = self.sample_game()
            item["moves"][0]["action"] = {"source": 0, "target": 1}
            source = self.write_records(root, "bronze.jsonl", [item])
            output = root / "silver"
            report = SilverPipeline().run_alphabeta([source], output)
            self.assertEqual(report.quarantined_records, 1)
            self.assertEqual(self.read_jsonl(output / "quarantine.jsonl")[0]["error_code"], "ILLEGAL_MOVE")
            self.assertEqual(self.read_jsonl(output / "games.jsonl"), [])

    def test_source_snapshot_mismatch_is_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = self.sample_game()
            item["moves"][0]["next_board"][0] = 0
            source = self.write_records(root, "bronze.jsonl", [item])
            output = root / "silver"
            SilverPipeline().run_alphabeta([source], output)
            self.assertEqual(
                self.read_jsonl(output / "quarantine.jsonl")[0]["error_code"], "SOURCE_MISMATCH"
            )

    def test_fabricated_winner_is_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = self.sample_game()
            item["outcome"] = {"status": "finished", "winner": "red", "reason": "checkmate"}
            source = self.write_records(root, "bronze.jsonl", [item])
            output = root / "silver"
            SilverPipeline().run_alphabeta([source], output)
            self.assertEqual(
                self.read_jsonl(output / "quarantine.jsonl")[0]["error_code"], "OUTCOME_MISMATCH"
            )

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_records(root, "bronze.jsonl", [self.sample_game()])
            output = root / "silver"
            pipeline = SilverPipeline()
            pipeline.run_alphabeta([source], output)
            with self.assertRaises(FileExistsError):
                pipeline.run_alphabeta([source], output)

    def test_dedup_across_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = self.sample_game()
            one = self.write_records(root, "a.jsonl", [item])
            two = self.write_records(root, "b.jsonl", [item])
            report = SilverPipeline().run_alphabeta([one, two], root / "silver")
            self.assertEqual((report.accepted_games, report.duplicate_games), (1, 1))

    def test_unknown_actor_is_not_expert(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_records(root, "bronze.jsonl", [self.sample_game(actor="unknown_agent")])
            output = root / "silver"
            SilverPipeline().run_alphabeta([source], output)
            game = next(iter(SilverGameReader(output / "games.jsonl")))
            self.assertEqual(game.moves[0].source_type, "unknown")


if __name__ == "__main__":
    unittest.main()
