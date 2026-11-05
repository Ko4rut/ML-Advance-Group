"""Schema/reader smoke tests (from project/backend: unittest discover -s tests)."""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.data.bronze.reader import AlphaBetaRawReader
from src.data.gold.models import OfflineRLSample, PolicySample, dense_legal_action_mask
from src.data.gold.reader import PolicyReader
from src.data.silver.models import SilverGame, SilverTransition
from src.data.silver.reader import SilverGameReader, SilverTransitionReader

BOARD = [0] * 90
ACTION = {"source": 70, "target": 68}
SOURCE = {
    "source": "alphabeta",
    "source_version": "batch-v1",
    "source_file": "games.jsonl",
    "file_sha256": "a" * 64,
}


class SchemaReaderTests(unittest.TestCase):
    def write_rows(self, folder, rows):
        path = Path(folder) / "samples.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_alphabeta_raw_reader_preserves_unknown_metadata(self):
        raw = {"schema_version": 1, "game_id": "abc", "initial_board": BOARD,
               "moves": [], "outcome": {"status": "truncated"}, "opening_book_version": "v1"}
        with tempfile.TemporaryDirectory() as folder:
            parsed = list(AlphaBetaRawReader(self.write_rows(folder, [raw])))
        self.assertEqual(parsed[0].model_dump()["opening_book_version"], "v1")

    def test_silver_game_reader(self):
        game = {"game_id": "g1", "provenance": SOURCE, "initial_board": BOARD,
                "initial_side": "red", "moves": [{"ply": 1, "action": ACTION,
                "source_type": "opening_book"}],
                "outcome": {"status": "truncated", "reason": "max_plies"},
                "canonical_game_sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as folder:
            rows = list(SilverGameReader(self.write_rows(folder, [game])))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].moves[0].source_type, "opening_book")
        with self.assertRaises(ValidationError):
            SilverGame.model_validate({**game, "initial_board": [0] * 89})
        with self.assertRaises(ValidationError):
            SilverGame.model_validate({**game, "moves": [{"ply": 2, "action": ACTION, "source_type": "human"}]})

    def test_silver_transition_requires_alternating_sides(self):
        sample = {"game_id": "g1", "ply": 1, "board": BOARD, "side": "red",
                  "action": ACTION, "next_board": BOARD, "next_side": "black",
                  "source_type": "alphabeta"}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(len(list(SilverTransitionReader(self.write_rows(folder, [sample])))), 1)
        with self.assertRaises(ValidationError):
            SilverTransition.model_validate({**sample, "next_side": "red"})

    def test_policy_sparse_mask_and_label(self):
        sample = {"game_id": "g1", "ply": 1, "split": "train", "state": BOARD,
                  "side": "red", "target_action_id": 70 * 90 + 68,
                  "legal_action_ids": [70 * 90 + 68, 70 * 90 + 67],
                  "behavior_source_type": "human"}
        with tempfile.TemporaryDirectory() as folder:
            policy = next(iter(PolicyReader(self.write_rows(folder, [sample]))))
        mask = dense_legal_action_mask(policy.legal_action_ids)
        self.assertEqual(len(mask), 8100)
        self.assertEqual(sum(mask), 2)
        with self.assertRaises(ValidationError):
            PolicySample.model_validate({**sample, "target_action_id": 12})

    def test_offline_rl_must_explicitly_define_reward(self):
        sample = {"game_id": "g1", "ply": 1, "split": "train", "state": BOARD,
                  "side": "red", "action_id": 70 * 90 + 68, "reward": 1.0,
                  "reward_definition": "terminal_outcome_v1", "next_state": BOARD,
                  "terminated": True, "truncated": False, "behavior_source_type": "human"}
        self.assertEqual(OfflineRLSample.model_validate(sample).reward, 1.0)
        with self.assertRaises(ValidationError):
            OfflineRLSample.model_validate({k: v for k, v in sample.items() if k != "reward_definition"})

    def test_reader_error_includes_file_and_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            path.write_text("\nnot-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"bad.jsonl:2: invalid JSON"):
                list(AlphaBetaRawReader(path))


if __name__ == "__main__":
    unittest.main()
