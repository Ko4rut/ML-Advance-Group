"""Check export of tournament AI-vs-AI records into frontend replay files."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_tournament_replay import (
    export_tournament_replay,
    normalize_tournament_game,
)


class ExportTournamentReplayTests(unittest.TestCase):
    @staticmethod
    def record(index: int, red: str = "alphabeta", black: str = "mcts") -> dict:
        return {
            "schema_version": 1,
            "run_id": "demo-run",
            "game_index": index,
            "game_key": f"game-{index}",
            "red_agent": red,
            "black_agent": black,
            "opening": {"id": "book", "plies": 0},
            "moves": [
                {"ply": 1, "side": "red", "actor": red, "source": 54, "target": 45},
                {"ply": 2, "side": "black", "actor": black, "source": 27, "target": 36},
            ],
            "outcome": {"status": "truncated", "winner": None, "reason": "max_plies"},
        }

    def test_normalization_preserves_metadata_and_actions(self):
        result = normalize_tournament_game(self.record(2))
        self.assertEqual(result["game_id"], "tournament:demo-run:game-2")
        self.assertEqual(result["initial_side"], "red")
        self.assertEqual(result["red_agent"], "alphabeta")
        self.assertEqual(result["black_agent"], "mcts")
        self.assertEqual(result["moves"][0]["action"], {"source": 54, "target": 45})
        self.assertEqual(result["moves"][1]["side"], "black")
        self.assertNotIn("source", result["moves"][0])

    def test_select_game_by_index_and_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "games.jsonl"
            records = [self.record(0, "rules", "random"), self.record(1)]
            source.write_text(
                "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8",
            )
            by_index = folder / "index.json"
            result = export_tournament_replay(source, by_index, game_index=1)
            self.assertEqual(len(result["moves"]), 2)
            self.assertEqual(json.loads(by_index.read_text())["red_agent"], "alphabeta")
            by_pair = folder / "pair.json"
            result = export_tournament_replay(source, by_pair, red="rules", black="random")
            self.assertEqual(result["black_agent"], "random")

    def test_missing_match_and_existing_output_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "games.jsonl"
            source.write_text(json.dumps(self.record(0)) + "\n", encoding="utf-8")
            output = folder / "single.json"
            with self.assertRaisesRegex(ValueError, "no matching"):
                export_tournament_replay(source, output, game_index=900)
            export_tournament_replay(source, output, game_index=0)
            with self.assertRaises(FileExistsError):
                export_tournament_replay(source, output)

    def test_corrupt_move_is_rejected(self):
        bad = self.record(0)
        bad["moves"][0]["source"] = 99
        with self.assertRaisesRegex(ValueError, "invalid source"):
            normalize_tournament_game(bad)

    def test_real_self_play_is_replayable(self):
        """Exercise actual engine-selected moves, not just a synthetic JSON fixture."""
        from src.environment import Move, XiangqiEnv
        from src.training.batch_games import play_game

        source = play_game("random", "rules", seed=7, max_plies=6, opening_mode="none")
        tournament_record = self.record(3, "random", "rules")
        tournament_record["moves"] = [
            {
                "ply": row["ply"],
                "side": row["side"],
                "actor": row["actor"],
                "source": row["action"]["source"],
                "target": row["action"]["target"],
            }
            for row in source["moves"]
        ]
        converted = normalize_tournament_game(tournament_record)
        env = XiangqiEnv()
        for row in converted["moves"]:
            action = row["action"]
            move = Move(action["source"], action["target"])
            self.assertIn(move, env.get_legal_actions())
            env.step(move)
        self.assertEqual(list(env.state.board), source["final_board"])


if __name__ == "__main__":
    unittest.main()
