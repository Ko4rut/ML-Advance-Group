"""End-to-end regression for duplicate rejection and Top-K search diversity."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents import AlphaBetaAgent
from src.environment import GameEngine, GameState, Move
from src.training.batch_games import generate_batch, play_game
from src.training.topk_exploration import sample_top_k


class DatasetExplorationTests(unittest.TestCase):
    def test_topk_probability_and_score_margin(self):
        actions = [(Move(54, 45), 40), (Move(56, 47), 35), (Move(58, 49), 4)]
        move, score, distribution = sample_top_k(
            actions, random.Random(11),
            top_k=3, score_margin=8, temperature=6.0,
        )
        self.assertIn(move, [actions[0][0], actions[1][0]])
        self.assertIn(score, (40, 35))
        self.assertEqual(len(distribution), 2)
        self.assertAlmostEqual(sum(row[2] for row in distribution), 1.0)
        self.assertGreater(distribution[0][2], distribution[1][2])

    def test_only_best_is_eligible_when_gap_large(self):
        actions = [(Move(54, 45), 100), (Move(56, 47), -200)]
        for seed in range(10):
            chosen, _, distribution = sample_top_k(
                actions, random.Random(seed), top_k=2,
                score_margin=10, temperature=8.0,
            )
            self.assertEqual(chosen, actions[0][0])
            self.assertEqual(len(distribution), 1)

    def test_exact_root_scores_match_alpha_beta_depth_one(self):
        state = GameState.initial()
        agent = AlphaBetaAgent(GameEngine(), max_depth=1)
        ranked, nodes = agent.rank_actions(state)
        self.assertGreater(nodes, 1)
        self.assertEqual(ranked[0][0], agent.select_action(state).move)
        self.assertEqual({action for action, _ in ranked},
                         set(agent.engine.legal_moves(state)))

    def test_topk_game_metadata_and_legal_moves(self):
        record = play_game(
            "alphabeta", "random",
            seed=43, max_plies=5,
            opening_mode="book", opening_plies=2,
            red_depth=1, exploration_top_k=3,
            exploration_rate=1.0, exploration_window=4,
            exploration_max_moves=2, exploration_score_margin=1000,
        )
        self.assertEqual(record["opening_applied_plies"], 2)
        self.assertGreaterEqual(record["exploration_applied_moves"], 1)
        sampled = [row for row in record["moves"]
                   if row["actor"] == "alphabeta_exploration"]
        self.assertTrue(sampled)
        for row in sampled:
            details = row["exploration"]
            self.assertLessEqual(len(details["candidates"]), 3)
            self.assertGreaterEqual(details["best_score"], details["chosen_score"])
            self.assertGreater(row["nodes"], 0)
            self.assertIn(row["action"], [
                {"source": action["source"], "target": action["target"]}
                for action in details["candidates"]
            ])
        engine = GameEngine()
        state = GameState.initial()
        for row in record["moves"]:
            action = Move(**row["action"])
            self.assertIn(action, engine.legal_moves(state))
            state = state.moved(action)

    def test_duplicate_rejected_before_jsonl_and_bounded_retry_failure(self):
        duplicate = play_game("random", "random", max_plies=2, seed=91)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            def duplicate_game(red, black, *, game_index, **kwargs):
                record = dict(duplicate)
                record["game_index"] = game_index
                record["seed"] = 42 + game_index * 1009
                record["retry_attempt"] = kwargs.get("retry_attempt", 0)
                return record

            with patch("src.training.batch_games.play_game", side_effect=duplicate_game) as mocked:
                with self.assertRaisesRegex(RuntimeError, "unique game_index=1"):
                    generate_batch(
                        "random", "random", games=2, output=path,
                        max_plies=2, deduplicate_games=True,
                        max_duplicate_retries=1, quiet=True,
                    )
            self.assertEqual(mocked.call_count, 3)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["fingerprint"], duplicate["fingerprint"])
            checkpoint = json.loads(
                Path(str(path) + ".checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["completed_indices"], [0])

    def test_multiworker_writer_rejects_duplicate_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            result = generate_batch(
                "random", "random", games=4, output=path, seed=42,
                opening_mode="random", opening_plies=2,
                max_plies=5, workers=2,
                deduplicate_games=True, max_duplicate_retries=8,
                quiet=True,
            )
            self.assertEqual(result["games"], 4)
            self.assertEqual(result["duplicate_games_in_batch"], 0)
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len({r["fingerprint"] for r in records}), 4)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                generate_batch(
                    "random", "random", games=4, output=path, seed=42,
                    opening_mode="random", opening_plies=2,
                    max_plies=5, workers=1, deduplicate_games=True,
                    exploration_top_k=3, resume=True, quiet=True,
                )

    def test_retry_replay_is_deterministic(self):
        one = play_game(
            "random", "random", game_index=2, retry_attempt=3,
            opening_mode="book", opening_plies=6, max_plies=9,
        )
        two = play_game(
            "random", "random", game_index=2, retry_attempt=3,
            opening_mode="book", opening_plies=6, max_plies=9,
        )
        self.assertEqual(one["fingerprint"], two["fingerprint"])
        self.assertEqual(one["opening_id"], two["opening_id"])
        self.assertEqual(one["seed"], two["seed"])
        self.assertEqual(one["retry_attempt"], 3)

    def test_reject_invalid_exploration_options(self):
        with self.assertRaises(ValueError):
            play_game("random", "random", exploration_temperature=0)
        with self.assertRaises(ValueError):
            sample_top_k([(Move(54, 45), 2)], random.Random(1),
                         top_k=0, score_margin=1, temperature=1)


if __name__ == "__main__":
    unittest.main()
