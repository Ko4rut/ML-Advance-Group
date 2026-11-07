"""Reward v2: actors, terminal/truncation semantics, potentials and provenance."""

import math
import tempfile
import unittest
from pathlib import Path

from src.data.gold.data_pipeline import GoldPipeline, RewardConfig, calculate_reward, material_potential
from src.data.gold.reader import OfflineRLReader
from tests.data_pipeline.test_gold_pipeline import make_game, write_silver


class RewardTests(unittest.TestCase):
    def test_side_symmetry_and_bounded_potential(self):
        game, steps = make_game([(54, 45)], status="truncated", winner=None)
        board = steps[0].board
        self.assertAlmostEqual(
            material_potential(board, "red"),
            -material_potential(board, "black"),
        )
        self.assertLessEqual(abs(material_potential(board, "red")), 1.0)

    def test_pawn_advancement_changes_potential(self):
        game, steps = make_game([(54, 45)], status="truncated", winner=None)
        before = material_potential(steps[0].board, "red")
        after = material_potential(steps[0].next_board, "red")
        self.assertGreater(after, before)

    def test_sparse_baseline_unchanged(self):
        game, steps = make_game([(54, 45), (27, 36)], winner="black")
        cfg = RewardConfig(mode="sparse")
        self.assertEqual([calculate_reward(game, row, index == 1, cfg)
                          for index, row in enumerate(steps)], [0.0, 1.0])

    def test_nonterminal_shaping_is_actor_perspective(self):
        game, steps = make_game([(54, 45)], status="truncated", winner=None)
        step = steps[0]
        cfg = RewardConfig(mode="shaped", gamma=0.99, shaping_weight=0.15)
        p_now = material_potential(step.board, step.side)
        p_next = material_potential(step.next_board, step.side)
        expected = cfg.shaping_weight * (cfg.gamma * p_next - p_now)
        self.assertAlmostEqual(calculate_reward(game, step, False, cfg), expected)
        # Important: the next side is the opponent, and must NOT be used here.
        self.assertNotAlmostEqual(expected, cfg.shaping_weight *
                                  (cfg.gamma * material_potential(step.next_board, step.next_side)
                                   - p_now))

    def test_finished_win_and_draw_have_no_successor_potential(self):
        cfg = RewardConfig(mode="shaped")
        game, steps = make_game([(54, 45), (27, 36)], winner="black")
        final = steps[-1]
        expected = 1.0 - cfg.shaping_weight * material_potential(final.board, final.side)
        self.assertAlmostEqual(calculate_reward(game, final, True, cfg), expected)

        draw_game, draw_steps = make_game([(54, 45)], winner=None)
        draw = draw_steps[-1]
        self.assertAlmostEqual(
            calculate_reward(draw_game, draw, True, cfg),
            -cfg.shaping_weight * material_potential(draw.board, draw.side),
        )

    def test_truncation_is_not_a_draw_and_has_successor_potential(self):
        game, steps = make_game([(54, 45)], status="truncated", winner=None)
        cfg = RewardConfig(mode="shaped")
        reward = calculate_reward(game, steps[-1], False, cfg)
        self.assertNotEqual(reward, -cfg.shaping_weight * material_potential(steps[-1].board, "red"))
        self.assertTrue(math.isfinite(reward))

    def test_pipeline_emits_versioned_reward_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            game, steps = make_game([(54, 45), (27, 36)], winner="black")
            write_silver(root / "silver", [(game, steps)])
            cfg = RewardConfig(mode="shaped", gamma=0.97, shaping_weight=0.2)
            report = GoldPipeline(reward_config=cfg).run([root / "silver"], root / "gold")
            self.assertEqual(report["reward_definition"], cfg.definition)
            self.assertEqual(report["reward_config"]["gamma"], 0.97)
            split = GoldPipeline().config.split_for(game.canonical_game_sha256)
            samples = list(OfflineRLReader(root / "gold" / f"offline_rl_{split}.jsonl"))
            self.assertEqual(len(samples), 2)
            self.assertTrue(all(item.reward_definition == cfg.definition for item in samples))
            self.assertTrue(all(math.isfinite(item.reward) for item in samples))

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            RewardConfig(mode="shaped", gamma=0)
        with self.assertRaises(ValueError):
            RewardConfig(mode="shaped", shaping_weight=-1)
        with self.assertRaises(ValueError):
            RewardConfig(mode="shaped", material_scale=float("nan"))


if __name__ == "__main__":
    unittest.main()
