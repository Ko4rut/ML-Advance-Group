"""Tests for Gold offline tabular Q-learning and existing Agent compatibility."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agents.tabular_q_agent import TabularQAgent
from src.data.gold.models import OfflineRLSample
from src.environment import GameEngine, GameState
from src.training.tabular_qlearning import (
    TabularQConfig,
    TabularQTable,
    TabularQTrainer,
    load_gold_splits,
    state_key,
    train_from_gold,
)


def sample(
    game_id: str, ply: int, board: list[int], side: str,
    action: int, reward: float, next_board: list[int], next_side: str,
    *, split: str = "train", terminated: bool = False,
    truncated: bool = False, legal: list[int] | None = None,
    next_legal: list[int] | None = None,
) -> OfflineRLSample:
    return OfflineRLSample(
        game_id=game_id, ply=ply, split=split, state=board, side=side,
        action_id=action, reward=reward,
        reward_definition="terminal_outcome_potential_material_actor_perspective_v2",
        next_state=next_board, next_side=next_side,
        legal_action_ids=legal or [action],
        next_legal_action_ids=[] if terminated else (next_legal or [101]),
        terminated=terminated, truncated=truncated,
        behavior_source_type="human",
    )


def game_steps(split: str = "train") -> list[OfflineRLSample]:
    state = GameState.initial()
    engine = GameEngine()
    first = engine.legal_moves(state)[0]
    second_state = engine.apply_move(state, first)
    second = engine.legal_moves(second_state)[0]
    last_state = second_state.moved(second)
    a1 = first.source * 90 + first.target
    a2 = second.source * 90 + second.target
    return [
        sample("game-1", 1, list(state.board), "red", a1, 0.0,
               list(second_state.board), "black", split=split,
               legal=[a1], next_legal=[a2]),
        sample("game-1", 2, list(second_state.board), "black", a2, 1.0,
               list(last_state.board), "red", split=split,
               legal=[a2], terminated=True),
    ]


class TabularQTests(unittest.TestCase):
    def test_state_keys_are_exact_and_side_aware(self):
        state = list(GameState.initial().board)
        self.assertEqual(len(state_key(state, "red")), 91)
        self.assertNotEqual(state_key(state, "red"), state_key(state, "black"))
        with self.assertRaises(ValueError):
            state_key([0] * 89, "red")

    def test_negamax_backup_and_reverse_episode_propagation(self):
        steps = game_steps()
        trainer = TabularQTrainer(TabularQConfig(alpha=1.0, gamma=0.9, epochs=1))
        history = trainer.fit([steps])
        first = steps[0]
        second = steps[1]
        q_red = trainer.table.values[state_key(first.state, first.side)][first.action_id]
        q_black = trainer.table.values[state_key(second.state, second.side)][second.action_id]
        self.assertAlmostEqual(q_black, 1.0)
        self.assertAlmostEqual(q_red, -0.9)
        self.assertEqual(history[0]["updates"], 2)

    def test_unknown_actions_do_not_bootstrap_or_play(self):
        table = TabularQTable()
        board = list(GameState.initial().board)
        key = state_key(board, "red")
        table.observe(key, 123)
        table.values[key][123] = -0.2
        self.assertIsNone(table.supported_max(key, [456]))
        self.assertEqual(table.select(key, [123, 456]), 123)
        self.assertIsNone(table.select(key, [456]))

    def test_truncated_transition_skipped_not_assumed_draw(self):
        steps = game_steps()
        last = steps[-1].model_copy(update={
            "terminated": False, "truncated": True,
            "next_legal_action_ids": [123],
        })
        trainer = TabularQTrainer(TabularQConfig(alpha=1.0, gamma=0.9, epochs=1))
        history = trainer.fit([[steps[0], last]])
        self.assertEqual(history[0]["skipped_truncated"], 1)
        self.assertEqual(history[0]["updates"], 1)
        self.assertEqual(history[0]["missing_next_action_support"], 0)

    def test_checkpoint_and_agent(self):
        steps = game_steps()
        trainer = TabularQTrainer(TabularQConfig(alpha=1.0, epochs=1))
        trainer.fit([steps])
        state = GameState.initial()
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model.gz"
            trainer.table.save(model, trainer.config, {"source": "unit-test"})
            restored, payload = TabularQTable.load(model)
            self.assertEqual(payload["metadata"]["source"], "unit-test")
            agent = TabularQAgent(restored, seed=42)
            decision = agent.select_action(state)
            self.assertTrue(agent.last_action_supported)
            self.assertIn(decision.move, GameEngine().legal_moves(state))
            self.assertEqual(decision.move.source * 90 + decision.move.target, steps[0].action_id)

    def test_train_only_and_validation_unseen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "val", "test"):
                rows = game_steps(split)
                if split != "train":
                    # Different game IDs only; state overlap remains intentionally visible.
                    rows = [
                        row.model_copy(update={"game_id": "held-out-game"})
                        for row in rows
                    ]
                with (root / f"offline_rl_{split}.jsonl").open("w") as stream:
                    for row in rows:
                        stream.write(row.model_dump_json() + "\n")
            (root / "report.json").write_text(json.dumps({
                "reward_definition": "terminal_outcome_potential_material_actor_perspective_v2",
                "reward_config": {"gamma": 0.99},
            }))
            report = train_from_gold(root, root / "model", TabularQConfig(epochs=2))
            self.assertEqual(report["train_games"], 1)
            self.assertEqual(report["train_transitions"], 2)
            self.assertEqual(report["splits"]["test"]["transitions"], 2)
            self.assertTrue((root / "model" / "q_table.json.gz").is_file())
            with self.assertRaises(FileExistsError):
                train_from_gold(root, root / "model", TabularQConfig(epochs=2))

    def test_reject_gamma_mismatch(self):
        with self.assertRaises(ValueError):
            TabularQConfig(alpha=0)
        with self.assertRaises(ValueError):
            TabularQConfig(gamma=1.5)


if __name__ == "__main__":
    unittest.main()
