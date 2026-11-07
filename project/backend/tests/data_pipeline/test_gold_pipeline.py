"""Silver -> Gold integration tests using real legal Xiangqi moves."""

import tempfile
import unittest
from pathlib import Path

from src.data.gold.data_pipeline import GoldConfig, GoldPipeline
from src.data.gold.reader import OfflineRLReader, PolicyReader
from src.data.models import MoveModel
from src.data.silver.data_pipeline.deduplication import canonical_game_hash
from src.data.silver.models import GameMove, GameOutcome, SilverGame, SilverTransition
from src.environment import GameEngine, GameState, Move


def make_game(actions, *, status="finished", winner="red", source="ccpd"):
    state = GameState.initial()
    moves = [GameMove(ply=i, action=MoveModel(source=s, target=t), source_type="human", actor="human")
             for i, (s, t) in enumerate(actions, start=1)]
    digest = canonical_game_hash(list(state.board), "red", [m.action for m in moves])
    game = SilverGame(
        game_id=f"{source}:{digest[:32]}",
        provenance={"source": source, "source_version": "test-v1",
                    "source_file": "test.pgn", "file_sha256": "a" * 64},
        initial_board=list(state.board), initial_side="red", moves=moves,
        outcome=GameOutcome(status=status, winner=winner if status == "finished" else None,
                            reason="pgn_result" if status == "finished" else "max_plies",
                            result_source="recorded" if status == "finished" else "engine"),
        canonical_game_sha256=digest,
    )
    transitions = []
    engine = GameEngine()
    for index, action in enumerate(moves):
        next_state = engine.apply_move(state, Move(action.action.source, action.action.target))
        last = index == len(moves) - 1
        transitions.append(SilverTransition(
            game_id=game.game_id, ply=index + 1,
            board=list(state.board), side="red" if state.side_to_move.value == 1 else "black",
            action=action.action,
            next_board=list(next_state.board),
            next_side="red" if next_state.side_to_move.value == 1 else "black",
            source_type="human", actor="human",
            recorded_game_end=last and status == "finished",
            truncated=last and status != "finished",
        ))
        state = next_state
    return game, transitions


def write_silver(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "games.jsonl").open("w", encoding="utf-8") as games, \
            (folder / "transitions.jsonl").open("w", encoding="utf-8") as steps:
        for game, transitions in rows:
            games.write(game.model_dump_json() + "\n")
            for step in transitions:
                steps.write(step.model_dump_json() + "\n")


def read_split(output, name, split):
    reader = PolicyReader if name == "policy" else OfflineRLReader
    return list(reader(output / f"{name}_{split}.jsonl"))


class GoldPipelineTests(unittest.TestCase):
    def test_recorded_terminal_reward_and_legal_masks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            game, transitions = make_game([(54, 45), (27, 36)], winner="black")
            write_silver(root / "silver", [(game, transitions)])
            output = root / "gold"
            report = GoldPipeline().run([root / "silver"], output)
            split = GoldConfig().split_for(game.canonical_game_sha256)
            policy = read_split(output, "policy", split)
            rl = read_split(output, "offline_rl", split)
            self.assertEqual(report["accepted_games"], 1)
            self.assertEqual(report["transitions"], 2)
            self.assertEqual(len(policy), len(rl))
            self.assertEqual([step.reward for step in rl], [0.0, 1.0])
            self.assertFalse(rl[0].terminated)
            self.assertTrue(rl[-1].terminated)
            self.assertEqual(rl[-1].termination_source, "recorded")
            self.assertFalse(rl[-1].truncated)
            self.assertEqual(rl[-1].next_legal_action_ids, [])
            self.assertTrue(rl[0].next_legal_action_ids)
            self.assertEqual(rl[0].next_side, "black")
            self.assertIn(policy[0].target_action_id, policy[0].legal_action_ids)
            self.assertEqual(report["games_by_split"][split], 1)
            self.assertEqual(len(list(output.glob("*.jsonl"))), 6)

    def test_truncated_keeps_next_actions_and_does_not_reward(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            game, transitions = make_game([(54, 45)], status="truncated", winner=None)
            write_silver(root / "silver", [(game, transitions)])
            output = root / "gold"
            GoldPipeline().run([root / "silver"], output)
            sample = read_split(output, "offline_rl", GoldConfig().split_for(game.canonical_game_sha256))[0]
            self.assertFalse(sample.terminated)
            self.assertTrue(sample.truncated)
            self.assertEqual(sample.reward, 0.0)
            self.assertTrue(sample.next_legal_action_ids)

    def test_identical_games_across_silver_runs_are_not_double_counted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            game, transitions = make_game([(54, 45)], winner="red")
            write_silver(root / "silver1", [(game, transitions)])
            write_silver(root / "silver2", [(game, transitions)])
            result = GoldPipeline().run([root / "silver1", root / "silver2"], root / "gold")
            self.assertEqual(result["accepted_games"], 1)
            self.assertEqual(result["cross_run_duplicate_games"], 1)
            self.assertEqual(result["transitions"], 1)

    def test_bad_silver_fails_without_publishing_gold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            game, transitions = make_game([(54, 45)])
            transitions[0] = transitions[0].model_copy(update={"next_board": [0] * 90})
            write_silver(root / "silver", [(game, transitions)])
            with self.assertRaisesRegex(ValueError, "invalid successor"):
                GoldPipeline().run([root / "silver"], root / "gold")
            self.assertFalse((root / "gold").exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            game, transitions = make_game([(54, 45)])
            write_silver(root / "silver", [(game, transitions)])
            (root / "gold").mkdir()
            with self.assertRaises(FileExistsError):
                GoldPipeline().run([root / "silver"], root / "gold")

    def test_game_split_is_stable_and_seeded(self):
        hash_value = "a" * 64
        self.assertEqual(GoldConfig(seed=42).split_for(hash_value), GoldConfig(seed=42).split_for(hash_value))
        with self.assertRaises(ValueError):
            GoldConfig(train_ratio=0.9, val_ratio=0.2)


if __name__ == "__main__":
    unittest.main()
