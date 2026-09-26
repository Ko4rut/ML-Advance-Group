import unittest

from src.environment.engine import RulesConfig, apply_move, outcome
from src.environment.env import XiangqiEnv
from src.environment.movegen import legal_moves
from src.environment.state import GameState
from src.environment.types import Move, Side


def position(pieces, side=Side.RED, **kwargs):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side, **kwargs)


class EnvironmentIntegrationTests(unittest.TestCase):
    def test_apply_move_is_immutable_and_updates_counters(self):
        state = position(
            {(0, 3): -1, (9, 4): 1, (5, 0): 5, (4, 0): -7},
            no_capture_count=10,
        )
        child = apply_move(state, Move(45, 36))

        self.assertEqual(state.board[45], 5)
        self.assertEqual(child.board[45], 0)
        self.assertEqual(child.board[36], 5)
        self.assertEqual(child.side_to_move, Side.BLACK)
        self.assertEqual(child.no_capture_count, 0)
        self.assertEqual(child.ply_count, state.ply_count + 1)
        self.assertEqual(len(child.history), len(state.history) + 1)

    def test_illegal_move_does_not_mutate_environment(self):
        env = XiangqiEnv()
        original = env.clone_state()

        with self.assertRaises(ValueError):
            env.step(Move(54, 55))

        self.assertEqual(env.state, original)

    def test_terminal_environment_rejects_further_steps(self):
        env = XiangqiEnv(RulesConfig(no_capture_limit=1))
        result = env.step(Move(54, 45))

        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.result.reason, "no_capture_limit")
        self.assertEqual(env.get_legal_actions(), [])
        with self.assertRaises(ValueError):
            env.step(Move(27, 36))

    def test_repetition_is_counted_with_side_to_move(self):
        state = GameState.initial()
        cycle = [Move(82, 65), Move(1, 20), Move(65, 82), Move(20, 1)]
        for move in cycle * 2:
            state = apply_move(state, move)

        result = outcome(state)
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, "repetition")

    def test_initial_position_has_legal_moves(self):
        state = GameState.initial()
        self.assertEqual(len(legal_moves(state)), 44)
        self.assertIsNone(outcome(state))


if __name__ == "__main__":
    unittest.main()
