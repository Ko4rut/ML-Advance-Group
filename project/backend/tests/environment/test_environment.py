import unittest

from project.backend.src.environment.engine import RulesConfig, apply_move, outcome
from project.backend.src.environment.env import XiangqiEnv
from project.backend.src.environment.movegen import legal_moves
from project.backend.src.environment.rules import is_in_check, targets
from project.backend.src.environment.state import GameState
from project.backend.src.environment.types import Move, Side


def position(pieces, side=Side.RED, **kwargs):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side, **kwargs)


class RulesTests(unittest.TestCase):
    def test_initial_position_and_move_count(self):
        state = GameState.initial()
        self.assertEqual(sum(piece != 0 for piece in state.board), 32)
        self.assertEqual(len(legal_moves(state)), 44)
        self.assertFalse(is_in_check(state.board, Side.RED))
        self.assertFalse(is_in_check(state.board, Side.BLACK))

    def test_horse_leg(self):
        board = [0] * 90
        board[7 * 9 + 4] = 4
        board[6 * 9 + 4] = 7
        moves = set(targets(tuple(board), 7 * 9 + 4))
        self.assertNotIn(5 * 9 + 3, moves)
        self.assertIn(6 * 9 + 2, moves)

    def test_elephant_eye_and_river(self):
        board = [0] * 90
        board[5 * 9 + 4] = 3
        board[6 * 9 + 5] = 7
        moves = set(targets(tuple(board), 5 * 9 + 4))
        self.assertEqual(moves, {7 * 9 + 2})

    def test_cannon_screen_and_rook_block(self):
        board = [0] * 90
        board[7 * 9 + 1] = 6
        board[5 * 9 + 1] = 7
        board[2 * 9 + 1] = -5
        board[0 * 9 + 1] = -4
        moves = set(targets(tuple(board), 7 * 9 + 1))
        self.assertIn(6 * 9 + 1, moves)
        self.assertIn(2 * 9 + 1, moves)
        self.assertNotIn(5 * 9 + 1, moves)
        rook_board = [0] * 90
        rook_board[5 * 9 + 4] = 5
        rook_board[5 * 9 + 6] = -7
        rook_moves = set(targets(tuple(rook_board), 5 * 9 + 4))
        self.assertNotIn(5 * 9 + 7, rook_moves)

    def test_pawn_after_crossing_river(self):
        board = [0] * 90
        board[4 * 9 + 4] = 7
        self.assertEqual(
            set(targets(tuple(board), 4 * 9 + 4)),
            {3 * 9 + 4, 4 * 9 + 3, 4 * 9 + 5},
        )

    def test_facing_kings_and_pin(self):
        state = position({(0, 4): -1, (9, 4): 1, (5, 4): 5})
        self.assertNotIn(Move(49, 48), legal_moves(state))
        self.assertIn(Move(49, 40), legal_moves(state))
        exposed = position({(0, 4): -1, (9, 4): 1})
        self.assertTrue(is_in_check(exposed.board, Side.RED))
        self.assertTrue(is_in_check(exposed.board, Side.BLACK))

    def test_mate_and_stalemate_are_terminal_losses(self):
        common = {(0, 4): -1, (9, 3): 1, (2, 3): 5, (2, 5): 5}
        mate = position({**common, (1, 4): 5, (2, 4): 5}, Side.BLACK)
        stalemate = position({**common, (1, 3): 5}, Side.BLACK)
        self.assertEqual(outcome(mate).reason, "checkmate")
        self.assertEqual(outcome(stalemate).reason, "stalemate")
        self.assertEqual(outcome(stalemate).winner, Side.RED)

    def test_capture_and_repetition_bookkeeping(self):
        state = position(
            {(0, 3): -1, (9, 4): 1, (5, 0): 5, (4, 0): -7},
            no_capture_count=10,
        )
        child = apply_move(state, Move(45, 36))
        self.assertEqual(child.no_capture_count, 0)
        self.assertEqual(child.side_to_move, Side.BLACK)
        self.assertEqual(child.ply_count, 1)
        self.assertEqual(state.board[45], 5)

        state = GameState.initial()
        for move in [Move(82, 65), Move(1, 20), Move(65, 82), Move(20, 1)] * 2:
            state = apply_move(state, move)
        self.assertEqual(outcome(state).reason, "repetition")

    def test_environment_step_reward_and_clone(self):
        env = XiangqiEnv(RulesConfig(no_capture_limit=1))
        original = env.clone_state()
        result = env.step(Move(54, 45))
        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.result.reason, "no_capture_limit")
        self.assertEqual(original, GameState.initial())
        with self.assertRaises(ValueError):
            env.step(Move(45, 36))


if __name__ == "__main__":
    unittest.main()
