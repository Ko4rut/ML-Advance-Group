"""Regression tests for standard Connect 4 heuristic alpha-beta (Lab05 Task 4)."""
import unittest

import numpy as np

from connect4_core import empty_board, result, terminal, utility
from connect4_heuristic import (
    HeuristicAlphaBetaAgent,
    heuristic_evaluation,
    play_heuristic_match,
)


class Connect4HeuristicTests(unittest.TestCase):
    def test_pr59_tactical_positions_and_heuristic_bounds(self):
        # Same five positions and tactical choices as the original PR #59.
        empty = np.zeros((6, 7), dtype=int)
        horizontal_win = empty.copy()
        horizontal_win[-1, :3] = 1
        vertical_win = empty.copy()
        vertical_win[-3:, 2] = -1
        blocking_position = empty.copy()
        blocking_position[-1, :3] = -1
        diagonal_win = empty.copy()
        diagonal_win[-1, 0] = 1
        diagonal_win[-1, 1] = -1
        diagonal_win[-2, 1] = 1
        diagonal_win[-1, 2] = -1
        diagonal_win[-2, 2] = -1
        diagonal_win[-3, 2] = 1
        diagonal_win[-1, 3] = -1
        diagonal_win[-2, 3] = -1
        diagonal_win[-3, 3] = -1

        agent = HeuristicAlphaBetaAgent(depth=3)
        tests = [
            (empty, 1, 3),
            (horizontal_win, 1, 3),
            (vertical_win, -1, 2),
            (blocking_position, 1, 3),
            (diagonal_win, 1, 3),
        ]
        for board, player, expected in tests:
            with self.subTest(board=board.tolist(), player=player):
                before = board.copy()
                self.assertEqual(agent(board, player), expected)
                np.testing.assert_array_equal(board, before)
                self.assertTrue(-1 < heuristic_evaluation(board, player) < 1)

        win_board = result(horizontal_win, 1, 3)
        self.assertTrue(terminal(win_board))
        self.assertEqual(utility(win_board, 1), 1)
        self.assertEqual(heuristic_evaluation(win_board, 1), 1.0)
        self.assertEqual(heuristic_evaluation(win_board, -1), -1.0)

    def test_cutoff_and_nodes(self):
        board = empty_board((6, 7))
        counts = []
        for depth in (1, 2, 3):
            agent = HeuristicAlphaBetaAgent(depth=depth)
            self.assertIn(agent(board, 1), range(7))
            self.assertGreater(agent.nodes, 1)
            self.assertGreaterEqual(agent.last_elapsed, 0)
            counts.append(agent.nodes)
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])

    def test_invalid_depth_and_finished_board(self):
        for depth in (0, -1, True, 1.5, None):
            with self.subTest(depth=depth), self.assertRaises((TypeError, ValueError)):
                HeuristicAlphaBetaAgent(depth=depth)
        board = empty_board((4, 4))
        for player, column in ((1, 0), (-1, 1), (1, 0), (-1, 1),
                               (1, 0), (-1, 1), (1, 0)):
            board = result(board, player, column)
        self.assertTrue(terminal(board))
        with self.assertRaises(ValueError):
            HeuristicAlphaBetaAgent(depth=2)(board, -1)

    def test_match_returns_terminal_or_draw(self):
        result_info = play_heuristic_match(
            HeuristicAlphaBetaAgent(depth=2),
            HeuristicAlphaBetaAgent(depth=3),
            shape=(4, 5),
        )
        self.assertIn(result_info["winner"], (1, -1, None))
        self.assertGreater(result_info["plies"], 0)
        self.assertLessEqual(result_info["plies"], 20)
        self.assertTrue(terminal(result_info["board"]))


if __name__ == "__main__":
    unittest.main()
