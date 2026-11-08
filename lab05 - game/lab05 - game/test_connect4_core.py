import unittest

import numpy as np

from connect4_core import (
    actions,
    empty_board,
    random_player,
    result,
    terminal,
    utility,
)


class Connect4CoreTests(unittest.TestCase):
    def test_empty_board_and_available_actions(self):
        board = empty_board()

        self.assertEqual(board.shape, (6, 7))
        self.assertTrue(np.issubdtype(board.dtype, np.integer))
        self.assertTrue(np.all(board == 0))
        self.assertEqual(actions(board), list(range(7)))
        self.assertFalse(terminal(board))
        self.assertIsNone(utility(board, 1))

    def test_result_drops_disc_and_does_not_mutate_input(self):
        board = empty_board((4, 4))

        next_board = result(board, 1, 2)

        self.assertEqual(next_board[3, 2], 1)
        self.assertTrue(np.all(board == 0))
        self.assertIsNot(next_board, board)

    def test_full_column_is_not_legal_and_cannot_be_played(self):
        board = empty_board((4, 3))
        board[:, 1] = [1, -1, 1, -1]

        self.assertEqual(actions(board), [0, 2])
        with self.assertRaises(ValueError):
            result(board, 1, 1)

    def test_horizontal_win(self):
        board = empty_board()
        board[5, :4] = 1

        self.assertTrue(terminal(board))
        self.assertEqual(actions(board), [])
        self.assertEqual(utility(board, 1), 1)
        self.assertEqual(utility(board, -1), -1)

    def test_vertical_win(self):
        board = empty_board()
        board[2:, 3] = -1

        self.assertTrue(terminal(board))
        self.assertEqual(utility(board, -1), 1)

    def test_diagonal_down_right_win(self):
        board = empty_board()
        for offset in range(4):
            board[2 + offset, offset] = 1

        self.assertTrue(terminal(board))
        self.assertEqual(utility(board, 1), 1)

    def test_diagonal_up_right_win(self):
        board = empty_board()
        for offset in range(4):
            board[5 - offset, offset] = -1

        self.assertTrue(terminal(board))
        self.assertEqual(utility(board, -1), 1)

    def test_full_board_without_possible_four_is_draw(self):
        board = np.array([[1, -1, 1, -1], [-1, 1, -1, 1], [1, -1, 1, -1]])

        self.assertTrue(terminal(board))
        self.assertEqual(utility(board, 1), 0)
        self.assertEqual(utility(board, -1), 0)
        self.assertEqual(actions(board), [])

        with self.assertRaisesRegex(ValueError, "terminal state"):
            result(board, 1, 0)

    def test_nonterminal_board(self):
        board = result(empty_board((4, 4)), 1, 0)

        self.assertFalse(terminal(board))
        self.assertIsNone(utility(board, 1))

    def test_actions_excludes_full_columns(self):
        board = empty_board((3, 3))
        board[:, 0] = 1

        self.assertNotIn(0, actions(board))

    def test_random_player_selects_only_a_legal_column(self):
        board = empty_board((4, 3))
        board[:, 1] = [1, -1, 1, -1]

        for _ in range(100):
            self.assertIn(random_player(board, -1), [0, 2])

    def test_invalid_moves_and_states_are_reported(self):
        board = empty_board((4, 4))
        with self.assertRaises(ValueError):
            result(board, 1, 4)
        with self.assertRaises(ValueError):
            result(board, 0, 0)
        with self.assertRaises(TypeError):
            result(board, 1, 1.5)
        with self.assertRaises(ValueError):
            empty_board((0, 4))
        with self.assertRaises(ValueError):
            actions(np.array([[2]], dtype=int))

    def test_random_player_reports_no_available_move(self):
        board = np.ones((3, 2), dtype=int)

        with self.assertRaises(ValueError):
            random_player(board)

    def test_random_player_rejects_a_won_game(self):
        board = empty_board()
        board[5, :4] = 1

        with self.assertRaises(ValueError):
            random_player(board)

        with self.assertRaisesRegex(ValueError, "terminal state"):
            result(board, -1, 4)

    def test_utility_rejects_both_players_winning(self):
        board = empty_board()
        board[5, :4] = 1
        board[4, :4] = -1

        with self.assertRaises(ValueError):
            utility(board, 1)


if __name__ == "__main__":
    unittest.main()
