import unittest

import numpy as np

from connect4_core import empty_board, result, terminal
from connect4_minimax import board_from_moves
from connect4_montecarlo import (
    PMCSAgent,
    playout,
    playouts,
    pmcs,
    pmcs_values,
)

SEED = 20261110

# the first boards of the Task 3 table (same moves as board_specs in assignment_connect4.ipynb)
BOARD_SPECS = {
    "horizontal win": ((4, 4), [0, 0, 1, 0, 2, 0], 3),
    "vertical win": ((4, 4), [0, 1, 0, 1, 0, 2], 0),
    "diagonal win": ((4, 4), [1, 3, 0, 2, 1, 2, 3, 3, 2, 1], 3),
    "must block": ((4, 4), [0, 0, 1, 0, 2], 3),
}


class Connect4MonteCarloTests(unittest.TestCase):
    def board(self, name):
        shape, moves, _ = BOARD_SPECS[name]
        return board_from_moves(shape, moves)

    def test_playout_returns_a_utility_and_keeps_the_board(self):
        np.random.seed(SEED)
        board = empty_board((6, 7))
        before = board.copy()
        for column in range(7):
            for u in playouts(board, column, 1, N = 20):
                self.assertIn(u, (-1, 0, 1))
        self.assertTrue(np.array_equal(board, before))

    def test_playout_of_an_immediate_win_is_one(self):
        for name in ("horizontal win", "vertical win", "diagonal win"):
            board, player = self.board(name)
            column = BOARD_SPECS[name][2]
            with self.subTest(board = name):
                self.assertEqual(playout(board, column, player), 1)

    def test_pmcs_finds_the_immediate_win(self):
        for name in ("horizontal win", "vertical win"):
            board, player = self.board(name)
            np.random.seed(SEED)
            with self.subTest(board = name):
                self.assertEqual(pmcs(board, N = 200, player = player), BOARD_SPECS[name][2])

    def test_pmcs_blocks_the_opponent(self):
        board, player = self.board("must block")
        self.assertEqual(player, -1)
        np.random.seed(SEED)
        self.assertEqual(pmcs(board, N = 500, player = player), BOARD_SPECS["must block"][2])

    def test_n_smaller_than_the_number_of_columns_still_works(self):
        board = empty_board((6, 7))
        np.random.seed(SEED)
        # 7 columns, N = 3: every column still gets one playout
        values = pmcs_values(board, N = 3)
        self.assertEqual(sorted(values), list(range(7)))
        self.assertIn(pmcs(board, N = 3), range(7))

    def test_finished_board_raises_value_error(self):
        board, player = self.board("horizontal win")
        board = result(board, player, BOARD_SPECS["horizontal win"][2])
        self.assertTrue(terminal(board))
        with self.assertRaises(ValueError):
            pmcs(board, N = 10, player = -player)

    def test_agent_returns_legal_columns_and_remembers_move_times(self):
        np.random.seed(SEED)
        agent = PMCSAgent(N = 20)
        board = empty_board((4, 4))
        player = 1
        while not terminal(board):
            column = agent.act(board, player)
            self.assertIn(column, range(4))
            board = result(board, player, column)  # raises if the column is full
            player = -player
        self.assertEqual(len(agent.move_times), int(np.sum(board != 0)))
        self.assertTrue(all(t >= 0 for t in agent.move_times))

    def test_same_seed_gives_the_same_move(self):
        board = empty_board((6, 7))
        moves = []
        for _ in range(2):
            np.random.seed(SEED)
            moves.append(pmcs(board, N = 100))
        self.assertEqual(moves[0], moves[1])


if __name__ == "__main__":
    unittest.main()
