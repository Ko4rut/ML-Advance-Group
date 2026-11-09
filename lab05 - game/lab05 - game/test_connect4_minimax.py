import contextlib
import io
import unittest

import numpy as np

from connect4_core import actions, empty_board, random_player, result, terminal
from connect4_minimax import (
    AlphaBetaAgent,
    SearchLimitReached,
    alpha_beta_player,
    alpha_beta_search,
    board_from_moves,
    center_first,
    minimax_search,
    no_ordering,
    play,
    timed_search,
    win_block_center,
)

ORDERINGS = (no_ordering, center_first, win_block_center)


def small_boards(count=10, empty_cells=8, seed=20261109):
    """legal 4x4 boards with `empty_cells` empty cells, made of random moves."""
    np.random.seed(seed)
    boards = []
    while len(boards) < count:
        board, player = empty_board((4, 4)), 1
        while not terminal(board) and int(np.sum(board == 0)) > empty_cells:
            board = result(board, player, random_player(board, player))
            player = -player
        if not terminal(board):
            boards.append((board, player))
    return boards


class Connect4MinimaxTests(unittest.TestCase):
    def assert_keeps_the_win(self, board, player, move):
        """after `move` the opponent must lose, i.e. the win is still there."""
        after = result(board, player, move)
        if terminal(after):
            return
        self.assertEqual(alpha_beta_search(after, -player)["value"], -1)

    def test_finds_immediate_wins(self):
        winning_boards = {
            "horizontal": [0, 0, 1, 0, 2, 0],
            "vertical": [0, 1, 0, 1, 0, 2],
            "diagonal": [1, 3, 0, 2, 1, 2, 3, 3, 2, 1],
        }
        for name, moves in winning_boards.items():
            board, player = board_from_moves((4, 4), moves)
            for search in (minimax_search, alpha_beta_search):
                with self.subTest(line=name, search=search.__name__):
                    info = search(board, player)
                    self.assertEqual(info["value"], 1)
                    self.assert_keeps_the_win(board, player, info["move"])

    def test_blocks_the_only_losing_threat(self):
        board, player = board_from_moves((4, 4), [0, 0, 1, 0, 2])
        self.assertEqual(player, -1)

        for order in ORDERINGS:
            with self.subTest(order=order.__name__):
                self.assertEqual(alpha_beta_search(board, player, move_order=order)["move"], 3)
        self.assertEqual(minimax_search(board, player)["move"], 3)

    def test_minimax_and_alpha_beta_have_the_same_value(self):
        for board, player in small_boards():
            with self.subTest(board=board.tolist()):
                expected = minimax_search(board, player)["value"]
                self.assertEqual(alpha_beta_search(board, player)["value"], expected)

    def test_all_move_orderings_have_the_same_value(self):
        for board, player in small_boards():
            expected = minimax_search(board, player)["value"]
            for order in ORDERINGS:
                with self.subTest(order=order.__name__, board=board.tolist()):
                    self.assertEqual(
                        alpha_beta_search(board, player, move_order=order)["value"], expected
                    )

    def test_node_limit_stops_the_search(self):
        board = empty_board((4, 4))
        with self.assertRaises(SearchLimitReached):
            minimax_search(board, 1, node_limit=50)
        with self.assertRaises(SearchLimitReached):
            alpha_beta_search(board, 1, node_limit=50)

        info = timed_search(alpha_beta_search, board, 1, node_limit=50)
        self.assertFalse(info["finished"])
        self.assertIsNone(info["move"])
        self.assertEqual(info["nodes"], 50)

    def test_searches_are_silent_unless_verbose(self):
        board, player = board_from_moves((4, 4), [0, 0, 1, 0, 2])

        quiet = io.StringIO()
        with contextlib.redirect_stdout(quiet):
            alpha_beta_search(board, player)
            minimax_search(board, player)
        self.assertEqual(quiet.getvalue(), "")

        loud = io.StringIO()
        with contextlib.redirect_stdout(loud):
            info = alpha_beta_search(board, player, verbose=1)
        self.assertEqual(loud.getvalue(), f"Number of nodes searched: {info['nodes']}\n")

    def test_searches_do_not_share_a_node_counter(self):
        board = empty_board((4, 4))
        first = alpha_beta_search(board, 1, move_order=center_first)
        alpha_beta_search(board, 1)  # a search in between must not change the count of the next
        self.assertEqual(alpha_beta_search(board, 1, move_order=center_first)["nodes"], first["nodes"])

    def test_pruning_visits_fewer_nodes_than_plain_minimax(self):
        for board, player in small_boards():
            plain = minimax_search(board, player)
            pruned = alpha_beta_search(board, player)
            self.assertLessEqual(pruned["nodes"], plain["nodes"])

    def test_agent_returns_a_legal_column_and_keeps_the_board(self):
        board, player = board_from_moves((4, 4), [1, 2, 1, 2, 3])
        before = board.copy()

        for agent_move in (AlphaBetaAgent().act(board, player), alpha_beta_player(board, player)):
            self.assertIn(agent_move, actions(board))
        self.assertTrue(np.array_equal(board, before))

    def test_agent_remembers_positions(self):
        board, player = board_from_moves((4, 4), [1, 2, 1, 2, 3])
        agent = AlphaBetaAgent(move_order=win_block_center)

        first = agent.act(board, player)
        searches = len(agent.nodes)
        self.assertEqual(searches, 1)

        self.assertEqual(agent.act(board, player), first)
        self.assertEqual(len(agent.nodes), searches)
        self.assertEqual(len(agent.move_times), searches)

        forgetful = AlphaBetaAgent(remember_moves=False)
        forgetful.act(board, player)
        forgetful.act(board, player)
        self.assertEqual(len(forgetful.nodes), 2)

    def test_alpha_beta_does_not_lose_to_random_on_4x4(self):
        np.random.seed(20261109)
        agent = AlphaBetaAgent()

        as_first = play(agent.act, random_player, N=2)
        as_second = play(random_player, agent.act, N=2)

        self.assertEqual(sum(as_first.values()), 2)
        self.assertEqual(as_first["second"], 0)
        self.assertEqual(sum(as_second.values()), 2)
        self.assertEqual(as_second["first"], 0)


if __name__ == "__main__":
    unittest.main()
