"""Search evaluation regression tests for Minimax / Alpha-Beta."""

import unittest

from project.backend.src.agents import AlphaBetaAgent, MinimaxAgent
from project.backend.src.agents.position_evaluator import (
    evaluate_position,
    material_score,
)
from project.backend.src.environment import GameEngine, GameState, Side


def position(pieces, side=Side.RED, **kwargs):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side, **kwargs)


class PositionEvaluatorTests(unittest.TestCase):
    def test_initial_position_is_balanced(self):
        state = GameState.initial()
        self.assertEqual(evaluate_position(state, Side.RED), 0)
        self.assertEqual(evaluate_position(state, Side.BLACK), 0)

    def test_advancing_pawn_improves_score_without_capture(self):
        earlier = position({(0, 4): -1, (9, 4): 1, (6, 0): 7, (5, 4): 7})
        later = position({(0, 4): -1, (9, 4): 1, (4, 0): 7, (5, 4): 7})
        self.assertEqual(material_score(earlier, Side.RED), material_score(later, Side.RED))
        self.assertGreater(evaluate_position(later, Side.RED), evaluate_position(earlier, Side.RED))

    def test_repetition_does_not_cancel_losing_side_draw_incentive(self):
        state = position({(0, 4): -1, (9, 4): 1, (6, 0): -5, (5, 4): 7})
        repeated = GameState(
            state.board,
            state.side_to_move,
            history=(state.position_key(),) * 2,
        )
        self.assertGreater(
            evaluate_position(repeated, Side.RED),
            evaluate_position(state, Side.RED),
        )
        self.assertLess(
            evaluate_position(repeated, Side.BLACK),
            evaluate_position(state, Side.BLACK),
        )

    def test_repetition_discouraged_when_ahead(self):
        state = position({(0, 4): -1, (9, 4): 1, (6, 0): 5, (5, 4): 7})
        repeated = GameState(
            state.board,
            state.side_to_move,
            history=(state.position_key(),) * 2,
        )
        self.assertLess(
            evaluate_position(repeated, Side.RED),
            evaluate_position(state, Side.RED),
        )

    def test_agents_agree_on_a_tactical_position(self):
        state = position({(0, 4): -1, (9, 4): 1, (4, 4): 5})
        engine = GameEngine()
        alpha = AlphaBetaAgent(engine, max_depth=2).select_action(state)
        mini = MinimaxAgent(engine, max_depth=2).select_action(state)
        self.assertEqual(alpha.move, mini.move)
        self.assertLessEqual(alpha.nodes, mini.nodes)


if __name__ == "__main__":
    unittest.main()
