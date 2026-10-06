import unittest

from src.agents import RuleBasedAgent
from src.environment import GameEngine, GameState, Side, legal_moves


def position(pieces, side=Side.RED):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side)


class RuleBasedAgentTests(unittest.TestCase):
    def test_selects_a_legal_move_without_mutating_state(self):
        state = GameState.initial()
        decision = RuleBasedAgent(GameEngine()).select_action(state)

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(state, GameState.initial())

    def test_prioritizes_a_move_that_checkmates(self):
        state = position(
            {
                (0, 4): -1,
                (9, 4): 1,
                (1, 0): 5,
                (2, 3): 5,
                (2, 4): 7,
                (2, 5): 5,
            }
        )
        winning_move = next(
            move for move in legal_moves(state) if move.source == 9 and move.target == 13
        )
        self.assertEqual(GameEngine().outcome(state.moved(winning_move)).winner, Side.RED)

        decision = RuleBasedAgent(GameEngine()).select_action(state)

        decision_result = GameEngine().outcome(state.moved(decision.move))
        self.assertIsNotNone(decision_result)
        self.assertIs(decision_result.winner, Side.RED)

    def test_prefers_safe_high_value_capture_over_unsafe_capture(self):
        state = position(
            {
                (0, 5): -1,
                (9, 3): 1,
                (5, 4): 5,
                (5, 2): -7,
                (5, 6): -5,
            }
        )

        decision = RuleBasedAgent(GameEngine()).select_action(state)

        self.assertEqual(decision.move.source, 5 * 9 + 4)
        self.assertEqual(decision.move.target, 5 * 9 + 6)

    def test_rejects_a_state_without_legal_moves(self):
        state = GameState((0,) * 90, Side.RED)

        with self.assertRaises(ValueError):
            RuleBasedAgent(GameEngine()).select_action(state)


if __name__ == "__main__":
    unittest.main()
