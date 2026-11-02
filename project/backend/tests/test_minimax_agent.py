import unittest

from project.backend.src.agents import MinimaxAgent, SearchBudget
from project.backend.src.environment import GameEngine, GameState, Side, legal_moves

def position(pieces, side=Side.RED):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side)

class MinimaxAgentTests(unittest.TestCase):
    def test_selects_a_legal_move_without_mutating_state(self):
        state = GameState.initial()
        decision = MinimaxAgent(GameEngine()).select_action(state)

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(state, GameState.initial())
        self.assertGreater(decision.nodes, 0)
        self.assertGreater(decision.nodes, 1 + len(legal_moves(state)))

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

        decision = MinimaxAgent(GameEngine(), max_depth=1).select_action(state)

        result = GameEngine().outcome(state.moved(decision.move))
        self.assertIsNotNone(result)
        self.assertIs(result.winner, Side.RED)

    def test_respects_node_budget_with_a_legal_fallback(self):
        state = GameState.initial()

        decision = MinimaxAgent(GameEngine()).select_action(
            state,
            SearchBudget(max_nodes=1),
        )

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(decision.nodes, 1)

    def test_budget_depth_caps_configured_depth(self):
        state = GameState.initial()

        decision = MinimaxAgent(GameEngine(), max_depth=2).select_action(
            state,
            SearchBudget(max_depth=1),
        )

        self.assertEqual(decision.nodes, 1 + len(legal_moves(state)))

    def test_respects_time_budget_with_a_legal_fallback(self):
        state = GameState.initial()

        decision = MinimaxAgent(GameEngine()).select_action(
            state,
            SearchBudget(time_limit=1e-9),
        )

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(decision.nodes, 1)

    def test_rejects_a_state_without_legal_moves(self):
        state = GameState((0,) * 90, Side.RED)

        with self.assertRaises(ValueError):
            MinimaxAgent(GameEngine()).select_action(state)

    def test_rejects_a_drawn_state_with_legal_moves(self):
        state = GameState.initial()
        repeated = GameState(
            state.board,
            state.side_to_move,
            history=(state.position_key(),) * 3,
        )

        with self.assertRaisesRegex(ValueError, "terminal state"):
            MinimaxAgent(GameEngine()).select_action(repeated)

if __name__ == "__main__":
    unittest.main()