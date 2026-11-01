import unittest

from project.backend.src.agents import AlphaBetaAgent, MinimaxAgent, SearchBudget
from project.backend.src.environment import GameEngine, GameState, Side, legal_moves

def position(pieces, side=Side.RED):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side)

class AlphaBetaAgentTests(unittest.TestCase):
    def test_is_separate_and_prunes_nodes_vs_minimax(self):
        state = position(
            {
                (0, 4): -1,
                (9, 4): 1,
                (4, 4): 5,
            }
        )
        engine = GameEngine()

        minimax = MinimaxAgent(engine, max_depth=2).select_action(state)
        alpha_beta = AlphaBetaAgent(engine, max_depth=2).select_action(state)

        self.assertFalse(issubclass(AlphaBetaAgent, MinimaxAgent))
        self.assertEqual(alpha_beta.move, minimax.move)
        self.assertLess(alpha_beta.nodes, minimax.nodes)

    def test_selects_a_legal_move_and_searches_opponent_reply(self):
        state = GameState.initial()

        decision = AlphaBetaAgent(GameEngine()).select_action(state)

        self.assertIn(decision.move, legal_moves(state))
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

        decision = AlphaBetaAgent(GameEngine()).select_action(
            state,
            SearchBudget(max_depth=1),
        )

        result = GameEngine().outcome(state.moved(decision.move))
        self.assertIsNotNone(result)
        self.assertIs(result.winner, Side.RED)

    def test_respects_node_budget_with_a_legal_fallback(self):
        state = GameState.initial()

        decision = AlphaBetaAgent(GameEngine()).select_action(
            state,
            SearchBudget(max_nodes=1),
        )

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(decision.nodes, 1)

    def test_rejects_a_state_without_legal_moves(self):
        state = GameState((0,) * 90, Side.RED)

        with self.assertRaises(ValueError):
            AlphaBetaAgent(GameEngine()).select_action(state)

if __name__ == "__main__":
    unittest.main()