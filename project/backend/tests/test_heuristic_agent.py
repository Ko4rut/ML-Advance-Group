import unittest

from project.backend.src.agents.base import SearchBudget
from project.backend.src.agents.heuristic import HeuristicAgent
from project.backend.src.environment import GameEngine, GameState, Side, legal_moves


def position(pieces, side=Side.RED):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side)


class HeuristicAgentTests(unittest.TestCase):
    def test_selects_legal_move_without_mutating_and_reports_metrics(self):
        state = GameState.initial()
        agent = HeuristicAgent(GameEngine(), depth=1)

        decision = agent.select_action(state)

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(state, GameState.initial())
        self.assertEqual(decision.nodes, agent.last_nodes)
        self.assertIsInstance(agent.last_score, float)
        self.assertGreaterEqual(agent.last_duration_seconds, 0)
        self.assertEqual(agent.last_completed_depth, 1)

    def test_takes_immediate_checkmate(self):
        engine = GameEngine()
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

        decision = HeuristicAgent(engine).select_action(state)

        result = engine.outcome(state.moved(decision.move))
        self.assertIsNotNone(result)
        self.assertIs(result.winner, Side.RED)

    def test_supports_opponent_aware_depth_two_search(self):
        state = position({(0, 3): -1, (9, 3): 1})
        agent = HeuristicAgent(GameEngine(), depth=2)

        decision = agent.select_action(state)

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(agent.last_completed_depth, 2)

    def test_node_budget_returns_a_legal_move(self):
        state = GameState.initial()
        agent = HeuristicAgent(GameEngine(), depth=3)

        decision = agent.select_action(state, SearchBudget(max_nodes=2))

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(decision.nodes, 2)
        self.assertEqual(agent.last_completed_depth, 0)

    def test_rejects_terminal_state(self):
        state = GameState((0,) * 90, Side.RED)

        with self.assertRaises(ValueError):
            HeuristicAgent(GameEngine()).select_action(state)


if __name__ == "__main__":
    unittest.main()
