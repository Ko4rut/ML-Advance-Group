import unittest

from project.backend.src.agents import HeuristicAgent, SearchBudget
from project.backend.src.agents.position_evaluator import evaluate_position
from project.backend.src.environment import GameEngine, GameState, Side, legal_moves


def position(pieces, side=Side.RED):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side)


class HeuristicAgentTests(unittest.TestCase):
    def test_default_evaluator_matches_shared_position_evaluator(self):
        state = position(
            {
                (0, 4): -1,
                (9, 4): 1,
                (6, 0): 5,
                (5, 4): 7,
                (7, 2): 4,
            }
        )
        agent = HeuristicAgent(GameEngine())

        for side in (Side.RED, Side.BLACK):
            with self.subTest(side=side):
                self.assertEqual(
                    agent._evaluate(state, side),
                    evaluate_position(state, side),
                )

    def test_finds_immediate_checkmate_using_astar(self):
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
        agent = HeuristicAgent(engine)

        decision = agent.select_action(state)

        result = engine.outcome(state.moved(decision.move))
        self.assertIsNotNone(result)
        self.assertIs(result.winner, Side.RED)
        self.assertEqual(agent.last_cost, 1)
        self.assertGreater(decision.nodes, 0)

    def test_returns_legal_move_without_mutating_state(self):
        state = GameState.initial()
        agent = HeuristicAgent(GameEngine())

        decision = agent.select_action(state)

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(state, GameState.initial())

    def test_node_budget_returns_legal_fallback(self):
        state = GameState.initial()
        agent = HeuristicAgent(GameEngine(), max_depth=8)

        decision = agent.select_action(state, SearchBudget(max_nodes=1))

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(decision.nodes, 1)
        self.assertIsNone(agent.last_cost)

    def test_rejects_terminal_state(self):
        state = GameState((0,) * 90, Side.RED)

        with self.assertRaises(ValueError):
            HeuristicAgent(GameEngine()).select_action(state)


if __name__ == "__main__":
    unittest.main()
