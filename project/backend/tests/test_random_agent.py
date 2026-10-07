import unittest

from project.backend.src.agents import RandomAgent, SearchBudget
from project.backend.src.environment import GameState, legal_moves


class RandomAgentTests(unittest.TestCase):
    def test_decision_is_legal_and_has_uniform_policy(self):
        state = GameState.initial()
        moves = legal_moves(state)
        decision = RandomAgent(seed=7).select_action(state)

        self.assertIn(decision.move, moves)
        self.assertEqual(decision.nodes, 0)
        self.assertAlmostEqual(sum(decision.policy.values()), 1.0)
        self.assertEqual(set(decision.policy), set(moves))

    def test_seed_and_reset_are_reproducible(self):
        state = GameState.initial()
        first = RandomAgent(seed=11)
        second = RandomAgent(seed=11)

        self.assertEqual(first.select_action(state).move, second.select_action(state).move)
        expected = first.select_action(state).move
        first.reset()
        first.select_action(state)
        self.assertEqual(first.select_action(state).move, expected)

    def test_search_budget_rejects_non_positive_limits(self):
        with self.assertRaises(ValueError):
            SearchBudget(max_depth=0)


if __name__ == "__main__":
    unittest.main()
