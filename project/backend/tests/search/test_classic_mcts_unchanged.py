"""Classic UCT + rollout MCTS must behave exactly as before the neural integration.

The pinned values below were produced by the classic code at the commit that introduced
the neural search (Phase 4 start, which left search.py/node.py/selection.py/mcts_agent.py
untouched). Any future change to classic move choice, node counts or visit policy under a
fixed seed and simulation budget fails here. (``max_simulations`` budgets are used so the
result does not depend on wall-clock time.)
"""

import unittest

from src.agents import MCTSAgent, SearchBudget
from src.environment import GameEngine, GameState, Move, Side
from src.search.mcts import MCTSSearch, PUCTSelection, RolloutEvaluator, UCTSelection

ENGINE = GameEngine()

PINNED = {
    'initial_s7_random_d4': ([64, 37], 81, 80, [[4905, 0.0125], [5087, 0.025], [5269, 0.025], [5451, 0.0125], [5633, 0.025], [5761, 0.025], [5788, 0.025], [5797, 0.025], [5806, 0.025], [5815, 0.025], [5823, 0.025], [5825, 0.025], [5826, 0.025], [5827, 0.0125], [5828, 0.025], [5829, 0.025], [5833, 0.025], [6307, 0.025], [6334, 0.0125], [6343, 0.025], [6352, 0.025], [6361, 0.0125], [6365, 0.025], [6366, 0.025], [6367, 0.025], [6368, 0.025], [6369, 0.025], [6371, 0.025], [6379, 0.025], [7353, 0.025], [7362, 0.0125], [7443, 0.0125], [7445, 0.025], [7533, 0.025], [7537, 0.025], [7636, 0.025], [7726, 0.0125], [7816, 0.025], [7897, 0.025], [7901, 0.025], [7989, 0.025], [7991, 0.025], [8081, 0.025], [8090, 0.025]]),
    'initial_s0_capture_d2': ([88, 69], 81, 80, [[4905, 0.025], [5087, 0.025], [5269, 0.025], [5451, 0.025], [5633, 0.025], [5761, 0.0125], [5788, 0.025], [5797, 0.025], [5806, 0.025], [5815, 0.025], [5823, 0.0125], [5825, 0.025], [5826, 0.025], [5827, 0.025], [5828, 0.025], [5829, 0.0125], [5833, 0.0125], [6307, 0.025], [6334, 0.025], [6343, 0.025], [6352, 0.025], [6361, 0.025], [6365, 0.025], [6366, 0.025], [6367, 0.025], [6368, 0.025], [6369, 0.025], [6371, 0.025], [6379, 0.025], [7353, 0.025], [7362, 0.0125], [7443, 0.025], [7445, 0.025], [7533, 0.025], [7537, 0.025], [7636, 0.025], [7726, 0.025], [7816, 0.025], [7897, 0.025], [7901, 0.025], [7989, 0.025], [7991, 0.0125], [8081, 0.0125], [8090, 0.0125]]),
    'opening2_s3_random_d0': ([70, 7], 81, 80, [[4905, 0.025], [5087, 0.025], [5269, 0.025], [5451, 0.025], [5633, 0.025], [6061, 0.0375], [6093, 0.025], [6094, 0.025], [6095, 0.025], [6096, 0.025], [6098, 0.025], [6099, 0.025], [6106, 0.025], [6307, 0.0375], [6334, 0.0375], [6343, 0.0375], [6352, 0.025], [6361, 0.025], [6368, 0.0375], [6369, 0.0375], [6371, 0.0375], [6379, 0.025], [7353, 0.025], [7362, 0.025], [7443, 0.025], [7445, 0.0375], [7533, 0.025], [7636, 0.025], [7726, 0.025], [7816, 0.025], [7901, 0.025], [7989, 0.0375], [7991, 0.0375], [8081, 0.025], [8090, 0.025]]),
    'endgame_s5_random_d0': ([45, 53], 401, 400, [(4050, 0.02), (4059, 0.025), (4068, 0.03), (4077, 0.025), (4086, 0.025), (4096, 0.055), (4097, 0.03), (4098, 0.0175), (4099, 0.035), (4100, 0.035), (4101, 0.0425), (4102, 0.0325), (4103, 0.2525), (4104, 0.025), (4113, 0.0225), (4122, 0.025), (4131, 0.0225), (5896, 0.02), (5898, 0.0225), (5904, 0.0275), (5908, 0.035), (5922, 0.025), (5926, 0.035), (5932, 0.025), (5934, 0.03), (7726, 0.03), (7736, 0.03)]),
    'endgame_s2_capture_first_d2': ([45, 53], 401, 400, [(4050, 0.015), (4059, 0.02), (4068, 0.03), (4077, 0.0175), (4086, 0.0225), (4096, 0.065), (4097, 0.015), (4098, 0.02), (4099, 0.045), (4100, 0.0225), (4101, 0.0525), (4102, 0.0425), (4103, 0.16), (4104, 0.0225), (4113, 0.02), (4122, 0.0225), (4131, 0.02), (5896, 0.02), (5898, 0.0175), (5904, 0.0575), (5908, 0.0575), (5922, 0.015), (5926, 0.0275), (5932, 0.0475), (5934, 0.0575), (7726, 0.06), (7736, 0.0275)]),
}


def classic_agent(seed, policy, depth):
    evaluator = RolloutEvaluator(ENGINE, policy=policy, max_depth=depth, seed=seed)
    return MCTSAgent(MCTSSearch(ENGINE, UCTSelection(1.4), evaluator, seed=seed))


def opening2():
    state = GameState.initial()
    for source, target in ((64, 67), (18, 29)):
        state = state.moved(Move(source, target))
    return state


def endgame():
    """27 legal moves: few enough that the UCT exploration term steers 400 simulations."""
    pieces = {(9, 4): 1, (0, 3): -1, (5, 0): 5, (5, 8): -5, (7, 2): 4, (2, 6): -6}
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), Side.RED)


# name: (state factory, seed, rollout policy, rollout depth, simulations)
CASES = {
    "initial_s7_random_d4": (GameState.initial, 7, "random", 4, 80),
    "initial_s0_capture_d2": (GameState.initial, 0, "capture_first", 2, 80),
    "opening2_s3_random_d0": (opening2, 3, "random", 0, 80),
    "endgame_s5_random_d0": (endgame, 5, "random", 0, 400),
    "endgame_s2_capture_first_d2": (endgame, 2, "capture_first", 2, 400),
}


class ClassicMCTSUnchangedTests(unittest.TestCase):
    def test_pinned_classic_decisions(self):
        for name, (make_state, seed, policy, depth, simulations) in CASES.items():
            with self.subTest(name):
                decision = classic_agent(seed, policy, depth).select_action(
                    make_state(), SearchBudget(max_simulations=simulations))
                move, nodes, simulations, policy_rows = PINNED[name]
                self.assertEqual([decision.move.source, decision.move.target], move)
                self.assertEqual((decision.nodes, decision.simulations), (nodes, simulations))
                self.assertEqual(sorted((m.source * 90 + m.target, round(p, 9))
                                        for m, p in decision.policy.items()),
                                 [tuple(row) for row in policy_rows])

    def test_classic_expansion_still_uses_uniform_priors_and_one_child_per_simulation(self):
        agent = classic_agent(0, "random", 0)
        agent.select_action(GameState.initial(), SearchBudget(max_simulations=30))
        root = agent.last_root
        for child in root.children.values():
            self.assertAlmostEqual(child.prior, 1.0 / root.num_legal)
        self.assertEqual(agent.search.last_nodes, 31, "classic MCTS adds exactly one node per simulation")

    def test_classic_search_class_is_not_the_neural_one(self):
        self.assertEqual(MCTSSearch.DEFAULT_SIMULATIONS, 400)
        self.assertFalse(hasattr(MCTSSearch, "best_child"))
        # Classic selection objects keep working with the classic search.
        MCTSSearch(ENGINE, PUCTSelection(1.4), RolloutEvaluator(ENGINE))


if __name__ == "__main__":
    unittest.main()
