"""Neural MCTS with the REAL pretrained networks (distill-v4-large, distill-v4). Not mock tests.

Needs torch and the provisioned, SHA-256-pinned checkpoints (models/neural/README.md).
Without them the classes SKIP with an explicit reason; a skip is NOT verification. In the
model-validation environment set XIANGQI_NEURAL_REQUIRE_REAL=1 so a missing runtime or
checkpoint is an ERROR instead.
"""

import importlib.util
import os
import unittest

from src.agents import MCTSAgent, NeuralMCTSAgent, SearchBudget
from src.environment import GameEngine, GameState, Side
from src.networks.pretrained import NeuralModelSettings, load_adapter
from src.search.mcts import (
    MCTSSearch, NeuralEvaluator, NeuralMCTSSearch, PUCTSelection, RolloutEvaluator, UCTSelection,
)

from .neural_support import ENGINE, MATE_IN_ONE_BLACK, MATE_IN_ONE_RED, all_nodes, position

REQUIRE_REAL = os.environ.get("XIANGQI_NEURAL_REQUIRE_REAL") == "1"
HAVE_TORCH = importlib.util.find_spec("torch") is not None


def piece_removed(state, piece, side):
    return GameState(tuple(0 if p == piece else p for p in state.board), side)


def neural_agent(adapter, c_puct=1.5, shortcut=True):
    search = NeuralMCTSSearch(ENGINE, PUCTSelection(c_puct), NeuralEvaluator(adapter), root_win_shortcut=shortcut)
    return NeuralMCTSAgent(search)


class RealNeuralMCTSCases:
    variant: str

    @classmethod
    def setUpClass(cls):
        path = NeuralModelSettings(model=cls.variant).resolve().path
        problem = None if HAVE_TORCH and path.is_file() else (
            "PyTorch is not installed" if not HAVE_TORCH else f"checkpoint not provisioned at {path}")
        if problem:
            if REQUIRE_REAL:
                raise AssertionError(f"XIANGQI_NEURAL_REQUIRE_REAL=1 but {problem}")
            raise unittest.SkipTest(f"{problem}; real-weight search tests for {cls.variant} NOT run "
                                    "(a skip is not verification)")
        cls.adapter = load_adapter(NeuralModelSettings(model=cls.variant))

    def test_the_real_network_supplies_the_priors_and_is_invoked(self):
        state = GameState.initial()
        agent = neural_agent(self.adapter)
        before = self.adapter.inference_calls
        agent.select_action(state, SearchBudget(max_simulations=30))
        calls = self.adapter.inference_calls - before
        self.assertEqual(calls, agent.search.last_evaluations)
        self.assertGreaterEqual(calls, 5)
        reference = self.adapter.predict(state)
        root = agent.last_root
        self.assertEqual(set(root.children), set(reference.priors))
        for move, child in root.children.items():
            self.assertEqual(child.prior, reference.priors[move], "root priors are the network's, bit for bit")
        uniform = 1.0 / len(reference.priors)
        self.assertGreater(max(abs(c.prior - uniform) for c in root.children.values()), 0.05)
        # Priors are also the network's below the root.
        expanded = [n for n in all_nodes(root) if n.children and n is not root]
        self.assertTrue(expanded)
        node = expanded[0]
        for move, p in self.adapter.predict(node.state).priors.items():
            self.assertEqual(node.children[move].prior, p)

    def test_search_is_reproducible_with_the_real_network(self):
        state = GameState.initial()
        runs = []
        for _ in range(2):
            decision = neural_agent(self.adapter).select_action(state, SearchBudget(max_simulations=25))
            runs.append((decision.move, decision.nodes, sorted(decision.policy.items(), key=lambda kv: (kv[0].source, kv[0].target))))
        self.assertEqual(runs[0], runs[1])

    def test_value_sign_with_the_real_network_for_both_colours(self):
        initial = GameState.initial()
        for mover in (Side.RED, Side.BLACK):
            ahead = piece_removed(initial, -5 if mover is Side.RED else 5, mover)  # opponent lost both rooks
            agent = neural_agent(self.adapter)
            agent.select_action(ahead, SearchBudget(max_simulations=20))
            best = agent.search.best_child(agent.last_root)
            self.assertGreater(best.mean_value(), 0.3, f"{mover.name} is two rooks up")
            behind = piece_removed(initial, 5 if mover is Side.RED else -5, mover)  # mover lost both rooks
            agent = neural_agent(self.adapter)
            agent.select_action(behind, SearchBudget(max_simulations=20))
            self.assertLess(agent.search.best_child(agent.last_root).mean_value(), -0.3, f"{mover.name} is two rooks down")

    def test_mate_in_one_shortcut_and_exact_terminal_values_for_both_colours(self):
        for state in (MATE_IN_ONE_RED, MATE_IN_ONE_BLACK):
            before = self.adapter.inference_calls
            decision = neural_agent(self.adapter).select_action(state, SearchBudget(max_simulations=50))
            self.assertIs(ENGINE.outcome(state.moved(decision.move)).winner, state.side_to_move)
            self.assertEqual(self.adapter.inference_calls, before, "the shortcut needs no network call")
            # Without the shortcut the engine's exact terminal result must still reach the tree: some
            # mating child is visited and carries exactly +1 per visit (the network's own value for the
            # position is merely close to +1). Whether the mate is then the MOST visited move is NOT
            # guaranteed - near-saturated values spread visits among winning-looking moves (see
            # docs/neural_mcts/04_SEARCH_LIFECYCLE.md); that is why the root shortcut exists.
            agent = neural_agent(self.adapter, shortcut=False)
            agent.select_action(state, SearchBudget(max_simulations=150))
            mating = [c for c in agent.last_root.children.values()
                      if c.terminal_result is not None and c.terminal_result.winner is state.side_to_move]
            visited = [c for c in mating if c.N]
            self.assertTrue(visited, f"{state.side_to_move.name}: no mating child was ever visited")
            for child in visited:
                self.assertEqual(child.W, float(child.N))
            self.assertGreater(agent.search.last_terminal_leaves, 0)

    def test_self_play_game_prefix_has_only_legal_moves_for_both_sides(self):
        state, agent, sides = GameState.initial(), neural_agent(self.adapter), set()
        for ply in range(24):
            if ENGINE.outcome(state) is not None:
                break
            decision = agent.select_action(state, SearchBudget(max_simulations=15))
            sides.add(state.side_to_move)
            state = ENGINE.apply_move(state, decision.move)  # raises if illegal
        self.assertEqual(sides, {Side.RED, Side.BLACK})

    def test_neural_versus_classic_mcts_plays_legally_with_either_colour(self):
        classic_search = MCTSSearch(ENGINE, UCTSelection(1.4), RolloutEvaluator(ENGINE, max_depth=0, seed=0), seed=0)
        for neural_side in (Side.RED, Side.BLACK):
            players = {neural_side: neural_agent(self.adapter), neural_side.opponent: MCTSAgent(classic_search)}
            state = GameState.initial()
            for _ in range(16):
                if ENGINE.outcome(state) is not None:
                    break
                decision = players[state.side_to_move].select_action(state, SearchBudget(max_simulations=15))
                state = ENGINE.apply_move(state, decision.move)

    def test_time_budget_is_respected(self):
        agent = neural_agent(self.adapter)
        agent.select_action(GameState.initial(), SearchBudget(time_limit=0.4))
        self.assertGreaterEqual(agent.search.last_simulations, 2)
        self.assertLess(agent.search.last_elapsed, 0.4 + 0.15)


class RealNeuralMCTSDistillV4LargeTests(RealNeuralMCTSCases, unittest.TestCase):
    variant = "distill-v4-large"


class RealNeuralMCTSDistillV4Tests(RealNeuralMCTSCases, unittest.TestCase):
    variant = "distill-v4"


if __name__ == "__main__":
    unittest.main()
