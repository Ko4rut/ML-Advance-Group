"""Neural MCTS lifecycle tests with FAKE evaluators (mock tests; no weights, no torch).

They prove the search machinery: priors reach PUCT at every node, leaf values are the
network's, signs alternate per ply, the engine overrides the network at terminal nodes,
budgets and determinism hold, and invalid evaluator output is rejected. Real-weight
evidence is in test_neural_mcts_real.py.
"""

import math
import unittest

from src.agents import MCTSAgent, NeuralMCTSAgent, SearchBudget
from src.environment import GameEngine, GameState, Move, Side
from src.search.mcts import (
    MCTSSearch, NeuralEvaluator, NeuralMCTSSearch, PUCTSelection, RolloutEvaluator, UCTSelection,
)

from .neural_support import (
    ENGINE, FakeEvaluator, MATE_IN_ONE_BLACK, MATE_IN_ONE_RED, StubEvaluator, StubResult, all_nodes,
    fake_adapter, golden_states, material_value, normalized, position,
)


def make_search(evaluator, c_puct=1.5, **kwargs):
    return NeuralMCTSSearch(ENGINE, PUCTSelection(c_puct), evaluator, **kwargs)


def black_initial():
    return GameState(GameState.initial().board, Side.BLACK)


class ConstructionTests(unittest.TestCase):
    def test_refuses_uct_and_evaluators_without_priors(self):
        with self.assertRaises(TypeError):
            NeuralMCTSSearch(ENGINE, UCTSelection(1.4), FakeEvaluator())
        with self.assertRaises(TypeError):
            NeuralMCTSSearch(ENGINE, PUCTSelection(1.4), RolloutEvaluator(ENGINE))
        with self.assertRaises(TypeError):
            NeuralEvaluator(object())

    def test_agent_refuses_a_classic_search_and_a_missing_adapter(self):
        classic = MCTSSearch(ENGINE, UCTSelection(1.4), RolloutEvaluator(ENGINE))
        with self.assertRaises(TypeError):
            NeuralMCTSAgent(classic)
        with self.assertRaises(ValueError):
            NeuralMCTSAgent.from_config(ENGINE, {})

    def test_classic_agent_is_not_a_neural_agent(self):
        self.assertFalse(issubclass(MCTSAgent, NeuralMCTSAgent))
        self.assertTrue(issubclass(NeuralMCTSAgent, MCTSAgent))


class LifecycleAccountingTests(unittest.TestCase):
    def run_search(self, simulations, state=None):
        evaluator = FakeEvaluator(weight=lambda s, m: 1.0 + (m.target % 5))
        search = make_search(evaluator)
        root = search.search(state or GameState.initial(), SearchBudget(max_simulations=simulations))
        return search, evaluator, root

    def test_simulation_counts_for_1_2_and_100(self):
        for sims in (1, 2, 100):
            with self.subTest(simulations=sims):
                search, evaluator, root = self.run_search(sims)
                self.assertEqual(search.last_simulations, sims)
                self.assertEqual(root.N, sims)
                # The first simulation only evaluates and expands the root.
                self.assertEqual(sum(c.N for c in root.children.values()), sims - 1)
                self.assertEqual(len(evaluator.calls), search.last_evaluations)
                self.assertEqual(search.last_evaluations + search.last_terminal_leaves, sims)

    def test_node_accounting_matches_the_tree(self):
        search, _, root = self.run_search(60)
        self.assertEqual(search.last_nodes, sum(1 for _ in all_nodes(root)))
        self.assertGreater(search.last_nodes, search.last_simulations, "all legal children are created per expansion")

    def test_single_simulation_expands_only_the_root_with_network_priors(self):
        _, evaluator, root = self.run_search(1)
        moves = ENGINE.legal_moves(GameState.initial())
        self.assertEqual(set(root.children), set(moves))
        self.assertEqual(len(evaluator.calls), 1)
        expected = evaluator.expected_priors(GameState.initial(), moves)
        for move, child in root.children.items():
            self.assertAlmostEqual(child.prior, expected[move], places=12)
            self.assertEqual(child.N, 0)

    def test_evaluator_sees_each_position_once_and_only_nonterminal_ones(self):
        search, evaluator, root = self.run_search(120)
        keys = [(s.board, s.side_to_move, s.history) for s, _ in evaluator.calls]
        self.assertEqual(len(set(keys)), len(keys), "a node must not be evaluated twice")
        expanded = [n for n in all_nodes(root) if n.children]
        self.assertEqual(len(expanded), len(evaluator.calls))

    def test_does_not_mutate_the_input_and_rejects_terminal_roots(self):
        state = GameState.initial()
        snapshot = GameState.initial()
        make_search(FakeEvaluator()).search(state, SearchBudget(max_simulations=10))
        self.assertEqual(state, snapshot)
        mated = MATE_IN_ONE_RED.moved(next(
            m for m in ENGINE.legal_moves(MATE_IN_ONE_RED)
            if ENGINE.outcome(MATE_IN_ONE_RED.moved(m)) is not None))
        for terminal in (mated, GameState((0,) * 90, Side.RED)):
            with self.assertRaises(ValueError):
                make_search(FakeEvaluator()).search(terminal)


class PriorsReachPUCTTests(unittest.TestCase):
    def test_priors_are_set_from_the_network_at_every_expanded_node(self):
        evaluator = FakeEvaluator(weight=lambda s, m: 1.0 + (m.source * 7 + m.target) % 11)
        search = make_search(evaluator)
        root = search.search(GameState.initial(), SearchBudget(max_simulations=150))
        expanded = [n for n in all_nodes(root) if n.children]
        self.assertGreater(len(expanded), 20)
        distinct = set()
        for node in expanded:
            moves = ENGINE.legal_moves(node.state)
            expected = evaluator.expected_priors(node.state, moves)
            self.assertEqual(set(node.children), set(moves))
            for move, child in node.children.items():
                self.assertAlmostEqual(child.prior, expected[move], places=12)
                distinct.add(round(child.prior, 9))
            uniform = 1.0 / len(moves)
            self.assertTrue(any(abs(c.prior - uniform) > 1e-3 for c in node.children.values()),
                            "priors must not be the classic uniform 1/num_legal")
        self.assertGreater(len(distinct), 10)

    def test_a_policy_favourite_attracts_visits_when_values_are_neutral(self):
        state = GameState.initial()
        favourite = ENGINE.legal_moves(state)[5]
        biased = FakeEvaluator(weight=lambda s, m: 60.0 if (s == state and m == favourite) else 1.0)
        uniform = FakeEvaluator()
        shares = {}
        for name, evaluator in (("biased", biased), ("uniform", uniform)):
            root = make_search(evaluator).search(state, SearchBudget(max_simulations=100))
            shares[name] = root.children[favourite].N / 99
        self.assertGreater(shares["biased"], 0.5)
        self.assertLess(shares["uniform"], 0.2)

    def test_changing_the_policy_changes_the_chosen_move(self):
        state = GameState.initial()
        moves = ENGINE.legal_moves(state)
        for favourite in (moves[3], moves[17], moves[-1]):
            evaluator = FakeEvaluator(weight=lambda s, m, f=favourite: 100.0 if (s == state and m == f) else 1.0)
            agent = NeuralMCTSAgent(make_search(evaluator))
            self.assertEqual(agent.select_action(state, SearchBudget(max_simulations=30)).move, favourite)

    def test_with_one_simulation_the_agent_plays_the_policy_argmax(self):
        state = GameState.initial()
        moves = ENGINE.legal_moves(state)
        favourite = moves[11]
        evaluator = FakeEvaluator(weight=lambda s, m: 9.0 if m == favourite else 1.0)
        decision = NeuralMCTSAgent(make_search(evaluator)).select_action(state, SearchBudget(max_simulations=1))
        self.assertEqual(decision.move, favourite)


class ValueTests(unittest.TestCase):
    def test_leaf_value_sign_and_per_ply_flip_for_both_colours(self):
        for state in (GameState.initial(), black_initial()):
            with self.subTest(side=state.side_to_move.name):
                search = make_search(FakeEvaluator(value=lambda s: 0.5))
                root = search.search(state, SearchBudget(max_simulations=2))
                visited = [c for c in root.children.values() if c.N]
                self.assertEqual(len(visited), 1)
                child = visited[0]
                # Sim 1: root leaf, value +0.5 for its side to move -> W is the mover-into-root's view: -0.5.
                # Sim 2: child leaf +0.5 for ITS side to move -> child.W = -0.5 (its mover is the root's side),
                #        and the signal flips again at the root: +0.5.
                self.assertEqual((child.N, child.W), (1, -0.5))
                self.assertEqual((root.N, root.W), (2, 0.0))

    def test_a_material_up_side_looks_good_for_whoever_is_choosing(self):
        red = position({(9, 4): 1, (0, 3): -1, (5, 0): 5, (5, 8): -2}, Side.RED)
        black = position({(0, 4): -1, (9, 3): 1, (4, 8): -5, (4, 0): 2}, Side.BLACK)
        for state in (red, black):
            search = make_search(FakeEvaluator(value=material_value))
            root = search.search(state, SearchBudget(max_simulations=80))
            best = search.best_child(root)
            self.assertGreater(best.mean_value(), 0, state.side_to_move.name)

    def test_a_losing_material_balance_is_negative_for_the_chooser(self):
        down = position({(9, 4): 1, (0, 3): -1, (5, 0): -5, (6, 0): -5}, Side.RED)
        search = make_search(FakeEvaluator(value=material_value))
        root = search.search(down, SearchBudget(max_simulations=60))
        self.assertLess(search.best_child(root).mean_value(), 0)


class TerminalPrecedenceTests(unittest.TestCase):
    def test_engine_wins_over_a_network_that_loves_every_position(self):
        # The "network" says the side to move is winning everywhere (+1). After the mating move
        # the side to move is checkmated; only the engine can say -1 there.
        for state in (MATE_IN_ONE_RED, MATE_IN_ONE_BLACK):
            with self.subTest(side=state.side_to_move.name):
                evaluator = FakeEvaluator(value=lambda s: 1.0)
                search = make_search(evaluator, root_win_shortcut=False)
                root = search.search(state, SearchBudget(max_simulations=80))
                best = search.best_child(root)
                result = ENGINE.outcome(state.moved(best.move))
                self.assertIsNotNone(result)
                self.assertIs(result.winner, state.side_to_move)
                self.assertEqual(best.mean_value(), 1.0)
                self.assertGreater(search.last_terminal_leaves, 0)
                for call_state, _ in evaluator.calls:  # FakeEvaluator also asserts this itself
                    self.assertIsNone(ENGINE.outcome(call_state))

    def test_terminal_children_use_exact_engine_rewards(self):
        search = make_search(FakeEvaluator(value=lambda s: 0.0), root_win_shortcut=False)
        root = search.search(MATE_IN_ONE_RED, SearchBudget(max_simulations=60))
        terminal = [c for c in root.children.values() if c.terminal_result is not None]
        self.assertTrue(terminal)
        for child in terminal:
            self.assertEqual(child.W, float(child.N), "mate is +1 for the mating side on every visit")

    def test_root_win_shortcut_needs_no_network_call_for_either_colour(self):
        for state in (MATE_IN_ONE_RED, MATE_IN_ONE_BLACK):
            evaluator = FakeEvaluator()
            decision = NeuralMCTSAgent(make_search(evaluator)).select_action(state, SearchBudget(max_simulations=50))
            result = ENGINE.outcome(state.moved(decision.move))
            self.assertIs(result.winner, state.side_to_move)
            self.assertEqual(evaluator.calls, [])

    def test_draw_rules_still_come_from_the_engine(self):
        from dataclasses import replace
        near_draw = replace(GameState.initial(), no_capture_count=119)
        evaluator = FakeEvaluator(value=lambda s: 0.9)
        search = make_search(evaluator)
        root = search.search(near_draw, SearchBudget(max_simulations=60))
        # A quiet move makes no_capture_count 120: a drawn game (value 0), never sent to the network.
        # A capture resets the counter, so those children are live positions.
        quiet = captures = 0
        for move, child in root.children.items():
            if near_draw.board[move.target] == 0:
                quiet += 1
                self.assertEqual(child.state.no_capture_count, 120)
                self.assertEqual(ENGINE.outcome(child.state).reason, "no_capture_limit")
                if child.N:
                    self.assertEqual(child.mean_value(), 0.0, "engine draw, not the network's 0.9")
            else:
                captures += 1
                self.assertIsNone(ENGINE.outcome(child.state))
        self.assertTrue(quiet and captures)
        self.assertGreater(search.last_terminal_leaves, 0)
        for call_state, _ in evaluator.calls:
            self.assertLess(call_state.no_capture_count, 120)


class LegalityTests(unittest.TestCase):
    def test_every_node_holds_exactly_its_legal_moves(self):
        states = [GameState.initial(), black_initial()] + [s for _, s in golden_states(step=25)]
        for state in states:
            search = make_search(FakeEvaluator(weight=lambda s, m: 1.0 + m.target % 3))
            root = search.search(state, SearchBudget(max_simulations=40))
            for node in all_nodes(root):
                if node.children:
                    legal = ENGINE.legal_moves(node.state)
                    self.assertEqual(set(node.children), set(legal))
                for move, child in node.children.items():
                    self.assertEqual(child.state, node.state.moved(move))
                    self.assertIs(child.parent, node)

    def test_agent_moves_are_legal_in_a_sample_of_golden_positions(self):
        for name, state in golden_states(step=12):
            if ENGINE.outcome(state) is not None:
                continue
            agent = NeuralMCTSAgent(make_search(FakeEvaluator(weight=lambda s, m: 1.0 + m.source % 4)))
            decision = agent.select_action(state, SearchBudget(max_simulations=20))
            self.assertIn(decision.move, ENGINE.legal_moves(state), name)
            self.assertAlmostEqual(math.fsum(decision.policy.values()), 1.0, places=9)


class BudgetAndDeterminismTests(unittest.TestCase):
    def test_time_limit_stops_the_search_and_always_runs_one_simulation(self):
        evaluator = FakeEvaluator(delay=0.004)
        search = make_search(evaluator)
        search.search(GameState.initial(), SearchBudget(time_limit=0.08))
        self.assertGreaterEqual(search.last_simulations, 5)
        self.assertLess(search.last_simulations, 40)
        self.assertLess(search.last_elapsed, 0.08 + 0.08)
        tiny = make_search(FakeEvaluator())
        tiny.search(GameState.initial(), SearchBudget(time_limit=1e-9))
        self.assertEqual(tiny.last_simulations, 1)

    def test_simulation_cap_wins_over_a_generous_time_limit(self):
        search = make_search(FakeEvaluator())
        search.search(GameState.initial(), SearchBudget(max_simulations=7, time_limit=60.0))
        self.assertEqual(search.last_simulations, 7)

    def test_default_budget_is_bounded(self):
        search = make_search(FakeEvaluator())
        search.search(GameState.initial())
        self.assertEqual(search.last_simulations, NeuralMCTSSearch.DEFAULT_SIMULATIONS)

    def test_repeated_searches_are_identical_and_seed_independent(self):
        def run(seed):
            evaluator = FakeEvaluator(weight=lambda s, m: 1.0 + (m.source + m.target) % 7, value=material_value)
            search = make_search(evaluator, seed=seed)
            root = search.search(black_initial(), SearchBudget(max_simulations=120))
            return ([(m.source, m.target, c.N, round(c.W, 12)) for m, c in root.children.items()],
                    search.visit_policy(root, 1.0), [(s.board, s.side_to_move) for s, _ in evaluator.calls])
        self.assertEqual(run(0), run(0))
        self.assertEqual(run(0), run(12345), "the neural search uses no randomness")


class InvalidEvaluatorOutputTests(unittest.TestCase):
    def search_with(self, make_result):
        return make_search(StubEvaluator(make_result)).search(GameState.initial(), SearchBudget(max_simulations=3))

    def test_malformed_results_are_rejected_not_repaired(self):
        uniform = lambda moves: {m: 1.0 / len(moves) for m in moves}
        cases = {
            "missing a legal move": lambda s, m: StubResult(normalized({x: 1.0 for x in m[1:]})),
            "extra illegal move": lambda s, m: StubResult(normalized({**{x: 1.0 for x in m}, Move(0, 1): 1.0})),
            "not normalized": lambda s, m: StubResult({x: 1.0 for x in m}),
            "negative prior": lambda s, m: StubResult({**uniform(m), m[0]: -0.5}),
            "nan prior": lambda s, m: StubResult({**uniform(m), m[0]: math.nan}),
            "empty priors": lambda s, m: StubResult({}),
        }
        for name, make in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.search_with(make)

    def test_bad_values_are_rejected(self):
        for bad in (math.nan, math.inf, 1.5, -1.0001, "0.3", None, True):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.search_with(lambda s, m, b=bad: StubResult({x: 1.0 / len(m) for x in m}, value=b))

    def test_an_evaluator_that_calls_a_live_state_terminal_is_a_loud_error(self):
        with self.assertRaises(RuntimeError):
            self.search_with(lambda s, m: StubResult({}, 0.0, is_terminal=True))

    def test_evaluator_exceptions_propagate(self):
        def boom(s, m):
            raise OSError("network down")
        with self.assertRaises(OSError):
            self.search_with(boom)


class AdapterIntegrationTests(unittest.TestCase):
    """NeuralEvaluator over the real PolicyValueAdapter class (with a fake backend)."""

    def test_the_adapter_is_really_invoked_once_per_nonterminal_expansion(self):
        adapter = fake_adapter(value=0.2)
        search = NeuralMCTSSearch(ENGINE, PUCTSelection(1.5), NeuralEvaluator(adapter))
        search.search(GameState.initial(), SearchBudget(max_simulations=40))
        self.assertEqual(adapter.inference_calls, search.last_evaluations)
        self.assertEqual(adapter.backend.calls, search.last_evaluations)
        self.assertGreater(adapter.inference_calls, 5)

    def test_agent_from_config_exposes_model_identity(self):
        adapter = fake_adapter()
        agent = NeuralMCTSAgent.from_config(ENGINE, {"c_puct": 2.0, "seed": 3}, adapter)
        self.assertEqual(agent.model_info.name, "fake")
        self.assertEqual(agent.search.selection.exploration_constant, 2.0)
        decision = agent.select_action(GameState.initial(), SearchBudget(max_simulations=10))
        self.assertIn(decision.move, ENGINE.legal_moves(GameState.initial()))
        self.assertEqual(adapter.inference_calls, agent.search.last_evaluations)

    def test_adapter_terminal_results_are_not_counted_as_network_calls(self):
        adapter = fake_adapter()
        search = NeuralMCTSSearch(ENGINE, PUCTSelection(1.5), NeuralEvaluator(adapter), root_win_shortcut=False)
        search.search(MATE_IN_ONE_RED, SearchBudget(max_simulations=50))
        self.assertEqual(adapter.inference_calls, search.last_evaluations)
        self.assertGreater(search.last_terminal_leaves, 0)


class AgentTests(unittest.TestCase):
    def test_decision_shape_and_reset(self):
        agent = NeuralMCTSAgent(make_search(FakeEvaluator()))
        state = GameState.initial()
        decision = agent.select_action(state, SearchBudget(max_simulations=25))
        self.assertEqual(decision.simulations, 25)
        self.assertGreater(decision.nodes, 25)
        self.assertEqual(set(decision.policy), set(ENGINE.legal_moves(state)))
        self.assertIsNotNone(agent.last_root)
        agent.reset()
        self.assertIsNone(agent.last_root)

    def test_rejects_terminal_states(self):
        agent = NeuralMCTSAgent(make_search(FakeEvaluator()))
        with self.assertRaises(ValueError):
            agent.select_action(GameState((0,) * 90, Side.RED))


if __name__ == "__main__":
    unittest.main()
