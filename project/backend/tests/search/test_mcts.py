import unittest

from src.agents import MCTSAgent, SearchBudget
from src.environment import GameEngine, GameState, Move, Side, legal_moves
from src.search.mcts import (
    MCTSNode,
    MCTSSearch,
    RolloutEvaluator,
    UCTSelection,
)


def position(pieces, side=Side.RED):
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side)


def make_agent(engine, seed=0, policy="random", depth=0):
    evaluator = RolloutEvaluator(engine, policy=policy, max_depth=depth, seed=seed)
    return MCTSAgent(MCTSSearch(engine, UCTSelection(1.4), evaluator, seed=seed))


MATE_IN_ONE = {
    (0, 4): -1,
    (9, 4): 1,
    (1, 0): 5,
    (2, 3): 5,
    (2, 4): 7,
    (2, 5): 5,
}


class MCTSAgentTests(unittest.TestCase):
    def setUp(self):
        self.engine = GameEngine()

    def test_selects_a_legal_move_from_the_initial_position(self):
        state = GameState.initial()

        decision = make_agent(self.engine).select_action(
            state, SearchBudget(max_simulations=100)
        )

        self.assertIn(decision.move, legal_moves(state))
        self.assertEqual(decision.simulations, 100)

    def test_root_children_visits_add_up_to_simulations(self):
        agent = make_agent(self.engine)

        decision = agent.select_action(
            GameState.initial(), SearchBudget(max_simulations=150)
        )

        total = sum(child.N for child in agent.last_root.children.values())
        self.assertEqual(total, decision.simulations)
        self.assertEqual(total, 150)

    def test_finds_a_checkmate_in_one(self):
        state = position(MATE_IN_ONE)

        decision = make_agent(self.engine).select_action(
            state, SearchBudget(max_simulations=100)
        )

        result = self.engine.outcome(state.moved(decision.move))
        self.assertIsNotNone(result)
        self.assertIs(result.winner, Side.RED)

    def test_captures_an_unprotected_rook(self):
        state = position({(9, 4): 1, (0, 3): -1, (5, 0): 5, (5, 8): -5})
        capture = Move(5 * 9 + 0, 5 * 9 + 8)
        # The capture must be legal and there must be no immediate win, so the
        # winning-move shortcut cannot hide what the search itself decides.
        self.assertIn(capture, legal_moves(state))
        for move in legal_moves(state):
            result = self.engine.outcome(state.moved(move))
            self.assertFalse(result is not None and result.winner is Side.RED)

        decision = make_agent(self.engine).select_action(
            state, SearchBudget(max_simulations=200)
        )

        self.assertEqual(decision.move, capture)

    def test_value_sign_is_correct_for_both_colours(self):
        red_to_move = position({(9, 4): 1, (0, 3): -1, (5, 0): 5}, Side.RED)
        black_to_move = position({(0, 4): -1, (9, 3): 1, (4, 8): -5}, Side.BLACK)

        for state in (red_to_move, black_to_move):
            agent = make_agent(self.engine)
            agent.select_action(state, SearchBudget(max_simulations=100))
            best = max(
                agent.last_root.children.values(),
                key=lambda child: (child.N, child.mean_value()),
            )
            # Being a rook up must look good for the side that is choosing.
            self.assertGreater(best.mean_value(), 0)

    def test_same_seed_gives_the_same_result(self):
        state = GameState.initial()
        budget = SearchBudget(max_simulations=80)

        first = make_agent(self.engine, seed=7, depth=4).select_action(state, budget)
        second = make_agent(self.engine, seed=7, depth=4).select_action(state, budget)

        self.assertEqual(first.move, second.move)
        self.assertEqual(first.policy, second.policy)

    def test_does_not_mutate_state_and_rejects_terminal_state(self):
        state = GameState.initial()
        snapshot = GameState.initial()

        make_agent(self.engine).select_action(state, SearchBudget(max_simulations=20))

        self.assertEqual(state, snapshot)
        with self.assertRaises(ValueError):
            make_agent(self.engine).select_action(GameState((0,) * 90, Side.RED))


class MCTSInternalsTests(unittest.TestCase):
    def setUp(self):
        self.engine = GameEngine()
        self.search = MCTSSearch(
            self.engine, UCTSelection(1.4), RolloutEvaluator(self.engine)
        )

    def test_backpropagate_flips_sign_at_every_level(self):
        state = GameState.initial()
        moves = legal_moves(state)
        root = MCTSNode(state)
        child = MCTSNode(state.moved(moves[0]), parent=root, move=moves[0])
        grand_state = child.state.moved(legal_moves(child.state)[0])
        grand = MCTSNode(grand_state, parent=child)

        # -1 means the player about to move at `grand` is losing, so the player
        # who just moved into `grand` wins (+1), and the signs alternate upward.
        self.search.backpropagate([root, child, grand], -1.0)

        self.assertEqual((root.W, child.W, grand.W), (1.0, -1.0, 1.0))
        self.assertEqual((root.N, child.N, grand.N), (1, 1, 1))

    def test_static_evaluation_follows_the_side_to_move(self):
        pieces = {(9, 4): 1, (0, 3): -1, (5, 0): 5}
        evaluator = RolloutEvaluator(self.engine, max_depth=0)

        self.assertGreater(evaluator.evaluate(position(pieces, Side.RED)), 0)
        self.assertLess(evaluator.evaluate(position(pieces, Side.BLACK)), 0)

    def test_checkmated_side_evaluates_to_minus_one(self):
        state = position(MATE_IN_ONE)
        mating = next(
            move for move in legal_moves(state)
            if self.engine.outcome(state.moved(move)) is not None
            and self.engine.outcome(state.moved(move)).winner is Side.RED
        )
        evaluator = RolloutEvaluator(self.engine, max_depth=0)

        self.assertEqual(evaluator.evaluate(state.moved(mating)), -1.0)

    def test_visit_policy(self):
        agent = make_agent(self.engine)
        agent.select_action(GameState.initial(), SearchBudget(max_simulations=60))
        root = agent.last_root

        greedy = self.search.visit_policy(root, 0)
        best = max(root.children.values(), key=lambda c: (c.N, c.mean_value()))
        self.assertEqual(greedy[best.move], 1.0)
        self.assertEqual(sum(greedy.values()), 1.0)

        soft = self.search.visit_policy(root, 1.0)
        self.assertAlmostEqual(sum(soft.values()), 1.0)
        self.assertEqual(self.search.visit_policy(MCTSNode(GameState.initial())), {})


if __name__ == "__main__":
    unittest.main()
