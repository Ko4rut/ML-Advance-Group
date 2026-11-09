"""Reusable MCTS selection, expansion, evaluation and backup.

Value convention (see node.py):

* Every value lies in [-1, 1].
* ``evaluator.evaluate(state)`` is from the perspective of ``state.side_to_move``
  (+1 means the player about to move is winning).
* ``node.W`` and ``node.N`` are stored from the perspective of the player who
  JUST moved into the node (``node.parent.state.side_to_move``). ``Q = W / N``.
* Backpropagation flips the sign before updating the leaf and again at every
  level going up, so each node sees the value from its own mover's point of view.

The engine, the selection rule and the evaluator are injected, so this module
imports neither selection.py nor evaluators.py.
"""

import math
import random
import time

from ...agents.base import SearchBudget
from ...environment import GameEngine, GameState, Move
from .node import MCTSNode


class MCTSSearch:
    """Classic MCTS: select, expand one child, evaluate, back up."""

    DEFAULT_SIMULATIONS = 400

    def __init__(
        self,
        engine: GameEngine,
        selection,
        evaluator,
        seed: int | None = None,
    ) -> None:
        self.engine = engine
        self.selection = selection
        self.evaluator = evaluator
        self.seed = seed
        self.rng = random.Random(seed)
        self.last_simulations = 0
        self.last_nodes = 0
        self.last_elapsed = 0.0
        self.last_root: MCTSNode | None = None

    def reset(self) -> None:
        self.rng.seed(self.seed)
        reset = getattr(self.evaluator, "reset", None)
        if callable(reset):
            reset()

    def search(self, state: GameState, budget: SearchBudget | None = None) -> MCTSNode:
        started = time.monotonic()
        root = MCTSNode(state)
        nodes = 1
        root.ensure_generated(self.engine)
        if root.terminal_result is not None:
            raise ValueError("cannot search from a terminal state")

        # Winning shortcut: a move that ends the game in our favour needs no search.
        for move in reversed(root.untried_moves):
            result = self.engine.outcome(state.moved(move))
            if result is not None and result.winner == state.side_to_move:
                child = root.expand({move: 1.0 / root.num_legal})[0]
                # child.W is from the view of the player who moved into it (the
                # root player, who wins); root.W is from the opponent's view.
                child.update(1.0)
                root.update(-1.0)
                self._record(started, simulations=1, nodes=2, root=root)
                return root

        max_simulations = budget.max_simulations if budget is not None else None
        time_limit = budget.time_limit if budget is not None else None
        if max_simulations is None and time_limit is None:
            max_simulations = self.DEFAULT_SIMULATIONS
        deadline = time.monotonic() + time_limit if time_limit is not None else None

        simulations = 0
        while True:
            if max_simulations is not None and simulations >= max_simulations:
                break
            # Always run at least one simulation, even with a tiny time limit.
            if deadline is not None and simulations > 0 and time.monotonic() >= deadline:
                break
            path = self.select(root)
            leaf = path[-1]
            new_node = self.expand(leaf)
            if new_node is not None:
                path.append(new_node)
                nodes += 1
                leaf = new_node
            value = self.evaluate(leaf)
            self.backpropagate(path, value)
            simulations += 1

        self._record(started, simulations=simulations, nodes=nodes, root=root)
        return root

    def select(self, node: MCTSNode) -> list[MCTSNode]:
        """Walk down to a node that is terminal or still has untried moves."""
        path = [node]
        while True:
            node.ensure_generated(self.engine)
            if node.terminal_result is not None or not node.is_fully_expanded(self.engine):
                return path
            node = self.selection.select_child(node)
            path.append(node)

    def expand(self, node: MCTSNode) -> MCTSNode | None:
        """Add exactly one child; terminal nodes cannot be expanded."""
        node.ensure_generated(self.engine)
        if node.terminal_result is not None:
            return None
        move = node.untried_moves.pop()
        return node.expand({move: 1.0 / node.num_legal})[0]

    def evaluate(self, node: MCTSNode) -> float:
        """Value from the perspective of ``node.state.side_to_move``."""
        node.ensure_generated(self.engine)
        if node.terminal_result is not None:
            # Exact result, no need to ask the evaluator.
            return float(node.terminal_result.reward(node.state.side_to_move))
        return self.evaluator.evaluate(node.state)

    def backpropagate(self, path: list[MCTSNode], value: float) -> None:
        # `value` is from the perspective of path[-1].state.side_to_move (the player ABOUT to move at the leaf).
        # node.W is stored from the perspective of the player who JUST moved into that node, i.e. the opponent.
        # So we flip the sign BEFORE updating the leaf, then flip again at every level going up.
        for node in reversed(path):
            value = -value
            node.update(value)

    def visit_policy(self, root: MCTSNode, temperature: float = 1.0) -> dict[Move, float]:
        if not root.children:
            return {}
        if temperature <= 1e-6:
            best = max(
                root.children.values(),
                key=lambda child: (child.N, child.mean_value()),
            )
            return {move: float(child is best) for move, child in root.children.items()}

        # Divide by the largest count first so a small temperature cannot overflow.
        max_visits = max(child.N for child in root.children.values())
        if max_visits == 0:
            share = 1.0 / len(root.children)
            return {move: share for move in root.children}
        weights = {
            move: (child.N / max_visits) ** (1.0 / temperature)
            for move, child in root.children.items()
        }
        total = sum(weights.values())
        return {move: weight / total for move, weight in weights.items()}

    def _record(self, started: float, simulations: int, nodes: int, root: MCTSNode) -> None:
        self.last_simulations = simulations
        self.last_nodes = nodes
        self.last_elapsed = time.monotonic() - started
        self.last_root = root
