"""Neural-guided MCTS: PUCT selection with network priors and network leaf values.

This is a separate class from the classic ``MCTSSearch`` (it subclasses it only to
reuse ``select``, ``backpropagate``, ``visit_policy`` and the bookkeeping); the
classic UCT + rollout code paths are not modified.

Lifecycle of one simulation (AlphaZero-style):

1. **Select** from the root with PUCT ``Q + c * P * sqrt(N_parent) / (1 + N_child)``
   (``MCTSSearch.select``: descend while a node has been expanded) until a
   *leaf*: an unexpanded node, or a terminal one.
2. **Evaluate the leaf.** Terminal leaf (engine says checkmate/stalemate,
   repetition or no-capture draw): the engine's exact result,
   ``reward(side_to_move)``; the network is never consulted. Otherwise one
   network call ``evaluator.evaluate_with_priors(state, legal_moves)`` returns
   ``(priors over the legal moves, value)`` from the leaf's side-to-move view.
3. **Expand** the leaf with *all* its legal children, each carrying the
   network's prior (``MCTSNode.expand(priors)``). Child *states* are cheap
   (~1 us); the expensive legal-move generation for a child happens only when
   that child is itself selected as a leaf.
4. **Back up** the value with ``MCTSSearch.backpropagate`` (sign flipped at every
   ply; node ``W``/``N`` are from the view of the player who moved into the node).

Consequences worth knowing: the first simulation only expands the root (so the sum
of root-child visits is ``simulations - 1``); ``last_nodes`` counts every created
node including never-visited children, so it exceeds ``simulations``; and there is
no randomness in the search (no Dirichlet noise, deterministic tie-breaks by the
engine's move order), so a deterministic network gives a reproducible search.

Single-state inference only. ``_evaluate_leaf`` is the single place that talks to
the evaluator, which is where a future batched variant (collect several leaves,
one ``evaluate_with_priors_batch`` call, expand/back up each) would plug in.
"""

from __future__ import annotations

import math
import time

from ...agents.base import SearchBudget
from ...environment import GameEngine, GameState, Move
from .node import MCTSNode
from .search import MCTSSearch
from .selection import PUCTSelection

PRIOR_SUM_TOLERANCE = 1e-6


class NeuralMCTSSearch(MCTSSearch):
    """PUCT search whose priors and leaf values come from a policy-value evaluator."""

    DEFAULT_SIMULATIONS = 100

    def __init__(
        self,
        engine: GameEngine,
        selection: PUCTSelection,
        evaluator,
        seed: int | None = None,
        *,
        root_win_shortcut: bool = True,
    ) -> None:
        if not isinstance(selection, PUCTSelection):
            raise TypeError("NeuralMCTSSearch requires PUCTSelection: UCT would ignore the network priors")
        if not callable(getattr(evaluator, "evaluate_with_priors", None)):
            raise TypeError("evaluator must provide evaluate_with_priors(state, legal_moves)")
        super().__init__(engine, selection, evaluator, seed)
        self.root_win_shortcut = root_win_shortcut
        self.last_evaluations = 0  # network calls in the last search
        self.last_terminal_leaves = 0  # leaves resolved by the engine instead of the network

    def search(self, state: GameState, budget: SearchBudget | None = None) -> MCTSNode:
        started = time.monotonic()
        root = MCTSNode(state)
        nodes = 1
        root.ensure_generated(self.engine)
        if root.terminal_result is not None:
            raise ValueError("cannot search from a terminal state")
        self.last_evaluations = 0
        self.last_terminal_leaves = 0

        if self.root_win_shortcut:
            # The raw policy does not reliably prefer a mate-in-one (Phase 1), and the
            # engine knows the answer exactly: a move that wins immediately needs no search.
            for move in reversed(root.untried_moves):
                result = self.engine.outcome(state.moved(move))
                if result is not None and result.winner == state.side_to_move:
                    child = root.expand({move: 1.0})[0]
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
            # At least one simulation always runs, even with a tiny time limit.
            if deadline is not None and simulations > 0 and time.monotonic() >= deadline:
                break
            path = self.select(root)
            value, created = self._evaluate_leaf(path[-1])
            nodes += created
            self.backpropagate(path, value)
            simulations += 1

        self._record(started, simulations=simulations, nodes=nodes, root=root)
        return root

    def _evaluate_leaf(self, leaf: MCTSNode) -> tuple[float, int]:
        """Return ``(value from leaf.state.side_to_move's view, nodes created)``."""
        leaf.ensure_generated(self.engine)
        if leaf.terminal_result is not None:
            self.last_terminal_leaves += 1
            return float(leaf.terminal_result.reward(leaf.state.side_to_move)), 0
        legal_moves = list(leaf.untried_moves)  # the engine's exact legal set
        result = self.evaluator.evaluate_with_priors(leaf.state, legal_moves)
        self.last_evaluations += 1
        priors = self._validated_priors(leaf, legal_moves, result)
        return result.value, len(leaf.expand(priors))

    @staticmethod
    def _validated_priors(leaf: MCTSNode, legal_moves: list[Move], result) -> dict[Move, float]:
        """Refuse anything but a complete, normalized prior over exactly the legal moves."""
        if getattr(result, "is_terminal", False):
            raise RuntimeError("evaluator reported a terminal state the search engine considers live; "
                               "the search and the evaluator must use the same rules")
        value = result.value
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"evaluator value {value!r} is not a finite number in [-1, 1]")
        priors = dict(result.priors)
        if set(priors) != set(legal_moves):
            raise ValueError("evaluator priors must cover exactly the legal moves of the leaf")
        if any(not math.isfinite(p) or p < 0 for p in priors.values()) \
                or not math.isclose(math.fsum(priors.values()), 1.0, abs_tol=PRIOR_SUM_TOLERANCE):
            raise ValueError("evaluator priors must be finite, non-negative and sum to 1")
        return priors

    @staticmethod
    def best_child(root: MCTSNode) -> MCTSNode:
        """Most visited child; ties by mean value, then by the network prior.

        The prior tie-break matters with a tiny budget: if no child was visited
        (1 simulation) the move played is the policy's favourite.
        """
        if not root.children:
            raise ValueError("root has no children")
        return max(root.children.values(), key=lambda c: (c.N, c.mean_value(), c.prior))
