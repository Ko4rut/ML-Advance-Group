"""Tree node and visit/value statistics.

Value convention (shared with selection.py and search.py):

* Every value lies in [-1, 1].
* ``evaluator.evaluate(state)`` is from the perspective of ``state.side_to_move``
  (+1 means the player about to move is winning).
* ``node.W`` and ``node.N`` are stored from the perspective of the player who
  JUST moved into the node, i.e. ``node.parent.state.side_to_move`` (the
  opponent of ``node.state.side_to_move``). ``Q = W / N``.
* Therefore, at a parent, ``child.mean_value()`` is already from the point of
  view of the player choosing among the children, and selection can simply
  take the maximum without flipping any sign.
"""

from collections.abc import Iterable, Mapping

from ...agents.position_evaluator import order_moves
from ...environment import GameEngine, GameResult, GameState, Move


class MCTSNode:
    """One position in the search tree plus its visit/value statistics."""

    def __init__(
        self,
        state: GameState,
        parent: "MCTSNode | None" = None,
        prior: float = 0.0,
        move: Move | None = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.prior = prior
        # The move that led from ``parent.state`` to ``state`` (None at the root).
        self.move = move
        self.children: dict[Move, MCTSNode] = {}
        # None means "legal moves not generated yet"; generation is lazy because
        # it is the most expensive operation of the engine.
        self.untried_moves: list[Move] | None = None
        self.terminal_result: GameResult | None = None
        self.num_legal = 0
        self._generated = False
        self.N = 0
        self.W = 0.0

    def ensure_generated(self, engine: GameEngine) -> None:
        """Generate legal moves and the terminal result once, on first use."""
        if self._generated:
            return
        moves = engine.legal_moves(self.state)
        self.terminal_result = engine.outcome(self.state, legal_moves=moves)
        self.num_legal = len(moves)
        if self.terminal_result is not None:
            self.untried_moves = []
        else:
            # Reversed so that ``pop()`` returns the highest-priority move first
            # (big captures first). This helps MCTS see tactics on a small budget.
            self.untried_moves = list(reversed(order_moves(self.state, moves)))
        self._generated = True

    def is_terminal(self, engine: GameEngine) -> bool:
        self.ensure_generated(engine)
        return self.terminal_result is not None

    def is_fully_expanded(self, engine: GameEngine) -> bool:
        self.ensure_generated(engine)
        return self.terminal_result is None and not self.untried_moves

    def expand(
        self,
        action_priors: Mapping[Move, float] | Iterable[tuple[Move, float]],
    ) -> list["MCTSNode"]:
        """Create one child per (move, prior) pair and return the new children."""
        pairs = (
            action_priors.items()
            if isinstance(action_priors, Mapping)
            else action_priors
        )
        created: list[MCTSNode] = []
        for move, prior in pairs:
            if move in self.children:
                continue
            child = MCTSNode(
                self.state.moved(move),
                parent=self,
                prior=prior,
                move=move,
            )
            self.children[move] = child
            if self.untried_moves and move in self.untried_moves:
                self.untried_moves.remove(move)
            created.append(child)
        return created

    def update(self, value: float) -> None:
        """Add one visit; ``value`` is already from this node's perspective."""
        self.N += 1
        self.W += value

    def mean_value(self) -> float:
        if self.N == 0:
            return 0.0
        return self.W / self.N

    def __repr__(self) -> str:
        return f"MCTSNode(move={self.move}, N={self.N}, Q={self.mean_value():.3f})"
