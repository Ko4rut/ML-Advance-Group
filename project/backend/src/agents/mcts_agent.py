"""Monte Carlo Tree Search agent for Xiangqi."""

from typing import TYPE_CHECKING

from ..environment import GameEngine, GameState
from .base import Agent, AgentDecision, SearchBudget
from ._search_utils import validate_search_root

# search.mcts imports agents.position_evaluator, so importing it here at module
# level would be circular. It is only needed for type hints and in from_config.
if TYPE_CHECKING:
    from ..search.mcts.node import MCTSNode
    from ..search.mcts.search import MCTSSearch


class MCTSAgent(Agent):
    """Pick the most visited root move after a UCT search."""

    def __init__(self, search: "MCTSSearch") -> None:
        self.search = search
        self.last_root: "MCTSNode | None" = None

    @classmethod
    def from_config(cls, engine: GameEngine, config: dict) -> "MCTSAgent":
        """Build an agent from the keys of configs/mcts.yaml.

        ``time_limit`` is not stored here; callers pass it as a SearchBudget.
        """
        from ..search.mcts.evaluators import RolloutEvaluator
        from ..search.mcts.search import MCTSSearch
        from ..search.mcts.selection import UCTSelection

        seed = config.get("seed")
        evaluator = RolloutEvaluator(
            engine,
            policy=config.get("rollout_policy", "random"),
            max_depth=config.get("rollout_depth", 0),
            eval_scale=config.get("eval_scale", 100),
            seed=seed,
        )
        search = MCTSSearch(
            engine,
            UCTSelection(config.get("c", 1.4)),
            evaluator,
            seed=seed,
        )
        return cls(search)

    def select_action(
        self,
        state: GameState,
        budget: SearchBudget | None = None,
    ) -> AgentDecision:
        validate_search_root(self.search.engine, state)
        root = self.search.search(state, budget)
        self.last_root = root
        best = max(root.children.values(), key=lambda child: (child.N, child.mean_value()))
        return AgentDecision(
            move=best.move,
            nodes=self.search.last_nodes,
            simulations=self.search.last_simulations,
            policy=self.search.visit_policy(root, 1.0),
        )

    def debug_root(self, top_k: int = 5) -> list[tuple]:
        """Print and return the top root children as (move, N, Q, prior) rows."""
        if self.last_root is None:
            print("No search has been run yet.")
            return []
        children = sorted(
            self.last_root.children.values(),
            key=lambda child: child.N,
            reverse=True,
        )[:top_k]
        rows = [
            (child.move, child.N, child.mean_value(), child.prior)
            for child in children
        ]
        print(f"{'move':<28}{'N':>8}{'Q':>9}{'prior':>9}")
        for move, visits, value, prior in rows:
            source_row, source_col = divmod(move.source, 9)
            target_row, target_col = divmod(move.target, 9)
            label = (
                f"{move.source}->{move.target} "
                f"({source_row},{source_col})->({target_row},{target_col})"
            )
            print(f"{label:<28}{visits:>8}{value:>9.3f}{prior:>9.3f}")
        return rows

    def reset(self) -> None:
        self.search.reset()
        self.last_root = None
