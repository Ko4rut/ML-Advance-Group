"""Neural-guided MCTS agent: PUCT search with a pretrained policy-value network.

Separate from ``MCTSAgent`` (UCT + rollouts), which is unchanged. The network adapter
must be supplied explicitly; there is no fallback to rollouts or heuristics, so an
agent of this class is always backed by the real network or does not exist.
"""

from ..environment import GameEngine, GameState
from ._search_utils import validate_search_root
from .base import AgentDecision, SearchBudget
from .mcts_agent import MCTSAgent

DEFAULT_C_PUCT = 1.5


class NeuralMCTSAgent(MCTSAgent):
    """Play the most visited root move after a ``NeuralMCTSSearch``."""

    def __init__(self, search) -> None:
        # Duck-typed (importing search.mcts here would be circular). A classic
        # MCTSSearch must never be run under the Neural label.
        if not callable(getattr(search, "best_child", None)) \
                or not callable(getattr(getattr(search, "evaluator", None), "evaluate_with_priors", None)):
            raise TypeError("NeuralMCTSAgent requires a NeuralMCTSSearch with a policy-value evaluator")
        super().__init__(search)

    @classmethod
    def from_config(cls, engine: GameEngine, config: dict, adapter=None) -> "NeuralMCTSAgent":
        """Build from config keys ``c_puct`` and ``seed`` plus a loaded ``PolicyValueAdapter``.

        ``adapter`` is required: load it with ``networks.pretrained.load_adapter()`` (which
        fails loudly when the checkpoint or torch is missing). ``time_limit`` /
        ``max_simulations`` are not stored here; pass them as a ``SearchBudget``.
        """
        if adapter is None:
            raise ValueError("a loaded PolicyValueAdapter is required; there is no non-neural fallback")
        from ..search.mcts.evaluators import NeuralEvaluator
        from ..search.mcts.neural_search import NeuralMCTSSearch
        from ..search.mcts.selection import PUCTSelection

        search = NeuralMCTSSearch(
            engine,
            PUCTSelection(config.get("c_puct", DEFAULT_C_PUCT)),
            NeuralEvaluator(adapter),
            seed=config.get("seed"),
        )
        return cls(search)

    @property
    def model_info(self):
        """Identity of the network behind this agent (name, SHA-256, ...), if the adapter has it."""
        return getattr(self.search.evaluator.adapter, "info", None)

    def select_action(self, state: GameState, budget: SearchBudget | None = None) -> AgentDecision:
        validate_search_root(self.search.engine, state)
        root = self.search.search(state, budget)
        self.last_root = root
        best = self.search.best_child(root)
        return AgentDecision(
            move=best.move,
            nodes=self.search.last_nodes,
            simulations=self.search.last_simulations,
            policy=self.search.visit_policy(root, 1.0),
        )
