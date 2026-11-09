"""Configure the existing MCTS agent and enforce its per-move search budget."""

from pathlib import Path
from typing import Any

import yaml

from ...agents import MCTSAgent, SearchBudget
from ...environment import GameEngine, GameState, Move


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "mcts.yaml"


class MCTSService:
    """Keep API configuration separate from the MCTS search algorithm."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self._config_path = config_path
        self._config = dict(config) if config is not None else None

    def _settings(self) -> dict[str, Any]:
        if self._config is None:
            with self._config_path.open(encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("MCTS config must be a YAML mapping")
            self._config = loaded
        return self._config

    def make_agent(self, engine: GameEngine) -> MCTSAgent:
        """Create a fresh search tree and RNG for each match participant."""
        return MCTSAgent.from_config(engine, self._settings())

    def select_move(self, agent: MCTSAgent, state: GameState) -> Move:
        """Bound MCTS search without changing other agents' budgets."""
        config = self._settings()
        budget = SearchBudget(
            time_limit=config.get("time_limit", 1.0),
            max_simulations=config.get("max_simulations"),
        )
        return agent.select_action(state, budget).move
