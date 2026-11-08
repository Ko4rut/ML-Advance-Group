"""Tabular Q agent implementing the existing backend Agent interface."""

from __future__ import annotations

import random
from pathlib import Path

from src.environment import GameEngine, GameState

from .base import Agent, AgentDecision, SearchBudget
from src.training.tabular_qlearning import TabularQTable, state_key


class TabularQAgent(Agent):
    """Pick the best Q-valued legal TRAIN-supported move; fallback to random.

    Unknown-state fallback is explicit and measurable; it is NOT learned play.
    """

    def __init__(
        self,
        table: TabularQTable,
        engine: GameEngine | None = None,
        seed: int = 42,
        epsilon: float = 0.0,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        self.table = table
        self.engine = engine or GameEngine()
        self.seed = seed
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self.last_action_supported = False

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, **kwargs) -> "TabularQAgent":
        table, _ = TabularQTable.load(checkpoint)
        return cls(table, **kwargs)

    def select_action(
        self, state: GameState, budget: SearchBudget | None = None
    ) -> AgentDecision:
        del budget
        moves = self.engine.legal_moves(state)
        if not moves:
            raise ValueError("cannot select from a state with no legal moves")
        side = "red" if state.side_to_move.value == 1 else "black"
        encoded = state_key(state.board, side)
        action_to_move = {move.source * 90 + move.target: move for move in moves}
        selected = self.table.select(encoded, action_to_move)
        self.last_action_supported = selected is not None
        if selected is None or self._rng.random() < self.epsilon:
            return AgentDecision(move=self._rng.choice(moves))
        return AgentDecision(move=action_to_move[selected])

    def reset(self) -> None:
        self._rng.seed(self.seed)
        self.last_action_supported = False
