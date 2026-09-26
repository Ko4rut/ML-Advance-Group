"""Uniform random baseline agent."""

import random

from ..environment import GameState, legal_moves
from .base import Agent, AgentDecision, SearchBudget


class RandomAgent(Agent):
    """Choose uniformly from the legal moves using a private RNG."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def select_action(
        self,
        state: GameState,
        budget: SearchBudget | None = None,
    ) -> AgentDecision:
        del budget  # Random choice does not consume a search budget.
        moves = legal_moves(state)
        if not moves:
            raise ValueError("cannot select an action from a state with no legal moves")
        move = self._rng.choice(moves)
        probability = 1.0 / len(moves)
        return AgentDecision(
            move=move,
            policy={candidate: probability for candidate in moves},
        )

    def reset(self) -> None:
        self._rng.seed(self.seed)
