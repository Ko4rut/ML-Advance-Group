"""Common agent interface and decision metadata."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from ..environment import GameState, Move


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Optional limits understood by search-based agents."""

    max_depth: int | None = None
    max_nodes: int | None = None
    max_simulations: int | None = None
    time_limit: float | None = None

    def __post_init__(self) -> None:
        values = (self.max_depth, self.max_nodes, self.max_simulations, self.time_limit)
        if any(value is not None and value <= 0 for value in values):
            raise ValueError("search budget values must be positive")


@dataclass(frozen=True, slots=True)
class AgentDecision:
    """A selected move plus optional search diagnostics."""

    move: Move
    nodes: int = 0
    simulations: int = 0
    policy: Mapping[Move, float] | None = None


class Agent(ABC):
    """Interface implemented by every Xiangqi agent."""

    @abstractmethod
    def select_action(
        self,
        state: GameState,
        budget: SearchBudget | None = None,
    ) -> AgentDecision:
        """Select one legal action without mutating the supplied state."""

    def reset(self) -> None:
        """Reset state kept between games, if any."""
