'''Stateful environment wrapper shared by all agents.'''

from dataclasses import dataclass
from typing import Optional

from .engine import GameEngine
from .rules import RulesConfig
from .state import GameState
from .types import GameResult, Move, Side


@dataclass(frozen=True, slots=True)
class StepResult:
    state: GameState
    reward: int
    terminated: bool
    result: Optional[GameResult]


class XiangqiEnv:
    '''Small environment API; intentionally not coupled to Gymnasium.'''

    def __init__(self, rules: RulesConfig | None = None) -> None:
        self.engine = GameEngine(rules)
        self.state = GameState.initial()

    def reset(self, state: GameState | None = None) -> GameState:
        self.state = (state or GameState.initial()).clone()
        return self.state

    def step(self, action: Move) -> StepResult:
        if not isinstance(action, Move):
            raise TypeError("action must be a Move")
        current = self.state
        if self.engine.outcome(current) is not None:
            raise ValueError("cannot step a terminal environment")
        next_state = self.engine.apply_move(current, action)
        result = self.engine.outcome(next_state)
        moving_side = current.side_to_move
        reward = result.reward(moving_side) if result is not None else 0
        self.state = next_state
        return StepResult(next_state, reward, result is not None, result)

    def get_legal_actions(self) -> list[Move]:
        if self.engine.outcome(self.state) is not None:
            return []
        return self.engine.legal_moves(self.state)

    def is_terminal(self) -> bool:
        return self.engine.outcome(self.state) is not None

    def get_winner(self) -> Optional[Side]:
        result = self.engine.outcome(self.state)
        return None if result is None else result.winner

    def clone_state(self) -> GameState:
        return self.state.clone()

    def render(self) -> str:
        symbols = {
            1: "K", 2: "A", 3: "E", 4: "H", 5: "R", 6: "C", 7: "P",
            -1: "k", -2: "a", -3: "e", -4: "h", -5: "r", -6: "c", -7: "p",
        }
        return "\n".join(
            " ".join(symbols[p] if p else "." for p in self.state.board[row * 9:(row + 1) * 9])
            for row in range(10)
        )
