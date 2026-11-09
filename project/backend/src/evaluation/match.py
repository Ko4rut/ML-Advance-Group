"""Run a match, measure decisions and distinguish truncation."""

import time
from dataclasses import dataclass, field

from ..agents import Agent, SearchBudget
from ..environment import Move, Side, XiangqiEnv


@dataclass(slots=True)
class MoveRecord:
    """One decision: who moved, what, and how much search it cost."""

    side: Side
    move: Move
    elapsed: float
    nodes: int = 0
    simulations: int = 0


@dataclass(slots=True)
class MatchResult:
    winner: Side | None
    reason: str
    moves: list[Move]
    terminated: bool
    truncated: bool
    records: list[MoveRecord] = field(default_factory=list)

    @property
    def plies(self) -> int:
        return len(self.moves)


class MatchRunner:
    def __init__(
        self,
        env: XiangqiEnv,
        max_plies: int = 300,
        budget: SearchBudget | None = None,
    ) -> None:
        self.env = env
        self.max_plies = max_plies
        # The same budget is handed to the agent on every move.
        self.budget = budget

    def play(
        self,
        red_agent: Agent,
        black_agent: Agent,
        seed: int | None = None,
    ) -> MatchResult:
        # `seed` is kept for the interface; agents manage their own seeds.
        del seed
        red_agent.reset()
        black_agent.reset()
        state = self.env.reset()
        moves: list[Move] = []
        records: list[MoveRecord] = []
        while True:
            if len(moves) >= self.max_plies:
                return MatchResult(None, "max_plies", moves, False, True, records)
            side = state.side_to_move
            agent = red_agent if side is Side.RED else black_agent
            started = time.perf_counter()
            decision = agent.select_action(state, self.budget)
            elapsed = time.perf_counter() - started
            # An illegal move makes env.step raise; that is intentional.
            step = self.env.step(decision.move)
            moves.append(decision.move)
            records.append(
                MoveRecord(side, decision.move, elapsed, decision.nodes, decision.simulations)
            )
            if step.terminated:
                return MatchResult(
                    step.result.winner, step.result.reason, moves, True, False, records
                )
            state = step.state


def avg_stats(result: MatchResult, side: Side) -> tuple[float, float]:
    """Average (simulations, seconds) per move for one side."""
    own = [record for record in result.records if record.side is side]
    if not own:
        return 0.0, 0.0
    return (
        sum(record.simulations for record in own) / len(own),
        sum(record.elapsed for record in own) / len(own),
    )
