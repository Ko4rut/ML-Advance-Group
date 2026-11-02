"""A* search against a deterministic, one-ply opponent model."""

from __future__ import annotations

import heapq
import itertools
import math
from time import perf_counter
from typing import Protocol

from ..environment import GameEngine, GameResult, GameState, Move, Side
from .base import Agent, AgentDecision, SearchBudget


_PIECE_VALUES = {
    2: 20,
    3: 20,
    4: 40,
    5: 90,
    6: 45,
    7: 10,
}
_WIN_SCORE = 100_000.0


class Evaluator(Protocol):
    """Evaluate a position from the supplied side's perspective."""

    def evaluate(self, state: GameState, perspective: Side) -> float:
        ...


class HeuristicAgent(Agent):
    """Find a shortest predicted checkmate path using A*.

    The opponent is modeled as a deterministic greedy player that chooses its
    legal move with the best one-ply static evaluation. Each search edge is one
    agent move followed by that predicted reply. The path cost is the number of
    actual plies, and the admissible heuristic is zero. Therefore A* finds the
    shortest win against this model, not a forced win against an arbitrary
    human opponent.

    If the configured search limits expire before a win is found, the agent
    returns the best evaluated move seen so far (or a legal move if no search
    transition could be evaluated). That fallback is not an A* solution.
    """

    def __init__(
        self,
        engine: GameEngine,
        evaluator: Evaluator | None = None,
        max_depth: int = 6,
        max_nodes: int = 2_000,
        time_limit: float = 1.0,
    ) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if evaluator is not None and not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("evaluator must provide evaluate(state, perspective)")

        self.engine = engine
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.time_limit = time_limit
        self.last_nodes = 0
        self.last_duration_seconds = 0.0
        self.last_cost: int | None = None
        self.last_score: float | None = None
        self._moves_cache: dict[GameState, tuple[Move, ...]] = {}

    def select_action(
        self,
        state: GameState,
        budget: SearchBudget | None = None,
    ) -> AgentDecision:
        started = perf_counter()
        self._moves_cache = {}
        self.last_cost = None
        self.last_score = None

        moves = self._legal_moves(state)
        if not moves or self._terminal_result(state, moves) is not None:
            raise ValueError("cannot select an action from a terminal state")

        perspective = state.side_to_move
        max_depth = budget.max_depth if budget and budget.max_depth is not None else self.max_depth
        max_nodes = budget.max_nodes if budget and budget.max_nodes is not None else self.max_nodes
        deadline = (
            started + budget.time_limit
            if budget is not None and budget.time_limit is not None
            else started + self.time_limit
        )

        # h(n) = 0 is admissible here, so each OPEN priority is f(n) = g(n).
        order = itertools.count()
        open_set: list[tuple[int, int, int, GameState, Move | None]] = [
            (0, 0, next(order), state, None)
        ]
        best_cost = {state: 0}
        closed: dict[GameState, int] = {}
        nodes = 0
        fallback_move = moves[0]
        fallback_score = -math.inf
        solution: tuple[Move, int] | None = None

        while open_set:
            if deadline is not None and perf_counter() >= deadline:
                break
            if nodes >= max_nodes:
                break

            _, cost, _, current, first_move = heapq.heappop(open_set)
            if cost != best_cost.get(current):
                continue
            if cost >= closed.get(current, math.inf):
                continue
            closed[current] = cost
            nodes += 1

            current_result = self._terminal_result(current)
            if current_result is not None:
                if current_result.winner is perspective and first_move is not None:
                    solution = (first_move, cost)
                    break
                continue
            if cost >= max_depth:
                continue

            for move in self._legal_moves(current):
                if deadline is not None and perf_counter() >= deadline:
                    break

                after_agent = current.moved(move)
                opponent_moves = self._legal_moves(after_agent)
                after_agent_result = self._terminal_result(after_agent, opponent_moves)
                next_first_move = first_move or move

                if after_agent_result is not None:
                    if after_agent_result.winner is not perspective:
                        continue
                    next_state = after_agent
                    next_cost = cost + 1
                else:
                    reply = self._opponent_reply(
                        after_agent,
                        opponent_moves,
                        perspective.opponent,
                        deadline,
                    )
                    if perf_counter() >= deadline:
                        break
                    next_state = after_agent.moved(reply)
                    next_cost = cost + 2

                score = self._evaluate(next_state, perspective)
                if score > fallback_score:
                    fallback_move = next_first_move
                    fallback_score = score

                if next_cost > max_depth or next_cost >= best_cost.get(next_state, math.inf):
                    continue
                best_cost[next_state] = next_cost
                heuristic_cost = 0
                priority = next_cost + heuristic_cost
                heapq.heappush(
                    open_set,
                    (priority, next_cost, next(order), next_state, next_first_move),
                )

        self.last_nodes = nodes
        self.last_duration_seconds = perf_counter() - started
        if solution is not None:
            selected_move, self.last_cost = solution
            self.last_score = _WIN_SCORE
        else:
            selected_move = fallback_move
            self.last_score = fallback_score if fallback_score > -math.inf else None

        return AgentDecision(move=selected_move, nodes=nodes)

    def _opponent_reply(
        self,
        state: GameState,
        moves: tuple[Move, ...],
        perspective: Side,
        deadline: float,
    ) -> Move:
        best_move = moves[0]
        best_score = -math.inf
        for move in moves:
            if perf_counter() >= deadline:
                break
            score = self._evaluate(state.moved(move), perspective)
            if score > best_score:
                best_move = move
                best_score = score
        return best_move

    def _evaluate(self, state: GameState, perspective: Side) -> float:
        if self.evaluator is not None:
            score = self.evaluator.evaluate(state, perspective)
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("evaluator must return a finite numeric score")
            return float(score)

        red_score = sum(
            (1 if piece > 0 else -1) * _PIECE_VALUES.get(abs(piece), 0)
            for piece in state.board
            if piece
        )
        for side in (Side.RED, Side.BLACK):
            if self.engine.validator.is_in_check(state, side):
                red_score -= int(side) * 500
        return red_score if perspective is Side.RED else -red_score

    def _terminal_result(
        self,
        state: GameState,
        moves: tuple[Move, ...] | None = None,
    ) -> GameResult | None:
        if moves is None:
            moves = self._legal_moves(state)
        if not moves:
            winner = state.side_to_move.opponent
            reason = (
                "checkmate"
                if self.engine.validator.is_in_check(state, state.side_to_move)
                else "stalemate"
            )
            return GameResult(winner, reason)
        if state.history.count(state.position_key()) >= self.engine.rules.repetition_count:
            return GameResult(None, "repetition")
        if state.no_capture_count >= self.engine.rules.no_capture_limit:
            return GameResult(None, "no_capture_limit")
        return None

    def _legal_moves(self, state: GameState) -> tuple[Move, ...]:
        moves = self._moves_cache.get(state)
        if moves is None:
            moves = tuple(self.engine.legal_moves(state))
            self._moves_cache[state] = moves
        return moves
