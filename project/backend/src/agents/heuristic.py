"""Opponent-aware, A*-style heuristic agent for Xiangqi."""

from __future__ import annotations

import heapq
import logging
import math
from time import perf_counter
from typing import Protocol

from ..environment import GameEngine, GameState, Move, Side
from .base import Agent, AgentDecision, SearchBudget


_PIECE_VALUES = {
    1: 20_000,
    2: 220,
    3: 220,
    4: 400,
    5: 900,
    6: 420,
    7: 100,
}
_WIN_SCORE = 1_000_000
_LOGGER = logging.getLogger(__name__)


class Evaluator(Protocol):
    def evaluate(self, state: GameState, perspective: Side) -> float:
        """Return a position score from the supplied side's perspective."""


class _SearchLimit(Exception):
    """Raised when a search budget prevents completing a branch."""


class HeuristicAgent(Agent):
    """Choose a move using shallow adversarial best-first heuristic search.

    This is A*-style rather than standard A*: Xiangqi is adversarial, so each
    ply alternates between maximizing and minimizing the root player's score.
    """

    def __init__(
        self,
        engine: GameEngine,
        evaluator: Evaluator | None = None,
        depth: int = 1,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        if evaluator is not None and not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("evaluator must provide evaluate(state, perspective)")

        self.engine = engine
        self.evaluator = evaluator
        self.depth = depth
        self.last_score: float | None = None
        self.last_nodes = 0
        self.last_duration_seconds = 0.0
        self.last_completed_depth = 0
        self._moves_cache: dict[tuple[tuple[int, ...], Side], tuple[Move, ...]] = {}
        self._max_nodes: int | None = None
        self._deadline: float | None = None
        self._nodes = 0

    def select_action(
        self,
        state: GameState,
        budget: SearchBudget | None = None,
    ) -> AgentDecision:
        started = perf_counter()
        self._moves_cache = {}
        self._nodes = 1
        self.last_score = None
        self.last_completed_depth = 0
        self._max_nodes = budget.max_nodes if budget is not None else None
        self._deadline = None
        if budget is not None and budget.time_limit is not None:
            self._deadline = started + budget.time_limit

        try:
            if self.engine.outcome(state) is not None:
                raise ValueError("cannot select an action from a terminal state")

            moves = self._legal_moves(state)
            if not moves:
                raise ValueError("cannot select an action from a state with no legal moves")

            target_depth = self.depth
            if budget is not None and budget.max_depth is not None:
                target_depth = budget.max_depth

            perspective = state.side_to_move
            fallback: tuple[float, Move] | None = None
            completed: tuple[float, Move] | None = None

            try:
                root_queue: list[tuple[float, int, int, int, Move, GameState]] = []
                for order, move in enumerate(moves):
                    child = state.moved(move)
                    self._reserve_node()
                    score = self._f_score(child, perspective, 1)
                    candidate = (score, move)
                    if fallback is None or score > fallback[0]:
                        fallback = candidate
                    heapq.heappush(
                        root_queue,
                        (-score, move.source, move.target, order, move, child),
                    )

                while root_queue:
                    priority, _, _, _, move, child = heapq.heappop(root_queue)
                    score = self._search(
                        child,
                        perspective,
                        remaining_depth=target_depth - 1,
                        ply=1,
                        known_score=-priority,
                    )
                    if completed is None or score > completed[0]:
                        completed = (score, move)
            except _SearchLimit:
                pass

            selected = completed or fallback
            if selected is None:
                selected = (0.0, moves[0])

            self.last_score = selected[0]
            self.last_completed_depth = target_depth if completed is not None else 0
            self.last_duration_seconds = perf_counter() - started
            _LOGGER.info(
                "Heuristic move=%s f=%.2f nodes=%d elapsed=%.3fs depth=%d",
                selected[1],
                selected[0],
                self._nodes,
                self.last_duration_seconds,
                self.last_completed_depth,
            )
            return AgentDecision(move=selected[1], nodes=self._nodes)
        finally:
            self.last_nodes = self._nodes
            self.last_duration_seconds = perf_counter() - started

    def _search(
        self,
        state: GameState,
        perspective: Side,
        remaining_depth: int,
        ply: int,
        known_score: float,
    ) -> float:
        if remaining_depth <= 0:
            return known_score

        moves = self._legal_moves(state)
        if not moves:
            return known_score

        maximizing = state.side_to_move is perspective
        queue: list[tuple[float, int, int, int, Move, GameState]] = []
        for order, move in enumerate(moves):
            child = state.moved(move)
            self._reserve_node()
            score = self._f_score(child, perspective, ply + 1)
            priority = -score if maximizing else score
            heapq.heappush(
                queue,
                (priority, move.source, move.target, order, move, child),
            )

        best_score = -math.inf if maximizing else math.inf
        while queue:
            priority, _, _, _, _, child = heapq.heappop(queue)
            score = self._search(
                child,
                perspective,
                remaining_depth - 1,
                ply + 1,
                -priority if maximizing else priority,
            )
            if maximizing:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)
        return best_score

    def _f_score(self, state: GameState, perspective: Side, depth: int) -> float:
        """A*-style f = g + h; g is plies from the root, h is positional value."""
        terminal_score = self._terminal_score(state, perspective)
        if terminal_score is not None:
            return depth + terminal_score

        if self.evaluator is not None:
            heuristic = self.evaluator.evaluate(state, perspective)
            if not isinstance(heuristic, (int, float)) or not math.isfinite(heuristic):
                raise ValueError("evaluator must return a finite numeric score")
            return depth + float(heuristic)

        return depth + self._evaluate_position(state, perspective)

    def _terminal_score(self, state: GameState, perspective: Side) -> float | None:
        moves = self._legal_moves(state)
        if not moves:
            if state.side_to_move is perspective:
                return -_WIN_SCORE
            return _WIN_SCORE

        if state.history.count(state.position_key()) >= self.engine.rules.repetition_count:
            return 0.0
        if state.no_capture_count >= self.engine.rules.no_capture_limit:
            return 0.0
        return None

    def _evaluate_position(self, state: GameState, perspective: Side) -> float:
        red_score = 0.0
        board = state.board
        validator = self.engine.validator

        for square, piece in enumerate(board):
            if piece == 0:
                continue
            side_sign = 1 if piece > 0 else -1
            kind = abs(piece)
            row, col = divmod(square, 9)
            value = float(_PIECE_VALUES[kind])
            red_score += side_sign * value

            if kind == 7:
                advancement = 9 - row if piece > 0 else row
                crossed_river = row <= 4 if piece > 0 else row >= 5
                red_score += side_sign * advancement * 5
                if crossed_river:
                    red_score += side_sign * 35
            elif kind == 5:
                occupied_same_file = any(
                    board[other_row * 9 + col] != 0
                    for other_row in range(10)
                    if other_row != row
                )
                if not occupied_same_file:
                    red_score += side_sign * 25
                red_score += side_sign * len(validator.piece_targets(board, square)) * 2
            elif kind in (4, 6):
                red_score += side_sign * len(validator.piece_targets(board, square))

        for side in (Side.RED, Side.BLACK):
            sign = int(side)
            guards = sum(
                1
                for piece in board
                if piece * sign > 0 and abs(piece) in (2, 3)
            )
            red_score += sign * guards * 35
            if validator.is_in_check(state, side):
                red_score -= sign * 350

        mobility = {
            side: self._pseudo_mobility(board, side)
            for side in (Side.RED, Side.BLACK)
        }
        mobility[state.side_to_move] = len(self._legal_moves(state))
        red_score += (mobility[Side.RED] - mobility[Side.BLACK]) * 2

        threats: dict[tuple[int, Side], int] = {}
        for source, piece in enumerate(board):
            if piece == 0:
                continue
            attacker = Side.RED if piece > 0 else Side.BLACK
            for target in validator.piece_targets(board, source):
                victim = board[target]
                if victim == 0 or (victim > 0) == (piece > 0):
                    continue
                key = (target, attacker)
                threats[key] = max(threats.get(key, 0), min(_PIECE_VALUES[abs(victim)] // 5, 250))
        for (_, attacker), value in threats.items():
            red_score += int(attacker) * value

        if perspective is Side.BLACK:
            red_score = -red_score
        return red_score

    def _pseudo_mobility(self, board: tuple[int, ...], side: Side) -> int:
        return sum(
            len(self.engine.validator.piece_targets(board, source))
            for source, piece in enumerate(board)
            if piece != 0 and (piece > 0) == (side is Side.RED)
        )

    def _legal_moves(self, state: GameState) -> tuple[Move, ...]:
        key = (state.board, state.side_to_move)
        moves = self._moves_cache.get(key)
        if moves is None:
            moves = tuple(self.engine.legal_moves(state))
            self._moves_cache[key] = moves
        return moves

    def _reserve_node(self) -> None:
        if self._deadline is not None and perf_counter() >= self._deadline:
            raise _SearchLimit
        if self._max_nodes is not None and self._nodes >= self._max_nodes:
            raise _SearchLimit
        self._nodes += 1
