"""Rollout and neural leaf evaluation.

Every evaluator returns a value in [-1, 1] from the perspective of
``state.side_to_move`` (+1 means the player about to move is winning).
"""

import math
import random

from ...agents.position_evaluator import PIECE_VALUES, evaluate_position
from ...environment import GameEngine, GameState, Move

ROLLOUT_POLICIES = {"random", "capture_first"}


class RolloutEvaluator:
    """Play a short (or full) playout, then score the final position."""

    def __init__(
        self,
        engine: GameEngine,
        policy: str = "random",
        max_depth: int | None = 0,
        eval_scale: float = 100,
        eval_fn=evaluate_position,
        epsilon: float = 0.1,
        max_rollout_plies: int = 300,
        seed: int | None = None,
    ) -> None:
        if policy not in ROLLOUT_POLICIES:
            raise ValueError(f"policy must be one of {sorted(ROLLOUT_POLICIES)}")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if eval_scale <= 0:
            raise ValueError("eval_scale must be positive")
        self.engine = engine
        self.policy = policy
        # 0 = static evaluation only; None = play until the game ends.
        self.max_depth = max_depth
        self.eval_scale = eval_scale
        self.eval_fn = eval_fn
        self.epsilon = epsilon
        self.max_rollout_plies = max_rollout_plies
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.rng.seed(self.seed)

    def evaluate(self, state: GameState) -> float:
        root_side = state.side_to_move
        steps = self.max_depth if self.max_depth is not None else self.max_rollout_plies
        current = state
        for _ in range(steps):
            moves = self.engine.legal_moves(current)
            result = self.engine.outcome(current, legal_moves=moves)
            if result is not None:
                return float(result.reward(root_side))
            current = current.moved(self._choose_move(current, moves))

        # The last position reached may itself be terminal.
        result = self.engine.outcome(current)
        if result is not None:
            return float(result.reward(root_side))
        if self.max_depth is None:
            # Hit the ply cap without a result: treat it as a draw.
            return 0.0
        # eval_fn already receives ``root_side``, so the value is from the
        # perspective of the ORIGINAL side to move, whatever the rollout depth.
        return math.tanh(self.eval_fn(current, root_side) / self.eval_scale)

    def _choose_move(self, state: GameState, moves: list[Move]) -> Move:
        if self.policy == "random" or self.rng.random() < self.epsilon:
            return self.rng.choice(moves)

        # capture_first: take the most valuable victim, otherwise play randomly.
        best_value = 0
        best_moves: list[Move] = []
        for move in moves:
            victim = state.board[move.target]
            if not victim:
                continue
            value = PIECE_VALUES[abs(victim)]
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
        if best_moves:
            return self.rng.choice(best_moves)
        return self.rng.choice(moves)


class NeuralEvaluator:
    def __init__(self, network, observation_encoder, action_encoder) -> None:
        self.network = network
        self.observation_encoder = observation_encoder
        self.action_encoder = action_encoder

    def evaluate(self, state: GameState):
        raise NotImplementedError("NeuralEvaluator is not implemented yet")
