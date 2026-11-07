"""Reward schemes for alternating-player zero-sum Xiangqi offline RL.

The reward belongs to the player who took the recorded action. The next state's
Q-value belongs to the *opponent*, so DQN/negamax must bootstrap with a minus:
    target = reward - gamma * max_legal Q(next_state, action)

The potential is evaluated for the SAME action-taking side on both boards:
    r_shaped = r_terminal + weight * (gamma * Phi(next, actor) - Phi(now, actor))
Phi(next) is set to zero on true or recorded terminal transitions. With a
matching discount factor, the potential shaping term telescopes; unlike fixed
capture bonuses it does not change the relative returns of complete games
starting at the same state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from src.data.silver.models import SilverGame, SilverTransition

PIECE_VALUES = {
    1: 0,   # Kings are game-ending conditions, NOT material to optimize.
    2: 20,  # Advisor
    3: 20,  # Elephant
    4: 40,  # Horse
    5: 90,  # Rook
    6: 45,  # Cannon
    7: 10,  # Pawn
}

SPARSE_DEFINITION = "terminal_outcome_sparse_actor_perspective_v1"
SHAPED_DEFINITION = "terminal_outcome_potential_material_actor_perspective_v2"


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Versioned reward selection; the old sparse baseline remains reproducible."""

    mode: Literal["sparse", "shaped"] = "shaped"
    gamma: float = 0.99
    shaping_weight: float = 0.15
    material_scale: float = 300.0
    pawn_advancement_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("sparse", "shaped"):
            raise ValueError("reward mode must be 'sparse' or 'shaped'")
        if not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be finite and in (0, 1]")
        if not math.isfinite(self.shaping_weight) or not 0 <= self.shaping_weight <= 1:
            raise ValueError("shaping_weight must be finite and in [0, 1]")
        if not math.isfinite(self.material_scale) or self.material_scale <= 0:
            raise ValueError("material_scale must be finite and positive")
        if not math.isfinite(self.pawn_advancement_weight) or self.pawn_advancement_weight < 0:
            raise ValueError("pawn_advancement_weight must be finite and non-negative")

    @property
    def definition(self) -> str:
        if self.mode == "sparse":
            return SPARSE_DEFINITION
        return SHAPED_DEFINITION

    def as_dict(self) -> dict[str, float | str]:
        return {
            "mode": self.mode,
            "gamma": self.gamma,
            "shaping_weight": self.shaping_weight,
            "material_scale": self.material_scale,
            "pawn_advancement_weight": self.pawn_advancement_weight,
        }


def material_potential(
    board: list[int], side: str, *, material_scale: float = 300.0,
    pawn_advancement_weight: float = 1.0,
) -> float:
    """Bounded zero-sum state potential, red/black symmetric.

    Material is the dominant signal. Advanced pawns provide a small positional
    signal. This uses only the board and selected side, not source or outcome,
    to avoid leaking terminal labels into the state evaluator.
    """
    if len(board) != 90:
        raise ValueError("board must contain 90 squares")
    if side not in ("red", "black"):
        raise ValueError("side must be red or black")
    total_red = 0.0
    for square, piece in enumerate(board):
        if not piece:
            continue
        if abs(piece) not in PIECE_VALUES:
            raise ValueError(f"unexpected piece encoding at square {square}")
        score = float(PIECE_VALUES[abs(piece)])
        if abs(piece) == 7:
            row = square // 9
            # Symmetric advancement (increasing as a pawn approaches enemy).
            progress = 9 - row if piece > 0 else row
            score += pawn_advancement_weight * progress
        total_red += score if piece > 0 else -score
    actor_score = total_red if side == "red" else -total_red
    return math.tanh(actor_score / material_scale)


def terminal_reward(game: SilverGame, step: SilverTransition, terminated: bool) -> float:
    if not terminated or game.outcome.winner is None:
        return 0.0
    if step.side == game.outcome.winner:
        return 1.0
    return -1.0


def calculate_reward(
    game: SilverGame,
    step: SilverTransition,
    terminated: bool,
    config: RewardConfig,
) -> float:
    """Reward from step.side's perspective; never infer a draw from truncation."""
    base = terminal_reward(game, step, terminated)
    if config.mode == "sparse" or config.shaping_weight == 0:
        return base

    current = material_potential(
        step.board,
        step.side,
        material_scale=config.material_scale,
        pawn_advancement_weight=config.pawn_advancement_weight,
    )
    # Recorded PGN results and engine terminals both close this dataset episode.
    # A truncated episode is NOT terminal; keep its next-state potential.
    successor = 0.0
    if not terminated:
        successor = material_potential(
            step.next_board,
            step.side,  # IMPORTANT: actor, not opponent next_side
            material_scale=config.material_scale,
            pawn_advancement_weight=config.pawn_advancement_weight,
        )
    shaping = config.shaping_weight * (config.gamma * successor - current)
    return base + shaping
