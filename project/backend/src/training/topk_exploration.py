"""Controlled stochastic root selection from *comparable* Alpha-Beta scores.

Sampling changes which good action is tried, never the piece values or the
heuristic. All candidates must already be legal and have been searched to the
same depth. Do not use root alpha/beta bounds as if they were exact scores.
"""

from __future__ import annotations

import math
import random

from ..environment import Move


def sample_top_k(
    ranked: list[tuple[Move, int]],
    rng: random.Random,
    *,
    top_k: int,
    score_margin: int,
    temperature: float,
) -> tuple[Move, int, list[tuple[Move, int, float]]]:
    """Return sampled action, its exact score, and normalized probabilities."""
    if top_k < 1 or score_margin < 0 or temperature <= 0:
        raise ValueError("top_k and temperature must be positive; margin >= 0")
    if not ranked:
        raise ValueError("ranked root actions must not be empty")
    best_score = max(score for _, score in ranked)
    eligible = [
        (move, score)
        for move, score in ranked
        if score >= best_score - score_margin
    ]
    eligible.sort(key=lambda pair: pair[1], reverse=True)
    eligible = eligible[:top_k]
    weights = [
        math.exp((score - best_score) / temperature)
        for _, score in eligible
    ]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    threshold = rng.random()
    running = 0.0
    selection = eligible[-1]
    for pair, probability in zip(eligible, probabilities):
        running += probability
        if threshold < running:
            selection = pair
            break
    distribution = [
        (move, score, probability)
        for (move, score), probability in zip(eligible, probabilities)
    ]
    return selection[0], selection[1], distribution
