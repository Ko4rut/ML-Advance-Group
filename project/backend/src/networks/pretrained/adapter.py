"""Framework-independent policy/value inference adapter.

``GameState -> ObservationEncoder -> backend -> legal priors + value``.

This module is pure standard library. The neural framework hides behind the
tiny :class:`InferenceBackend` protocol, so the adapter's invariants are tested
with fake backends and ordinary backend installs never import torch.

Contract (see docs/neural_mcts/02_CONTRACT.md and 03_MODEL_CARD.md):

* ``priors``: ``{legal Move: p}`` over exactly the supplied legal moves, each
  finite and >= 0, summing to 1 (Phase 2 ``legal_priors``).
* ``value``: finite, in [-1, 1], from the perspective of ``state.side_to_move``
  (+1 = the player about to move is winning). No colour-dependent sign flip.
* Terminal precedence: if the engine says the state is over (no legal move,
  repetition, no-capture limit) the result carries the engine's outcome, the
  network is NOT called, ``priors`` is empty and ``value`` is
  ``GameResult.reward(side_to_move)`` in {-1, 0, +1}.
* Any invalid network output raises :class:`InvalidModelOutputError`; there is
  no uniform-prior or heuristic fallback.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from threading import Lock
from types import MappingProxyType
from typing import Protocol

from ...encoding import (
    CCLITE_VOCABULARY,
    OBSERVATION_LENGTH,
    CCLiteVocabulary,
    ObservationEncoder,
    legal_priors,
)
from ...environment import GameEngine, GameResult, GameState, Move
from .errors import InvalidModelOutputError


class InferenceBackend(Protocol):
    """One forward pass on one observation; returns (native policy logits, value)."""

    def infer(self, observation: array) -> tuple[Sequence[float], float]: ...


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Identity of the model behind an adapter (for logs, API status, benchmarks)."""

    name: str
    sha256: str
    source_repository: str
    source_revision: str
    observation: str
    action_vocabulary: str
    device: str
    parameter_count: int


@dataclass(frozen=True, slots=True)
class PolicyValueResult:
    """Prediction for one state; see the module docstring for the contract."""

    priors: Mapping[Move, float]
    value: float
    terminal: GameResult | None = None

    @property
    def is_terminal(self) -> bool:
        return self.terminal is not None


class PolicyValueAdapter:
    """Predict legal-move priors and a side-to-move value for engine states."""

    def __init__(
        self,
        backend: InferenceBackend,
        info: ModelInfo,
        *,
        engine: GameEngine | None = None,
        vocabulary: CCLiteVocabulary = CCLITE_VOCABULARY,
    ) -> None:
        if info.observation != ObservationEncoder.encoding_name or info.action_vocabulary != vocabulary.name:
            raise InvalidModelOutputError(
                f"model contract ({info.observation}, {info.action_vocabulary}) does not match the "
                f"production encoders ({ObservationEncoder.encoding_name}, {vocabulary.name})")
        self.backend = backend
        self.info = info
        self.engine = engine or GameEngine()
        self.vocabulary = vocabulary
        self._encoder = ObservationEncoder(self.engine.validator)
        self._lock = Lock()  # one forward at a time; also guards the counters
        self._inference_calls = 0
        self._terminal_shortcuts = 0

    @property
    def inference_calls(self) -> int:
        """Number of network forward passes made (terminal states do not count)."""
        return self._inference_calls

    @property
    def terminal_shortcuts(self) -> int:
        return self._terminal_shortcuts

    def predict(
        self,
        state: GameState,
        legal_moves: Iterable[Move] | None = None,
        *,
        verify_legal_moves: bool = False,
    ) -> PolicyValueResult:
        """Return priors over ``legal_moves`` and the side-to-move value.

        ``legal_moves`` defaults to the engine's list. A caller that already has
        it (MCTS) should pass it to avoid regenerating moves; it is trusted to be
        the engine's legal list unless ``verify_legal_moves`` is set (costs one
        move generation). Duplicates and non-``Move`` entries always raise
        ``ValueError``/``TypeError``.
        """
        if not isinstance(state, GameState):
            raise TypeError("state must be a GameState")
        if legal_moves is None:
            moves = self.engine.legal_moves(state)
        else:
            moves = list(legal_moves)
            if any(not isinstance(move, Move) for move in moves):
                raise TypeError("legal_moves must contain only Move objects")
            if len(set(moves)) != len(moves):
                raise ValueError("duplicate moves in legal_moves")
            if verify_legal_moves and set(moves) != set(self.engine.legal_moves(state)):
                raise ValueError("legal_moves differ from the engine's legal moves for this state")

        result = self.engine.outcome(state, legal_moves=moves)
        if result is not None:  # engine is authoritative; the network never sees terminal states
            with self._lock:
                self._terminal_shortcuts += 1
            return PolicyValueResult(MappingProxyType({}), float(result.reward(state.side_to_move)), result)

        observation = self._encoder.encode(state)
        if len(observation) != OBSERVATION_LENGTH:  # defensive: encoder contract
            raise InvalidModelOutputError("observation has unexpected length")
        with self._lock:
            logits, raw_value = self.backend.infer(observation)
            self._inference_calls += 1
        value = _checked_value(raw_value)
        try:
            priors = legal_priors(logits, moves, kind="logits", vocabulary=self.vocabulary)
        except ValueError as exc:  # inputs were validated above, so this is the model's output
            raise InvalidModelOutputError(f"invalid policy output: {exc}") from exc
        return PolicyValueResult(MappingProxyType(priors), value)


def _checked_value(raw: object) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise InvalidModelOutputError(f"value must be a real number, got {type(raw).__name__}")
    value = float(raw)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise InvalidModelOutputError(f"value {value!r} is not finite and within [-1, 1]")
    return value
