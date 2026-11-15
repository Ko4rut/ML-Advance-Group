"""Fakes and fixtures for neural-MCTS tests. Pure stdlib: no torch, no checkpoint."""

from array import array
from collections.abc import Callable
import json
import math
from pathlib import Path
import time
from types import MappingProxyType

from src.encoding import ACTION_SPACE_SIZE
from src.environment import GameEngine, GameState, Move, Side
from src.networks.pretrained import ModelInfo, PolicyValueAdapter, PolicyValueResult

GOLDEN = Path(__file__).resolve().parents[1] / "encoding" / "golden" / "cclite_v1.json"
ENGINE = GameEngine()
PIECE_VALUE = {1: 0.0, 2: 2.0, 3: 2.0, 4: 4.0, 5: 9.0, 6: 4.5, 7: 1.0}


def position(pieces, side=Side.RED) -> GameState:
    board = [0] * 90
    for (row, col), piece in pieces.items():
        board[row * 9 + col] = piece
    return GameState(tuple(board), side)


def color_swap(state: GameState) -> GameState:
    """Rotate 180 degrees and exchange colours (and the side to move)."""
    return GameState(tuple(-p for p in reversed(state.board)), state.side_to_move.opponent)


MATE_IN_ONE_RED = position({(0, 4): -1, (9, 4): 1, (1, 0): 5, (2, 3): 5, (2, 4): 7, (2, 5): 5})
MATE_IN_ONE_BLACK = color_swap(MATE_IN_ONE_RED)


def material_value(state: GameState) -> float:
    """Toy value from the side to move's view: tanh of the material balance."""
    balance = sum(PIECE_VALUE[abs(p)] * (1 if p > 0 else -1) for p in state.board if p)
    return math.tanh(balance * int(state.side_to_move) / 9.0)


def normalized(weights: dict[Move, float]) -> dict[Move, float]:
    total = math.fsum(weights.values())
    return {move: w / total for move, w in weights.items()}


class FakeEvaluator:
    """Policy-value evaluator with scripted priors/values. Records every call."""

    def __init__(self, weight: Callable[[GameState, Move], float] = lambda s, m: 1.0,
                 value: Callable[[GameState], float] = lambda s: 0.0, delay: float = 0.0):
        self.weight, self.value, self.delay = weight, value, delay
        self.calls: list[tuple[GameState, tuple[Move, ...]]] = []

    def expected_priors(self, state: GameState, moves) -> dict[Move, float]:
        return normalized({m: self.weight(state, m) for m in moves})

    def evaluate_with_priors(self, state, legal_moves):
        # The search must never hand a finished game to the network.
        assert ENGINE.outcome(state) is None, "terminal state sent to the evaluator"
        assert set(legal_moves) == set(ENGINE.legal_moves(state)), "evaluator got a wrong legal-move list"
        self.calls.append((state, tuple(legal_moves)))
        if self.delay:
            time.sleep(self.delay)
        return PolicyValueResult(MappingProxyType(self.expected_priors(state, legal_moves)), self.value(state))


class StubResult:
    """Arbitrary (possibly invalid) evaluator output for negative tests."""

    def __init__(self, priors, value=0.0, is_terminal=False):
        self.priors, self.value, self.is_terminal = priors, value, is_terminal


class StubEvaluator:
    def __init__(self, make_result):
        self.make_result = make_result

    def evaluate_with_priors(self, state, legal_moves):
        return self.make_result(state, list(legal_moves))


class FakeBackend:
    """Deterministic stand-in network behind a real PolicyValueAdapter."""

    def __init__(self, value=0.1):
        self.value = value
        self.calls = 0

    def infer(self, observation: array):
        self.calls += 1
        return [0.0] * ACTION_SPACE_SIZE, self.value


def fake_adapter(value=0.1) -> PolicyValueAdapter:
    info = ModelInfo(name="fake", sha256="0" * 64, source_repository="test", source_revision="none",
                     observation="absolute_planes16_v1", action_vocabulary="cclite_from_to_8100",
                     device="cpu", parameter_count=0)
    return PolicyValueAdapter(FakeBackend(value), info)


def golden_states(step: int = 1, limit: int | None = None):
    records = json.loads(GOLDEN.read_text())["states"]
    states = [(r["name"], GameState(tuple(r["board"]), Side[r["side"]])) for r in records
              if r["upstream_legal_action_ids"]][::step]
    return states[:limit] if limit else states


def all_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())
