"""Shared fixtures for the neural adapter tests (no torch import)."""

from array import array
from dataclasses import replace
import json
import math
from pathlib import Path

from src.encoding import ACTION_SPACE_SIZE, CCLITE_VOCABULARY
from src.environment import GameEngine, GameState, Move, Side
from src.networks.pretrained import ModelInfo

BACKEND = Path(__file__).resolve().parents[2]
GOLDEN_STATES = BACKEND / "tests" / "encoding" / "golden" / "cclite_v1.json"
ENGINE = GameEngine()


def fake_info(**overrides) -> ModelInfo:
    values = dict(name="fake", sha256="0" * 64, source_repository="test://fake", source_revision="none",
                  observation="absolute_planes16_v1", action_vocabulary=CCLITE_VOCABULARY.name,
                  device="cpu", parameter_count=0)
    values.update(overrides)
    return ModelInfo(**values)


class FakeBackend:
    """Deterministic stand-in network. Records every observation it receives."""

    def __init__(self, logits=None, value=0.25):
        self.logits = logits if logits is not None else [0.0] * ACTION_SPACE_SIZE
        self.value = value
        self.observations: list[array] = []

    def infer(self, observation):
        self.observations.append(array("f", observation))
        return self.logits, self.value

    @property
    def calls(self) -> int:
        return len(self.observations)


class ForbiddenBackend:
    def infer(self, observation):
        raise AssertionError("the network must not be called for this state")


def logits_favoring(move: Move, boost: float = 6.0):
    logits = [0.0] * ACTION_SPACE_SIZE
    logits[CCLITE_VOCABULARY.to_native(move)] = boost
    return logits


def black_to_move_initial() -> GameState:
    initial = GameState.initial()
    return GameState(initial.board, Side.BLACK)


def golden_state(name: str) -> GameState:
    for record in json.loads(GOLDEN_STATES.read_text())["states"]:
        if record["name"] == name:
            return GameState(tuple(record["board"]), Side[record["side"]])
    raise KeyError(name)


def golden_nonterminal_states(limit: int | None = None):
    states = []
    for record in json.loads(GOLDEN_STATES.read_text())["states"]:
        if record["upstream_legal_action_ids"]:
            states.append((record["name"], GameState(tuple(record["board"]), Side[record["side"]])))
    return states[:limit] if limit else states


def repetition_draw_state() -> GameState:
    initial = GameState.initial()
    return replace(initial, history=(initial.position_key(),) * 3)


def no_capture_draw_state() -> GameState:
    return replace(GameState.initial(), no_capture_count=120)


def assert_valid_priors(testcase, priors, moves):
    testcase.assertEqual(set(priors), set(moves))
    testcase.assertEqual(len(priors), len(moves))
    for prior in priors.values():
        testcase.assertTrue(math.isfinite(prior) and prior >= 0.0)
    testcase.assertAlmostEqual(math.fsum(priors.values()), 1.0, places=12)
