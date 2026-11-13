"""REAL pretrained-weight tests (cc-lite distill-v4-large and distill-v4). Not mock tests.

Needs torch and the provisioned, SHA-256-pinned checkpoints under models/neural/
(see models/neural/README.md). When they are absent the classes SKIP with an
explicit reason; a skip is NOT verification. In the model-validation environment
set XIANGQI_NEURAL_REQUIRE_REAL=1: a missing runtime/checkpoint then ERRORS
instead of skipping, so the real-model gate cannot pass vacuously.

Expected values in golden/real_reference.json were produced by the UPSTREAM
implementation (experiments/neural_mcts/phase3/generate_reference.py).
"""

import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from src.environment import GameState, Side
from src.networks.pretrained import (
    CheckpointIntegrityError, NeuralModelSettings, load_adapter, load_manifest,
)

from .support import (
    BACKEND, ENGINE, assert_valid_priors, black_to_move_initial, golden_nonterminal_states, golden_state,
    no_capture_draw_state, repetition_draw_state,
)

REQUIRE_REAL = os.environ.get("XIANGQI_NEURAL_REQUIRE_REAL") == "1"
HAVE_TORCH = importlib.util.find_spec("torch") is not None
REFERENCE = json.loads((BACKEND / "tests" / "networks" / "golden" / "real_reference.json").read_text())
MANIFEST = load_manifest()


def piece_removed(state: GameState, piece: int, side: Side) -> GameState:
    board = tuple(0 if p == piece else p for p in state.board)
    return GameState(board, side)


class RealCheckpointCases:
    variant: str

    @classmethod
    def setUpClass(cls):
        spec = MANIFEST.get(cls.variant)
        path = NeuralModelSettings(model=cls.variant).resolve().path
        problem = None
        if not HAVE_TORCH:
            problem = "PyTorch is not installed"
        elif not path.is_file():
            problem = f"checkpoint not provisioned at {path}"
        if problem:
            if REQUIRE_REAL:
                raise AssertionError(f"XIANGQI_NEURAL_REQUIRE_REAL=1 but {problem}")
            raise unittest.SkipTest(f"{problem}; real-weight tests for {cls.variant} NOT run (a skip is not verification)")
        cls.spec, cls.path = spec, path
        cls.adapter = load_adapter(NeuralModelSettings(model=cls.variant))

    def test_identity_of_the_loaded_weights(self):
        info = self.adapter.info
        self.assertEqual((info.name, info.sha256, info.parameter_count, info.device),
                         (self.variant, self.spec.sha256, self.spec.architecture.parameter_count, "cpu"))
        self.assertEqual(info.source_revision, "7d305c0e5adbfae4e23db40997603d8dc86b078b")

    def test_matches_the_upstream_reference_on_golden_states(self):
        entry = REFERENCE["models"][self.variant]
        self.assertEqual(entry["checkpoint_sha256"], self.spec.sha256)
        worst_prior = worst_value = 0.0
        moves_checked = 0
        for record in entry["states"]:
            state = GameState(tuple(record["board"]), Side[record["side"]])
            result = self.adapter.predict(state)
            expected = {ident: p for ident, p in record["priors"]}
            got = {m.source * 90 + m.target: p for m, p in result.priors.items()}
            self.assertEqual(set(got), set(expected), record["name"])
            worst_prior = max(worst_prior, max(abs(got[i] - expected[i]) for i in expected))
            worst_value = max(worst_value, abs(result.value - record["value"]))
            moves_checked += len(expected)
        self.assertGreaterEqual(len(entry["states"]), 50)
        self.assertLess(worst_prior, 1e-5)
        self.assertLess(worst_value, 1e-5)
        print(f"[{self.variant}] upstream reference: {len(entry['states'])} states, {moves_checked} legal moves, "
              f"max prior err {worst_prior:.2e}, max value err {worst_value:.2e}")

    def test_golden_states_have_valid_normalized_legal_priors(self):
        checked = 0
        for _, state in golden_nonterminal_states()[::5]:
            moves = ENGINE.legal_moves(state)
            if ENGINE.outcome(state, legal_moves=moves) is None:
                result = self.adapter.predict(state, moves)
                assert_valid_priors(self, result.priors, moves)
                self.assertTrue(-1.0 <= result.value <= 1.0)
                checked += 1
        self.assertGreater(checked, 80)

    def test_predictions_are_bit_identical_across_repeated_calls(self):
        for state in (GameState.initial(), black_to_move_initial(), golden_state("fixture/mate_in_one_red")):
            first = self.adapter.predict(state)
            for _ in range(3):
                again = self.adapter.predict(state)
                self.assertEqual((dict(again.priors), again.value), (dict(first.priors), first.value))

    def test_value_is_from_the_side_to_move_for_both_colours(self):
        # Material probes (Phase 1 §5): the side to move is ahead after the OTHER side loses both rooks.
        initial = GameState.initial()
        for mover in (Side.RED, Side.BLACK):
            ahead = piece_removed(initial, -5 if mover is Side.RED else 5, mover)
            behind = piece_removed(initial, 5 if mover is Side.RED else -5, mover)
            self.assertGreater(self.adapter.predict(ahead).value, 0.5, f"{mover.name} to move, up two rooks")
            self.assertLess(self.adapter.predict(behind).value, -0.5, f"{mover.name} to move, down two rooks")

    def test_mate_in_one_is_valued_as_winning_for_the_side_to_move(self):
        for name in ("fixture/mate_in_one_red", "fixture/mate_in_one_black"):
            result = self.adapter.predict(golden_state(name))
            self.assertFalse(result.is_terminal)
            self.assertGreater(result.value, 0.9, name)

    def test_engine_terminal_states_never_reach_the_network(self):
        before = self.adapter.inference_calls
        for state, value in ((golden_state("fixture/checkmated_red"), -1.0),
                             (golden_state("fixture/checkmated_black"), -1.0),
                             (golden_state("special/stalemate_black"), -1.0),
                             (repetition_draw_state(), 0.0), (no_capture_draw_state(), 0.0)):
            result = self.adapter.predict(state)
            self.assertEqual((result.is_terminal, result.value, dict(result.priors)), (True, value, {}))
        self.assertEqual(self.adapter.inference_calls, before)

    def test_inference_leaves_the_model_in_eval_mode_with_unchanged_weights(self):
        import torch
        model = self.adapter.backend.model
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for _, state in golden_nonterminal_states(40):
            self.adapter.predict(state)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], tensor), name)

    def test_a_single_flipped_byte_in_a_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / self.spec.file
            shutil.copyfile(self.path, copy)
            with copy.open("r+b") as handle:
                handle.seek(self.spec.bytes // 2)
                byte = handle.read(1)
                handle.seek(-1, 1)
                handle.write(bytes([byte[0] ^ 0x01]))
            with self.assertRaises(CheckpointIntegrityError):
                load_adapter(NeuralModelSettings(model=self.variant, path=copy))


class RealDistillV4LargeTests(RealCheckpointCases, unittest.TestCase):
    variant = "distill-v4-large"

    def test_initial_position_matches_phase1_observations(self):
        result = self.adapter.predict(GameState.initial())
        self.assertAlmostEqual(result.value, 0.126, places=2)
        top = max(result.priors, key=result.priors.get)
        self.assertEqual((top.source, top.target), (64, 67), "b2e2 central cannon, Phase 1 §4")


class RealDistillV4Tests(RealCheckpointCases, unittest.TestCase):
    variant = "distill-v4"

    def test_initial_position_matches_phase1_observations(self):
        result = self.adapter.predict(GameState.initial())
        self.assertAlmostEqual(result.value, -0.149, places=2)
        top = max(result.priors, key=result.priors.get)
        self.assertEqual((top.source, top.target), (60, 51), "g3g4 pawn push, Phase 1 §4")


if __name__ == "__main__":
    unittest.main()
