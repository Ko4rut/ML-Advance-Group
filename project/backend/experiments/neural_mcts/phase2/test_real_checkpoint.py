"""Production encoding stack vs the upstream reference, on the REAL pretrained checkpoint.

Runs the production ObservationEncoder + ActionEncoder/CCLITE_VOCABULARY + legal_priors
(src.encoding, pure Python) through the real network and compares against the
upstream `TorchEvaluator` on every non-terminal state of the golden fixture.
Explicitly skipped (never "passed") unless the Phase 1 runtime/source/checkpoint
environment variables are set. See phase1/README.md for setup. Optional:
XIANGQI_PHASE1_VARIANT (default distill-v4; or distill-v4-large).
"""

import json
import math
import os
from pathlib import Path
import unittest

from src.encoding import CCLITE_VOCABULARY, ObservationEncoder, batch_to_numpy, legal_priors, to_numpy
from src.environment import GameEngine, GameState, Side

GOLDEN = Path(__file__).resolve().parents[3] / "tests" / "encoding" / "golden" / "cclite_v1.json"


@unittest.skipUnless(os.environ.get("XIANGQI_PHASE1_SOURCE") and os.environ.get("XIANGQI_PHASE1_CHECKPOINT"),
                     "Real pretrained checkpoint and pinned source paths not provided")
class RealCheckpointProductionStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qualify import CheckpointProbe
        cls.probe = CheckpointProbe(Path(os.environ["XIANGQI_PHASE1_SOURCE"]),
                                    Path(os.environ["XIANGQI_PHASE1_CHECKPOINT"]),
                                    os.environ.get("XIANGQI_PHASE1_VARIANT", "distill-v4"))
        cls.engine, cls.encoder = GameEngine(), ObservationEncoder()
        cls.states = [(r["name"], GameState(tuple(r["board"]), Side[r["side"]]))
                      for r in json.loads(GOLDEN.read_text())["states"]]

    def test_production_stack_matches_upstream_evaluator_on_all_nonterminal_fixture_states(self):
        from contracts import fen
        import numpy as np
        checked = moves_checked = 0
        for name, state in self.states:
            moves = self.engine.legal_moves(state)
            if not moves:
                continue
            planes = to_numpy(self.encoder.encode(state))
            self.assertEqual(planes.shape, (16, 10, 9))
            self.assertEqual(planes.dtype, np.float32)
            native = self.probe.Board.from_fen(fen(state))
            np.testing.assert_array_equal(planes, self.probe.native_encoding.encode_board(native), err_msg=name)
            logits, value = self.probe.forward(planes)
            priors = legal_priors(logits, moves)
            expected, expected_value = self.probe.reference.evaluate(native)
            reference = {self.probe.native_encoding.MoveCodec.encode(m): p for m, p in expected.items()}
            self.assertEqual({CCLITE_VOCABULARY.to_native(m) for m in priors}, set(reference), name)
            error = max(abs(p - reference[CCLITE_VOCABULARY.to_native(m)]) for m, p in priors.items())
            self.assertLess(error, 1e-6, name)
            self.assertLess(abs(value - expected_value), 1e-6, name)
            self.assertAlmostEqual(math.fsum(priors.values()), 1.0, places=12)
            self.assertTrue(-1.0 <= value <= 1.0)
            checked += 1
            moves_checked += len(moves)
        self.assertGreaterEqual(checked, 400)
        print(f"[{self.probe.variant}] parity on {checked} states / {moves_checked} legal moves")

    def test_batched_production_encoding_equals_single_state_encoding_through_the_model(self):
        import numpy as np
        torch = self.probe.torch
        sample = [s for _, s in self.states if self.engine.legal_moves(s)][:16]
        batch = batch_to_numpy(self.encoder.encode_batch(sample))
        with torch.inference_mode():
            batched_logits, batched_values = self.probe.model(torch.from_numpy(batch))
        for i, state in enumerate(sample):
            logits, value = self.probe.forward(to_numpy(self.encoder.encode(state)))
            np.testing.assert_allclose(batched_logits[i].numpy(), logits, atol=1e-4)
            self.assertAlmostEqual(float(batched_values[i]), value, places=4)


if __name__ == "__main__":
    unittest.main()
