"""Production adapter vs the LIVE upstream cc-lite evaluator, on the REAL checkpoint.

Unlike tests/networks/test_real_checkpoint.py (which compares with stored upstream
outputs) this runs the pinned upstream ``TorchEvaluator`` side by side on every
non-terminal state of the Phase 2 golden fixture. Explicitly skipped (never "passed")
unless XIANGQI_PHASE1_SOURCE points at the pinned upstream source (Phase 1 README) and
the checkpoint is provisioned under models/neural/ (or XIANGQI_NEURAL_MODEL_PATH).
Optional: XIANGQI_PHASE1_VARIANT (default distill-v4-large).

    W=/tmp/neural-mcts-phase1   # runtime + source from experiments/neural_mcts/phase1/README.md
    PYTHONPATH=$W/deps:$PWD XIANGQI_PHASE1_SOURCE=$W/source XIANGQI_PHASE1_VARIANT=distill-v4 \\
      python -m unittest discover -s experiments/neural_mcts/phase3 -p test_real_checkpoint.py -v
"""

import json
import os
from pathlib import Path
import sys
import unittest

BACKEND = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND / "experiments" / "neural_mcts" / "phase1"))

from src.environment import GameEngine, GameState, Side  # noqa: E402
from src.networks.pretrained import NeuralModelSettings, load_adapter  # noqa: E402

GOLDEN = BACKEND / "tests" / "encoding" / "golden" / "cclite_v1.json"
VARIANT = os.environ.get("XIANGQI_PHASE1_VARIANT", "distill-v4-large")


@unittest.skipUnless(os.environ.get("XIANGQI_PHASE1_SOURCE"),
                     "XIANGQI_PHASE1_SOURCE (pinned upstream source) not provided; live upstream parity NOT run")
class LiveUpstreamParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qualify import CheckpointProbe
        source = Path(os.environ["XIANGQI_PHASE1_SOURCE"])
        checkpoint = source / "evidence" / VARIANT / "train" / "checkpoint.pt"
        cls.probe = CheckpointProbe(source, checkpoint, VARIANT)  # verifies upstream source + checkpoint hashes
        # Same process: the probe fixed torch to 1 thread, so both stacks run under identical settings.
        cls.adapter = load_adapter(NeuralModelSettings(model=VARIANT))
        cls.engine = GameEngine()

    def test_priors_and_value_match_upstream_on_every_nonterminal_golden_state(self):
        from contracts import fen
        records = json.loads(GOLDEN.read_text())["states"]
        states = checked = moves_checked = 0
        worst_prior = worst_value = 0.0
        bitwise_equal_logits = True
        for record in records:
            state = GameState(tuple(record["board"]), Side[record["side"]])
            moves = self.engine.legal_moves(state)
            if not moves:
                continue
            states += 1
            native = self.probe.Board.from_fen(fen(state))
            expected, expected_value = self.probe.reference.evaluate(native)
            codec = self.probe.native_encoding.MoveCodec
            reference = {codec.encode(m): p for m, p in expected.items()}
            result = self.adapter.predict(state, moves)
            got = {m.source * 90 + m.target: p for m, p in result.priors.items()}
            self.assertEqual(set(got), set(reference), record["name"])
            worst_prior = max(worst_prior, max(abs(got[i] - reference[i]) for i in got))
            worst_value = max(worst_value, abs(result.value - expected_value))
            moves_checked += len(moves)
            # raw logits: production backend vs upstream wrapper's forward on the same observation
            observation = self.adapter._encoder.encode(state)
            ours, _ = self.adapter.backend.infer(observation)
            theirs, _ = self.probe.forward(self.probe.native_encoding.encode_board(native))
            bitwise_equal_logits &= list(map(float, theirs)) == ours
            checked += 1
        self.assertGreaterEqual(checked, 400)
        self.assertLess(worst_prior, 1e-6)
        self.assertLess(worst_value, 1e-6)
        print(f"[{VARIANT}] live upstream parity: {checked} states / {moves_checked} legal moves, "
              f"max prior err {worst_prior:.2e}, max value err {worst_value:.2e}, "
              f"raw logits bit-identical: {bitwise_equal_logits}")


if __name__ == "__main__":
    unittest.main()
