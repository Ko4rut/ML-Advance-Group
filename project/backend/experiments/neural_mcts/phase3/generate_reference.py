"""Generate tests/networks/golden/real_reference.json from the UPSTREAM implementation.

Expected values come from the pinned, SHA-256-verified cc-lite source and its
``TorchEvaluator`` (via the Phase 1 ``CheckpointProbe``), never from the production
adapter. Needs the Phase 1 runtime (torch) and source directory; see
experiments/neural_mcts/phase1/README.md. Every ``STRIDE``-th non-terminal state of the
Phase 2 golden fixture is recorded (all legal-move priors + value) for both checkpoints.
"""

import argparse
import json
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(BACKEND), str(BACKEND / "experiments" / "neural_mcts" / "phase1")]

from contracts import fen  # noqa: E402
from qualify import CheckpointProbe  # noqa: E402
from src.environment import GameEngine, GameState, Side  # noqa: E402

GOLDEN = BACKEND / "tests" / "encoding" / "golden" / "cclite_v1.json"
OUTPUT = BACKEND / "tests" / "networks" / "golden" / "real_reference.json"
STRIDE = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=("distill-v4-large", "distill-v4"),
                        help="one variant per process (the probe fixes torch's interop threads); "
                             "results are merged into the output file")
    args = parser.parse_args()
    engine = GameEngine()
    states = [(r["name"], GameState(tuple(r["board"]), Side[r["side"]]))
              for r in json.loads(GOLDEN.read_text())["states"]]
    states = [(n, s) for n, s in states if engine.legal_moves(s)][::STRIDE]
    document = (json.loads(OUTPUT.read_text()) if OUTPUT.exists() else
                {"generator": "experiments/neural_mcts/phase3/generate_reference.py",
                 "upstream_revision": None, "stride": STRIDE, "models": {}})
    for variant in (args.variant,):
        checkpoint = args.source / "evidence" / variant / "train" / "checkpoint.pt"
        probe = CheckpointProbe(args.source, checkpoint, variant)
        document["upstream_revision"] = probe.manifest["revision"]
        entries = []
        for name, state in states:
            native = probe.Board.from_fen(fen(state))
            priors, value = probe.reference.evaluate(native)
            codec = probe.native_encoding.MoveCodec
            entries.append({
                "name": name, "board": list(state.board), "side": state.side_to_move.name,
                "value": round(float(value), 9),
                "priors": sorted([codec.encode(m), round(float(p), 9)] for m, p in priors.items()),
            })
        document["models"][variant] = {"checkpoint_sha256": probe.manifest["checkpoint"]["sha256"]
                                       if variant == "distill-v4"
                                       else probe.manifest["alternate_checkpoints"][variant]["sha256"],
                                       "states": entries}
    document["models"] = dict(sorted(document["models"].items()))
    OUTPUT.write_text(json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes, {len(states)} states/model)")


if __name__ == "__main__":
    main()
