# Phase 2 experiments

Support files for `docs/neural_mcts/02_CONTRACT.md`. Neither script is imported by `src/`.

- `generate_golden.py` — builds `tests/encoding/golden/cclite_v1.json` from the **upstream** cc-lite encoder/codec (source SHA-256-verified against `../phase1/artifacts.json`). Needs numpy and the Phase 1 source directory; output is deterministic.
- `test_real_checkpoint.py` — runs the production `src/encoding` stack through the real checkpoint and compares to upstream's evaluator. Explicitly skipped unless `XIANGQI_PHASE1_SOURCE` and `XIANGQI_PHASE1_CHECKPOINT` are set (optional `XIANGQI_PHASE1_VARIANT=distill-v4-large`).

Setup of the torch runtime and sources is described in `../phase1/README.md`. Example, from `project/backend`:

```sh
W=/tmp/neural-mcts-phase1   # runtime (deps/), pinned source (source/) from phase1
export PYTHONPATH="$W/deps:$PWD:$PWD/experiments/neural_mcts/phase1"
export XIANGQI_PHASE1_SOURCE=$W/source
XIANGQI_PHASE1_CHECKPOINT=$W/source/evidence/distill-v4/train/checkpoint.pt \
  ../../.venv/bin/python -m unittest discover -s experiments/neural_mcts/phase2 -p test_real_checkpoint.py -v
```
