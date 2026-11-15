# Phase 3 experiments and evidence tooling

Production code lives in `src/networks/pretrained/`. This directory holds the scripts that
exercise or measure it; none is imported by the backend.

| File | Purpose |
| --- | --- |
| `demo.py` | Load the configured checkpoint, print priors/value, show terminal precedence, argmax self-play legality check |
| `benchmark.py` | Load time, per-call latency (`predict` and raw forward), adapter overhead, RSS; one (model, threads) per process |
| `generate_reference.py` | Regenerates `tests/networks/golden/real_reference.json` from the **upstream** implementation (needs the Phase 1 runtime/source) |
| `test_real_checkpoint.py` | Production adapter vs the live upstream evaluator on all 467 non-terminal golden states (needs `XIANGQI_PHASE1_SOURCE`) |

Setup of the upstream source/runtime is in `../phase1/README.md`. Provision checkpoints as in
`models/neural/README.md`. Evidence logs: `docs/neural_mcts/evidence/phase3/`.

```sh
cd project/backend
W=/tmp/neural-mcts-phase1        # runtime (deps/) + pinned upstream source (source/) from Phase 1
export PYTHONPATH=$W/deps:$PWD
python experiments/neural_mcts/phase3/demo.py
python experiments/neural_mcts/phase3/benchmark.py --model distill-v4-large --threads 4 --output /tmp/b.json
XIANGQI_PHASE1_SOURCE=$W/source XIANGQI_PHASE1_VARIANT=distill-v4 \
  python -m unittest discover -s experiments/neural_mcts/phase3 -p test_real_checkpoint.py -v
```
