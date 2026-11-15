# Phase 1 isolated checkpoint qualification

This experiment does not change `src/`, core MCTS, backend dependencies, API availability, or weights. It uses original pinned cc-lite source from an external directory under its Apache-2.0 license. The executable source and real checkpoint are SHA-256 checked against `artifacts.json` before loading. There is no training, conversion, BatchNorm recalibration, or fallback predictor.

The CPU runtime plan is about 201 MB of wheels, plus the 15.5 MB pretrained checkpoint; allow approximately 1 GB of temporary storage for extracted packages and artifacts. Installation/download must follow the session's resource approval. Do not install this experimental runtime into the shared environment or backend production requirements.

From `project/backend`, with the existing project `.venv` supplying numpy/Pydantic and declared backend extras:

```sh
PY=../../.venv/bin/python
POC=experiments/neural_mcts/phase1
WORK=$(mktemp -d /tmp/neural-mcts-phase1.XXXXXX)

# Run only after approving the concrete download plan.
"$PY" -m pip install --no-cache-dir --target "$WORK/deps" \
  -r "$POC/requirements-cpu.txt"
"$PY" "$POC/fetch_artifacts.py" --directory "$WORK/source" --with-checkpoint

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WORK/deps:$PWD"
export XIANGQI_PHASE1_SOURCE="$WORK/source"
export XIANGQI_PHASE1_CHECKPOINT="$WORK/source/evidence/distill-v4/train/checkpoint.pt"

"$PY" -m unittest discover -s "$POC" -p test_poc.py -v
"$PY" "$POC/qualify.py" --source "$XIANGQI_PHASE1_SOURCE" \
  --checkpoint "$XIANGQI_PHASE1_CHECKPOINT" --output "$WORK/result.json"
```

Both real-checkpoint test methods skip explicitly if the two environment variables are absent. A run with skips does not pass the real-checkpoint gate. Contract tests use synthetic arrays only for invalid-data behavior; synthetic arrays never substitute for pretrained inference.

Optional second checkpoint (`distill-v4-large`, 38 MB, pinned in `artifacts.json` under `alternate_checkpoints`; not fetched by `fetch_artifacts.py`):

```sh
"$PY" "$POC/qualify.py" --variant distill-v4-large --source "$XIANGQI_PHASE1_SOURCE" \
  --checkpoint "$XIANGQI_PHASE1_SOURCE/evidence/distill-v4-large/train/checkpoint.pt" --output "$WORK/result-large.json"
```

`sanity_probe.py` (same arguments plus optional `--variant`, `--games-per-side`) runs semantic checks: an opening line in UCCI notation, material value-sign probes for both colors, and policy-only (argmax, no search) games against a seeded random mover. It is evidence of semantics, not strength. Results: `docs/neural_mcts/evidence/phase1/`.

`qualify.py` runs 12 curated fixture states (both colors, asymmetric positions, checks, mate-in-one, checkmated positions, repetition and no-capture draws), plus a fixed-seed legal trajectory of at most 40 positions. It compares observations exactly with the original encoder, complete legal-action sets with the source codec, and real P/V results with the original inference wrapper. Full logits and values repeat exactly within the same CPU process. The model state is checked unchanged after all calls. It profiles 20 warm forwards on one CPU thread and reports whole-process peak RSS, which includes imports, fixtures and verification snapshots; this is not model-only memory.

The experiment implements only enough temporary codec/validation logic to qualify the checkpoint. Phase 2 still owns the production encoding contract, comprehensive adapters and regression tests; later phases own inference service and neural MCTS. No MCTS game or strength benchmark runs here.
