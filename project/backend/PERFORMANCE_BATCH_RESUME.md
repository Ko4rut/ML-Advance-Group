# Xiangqi CPU performance, reproducible benchmark, and restart-safe batches

All commands below run from `project/backend` in branch
`perf/alphabeta-batch-resume`. No changes are required to the public API.

## What is optimized

- Candidate legality checks run on one reusable board, avoiding a temporary
  GameState and full game-history validation for every pseudo-legal move.
- The engine asks whether **any** legal reply exists for terminal checks
  instead of creating a complete list at leaf nodes.
- Trusted GameState transitions avoid re-validating every historical position.
- Alpha-Beta tries the previous iterative-deepening principal variation first
  while preserving original root-order tie breaking.

The benchmark contains the original move generator, original state transition,
and original Alpha-Beta root order in a separate reference implementation.
Speedup numbers must be **measured on the same machine**, not inferred from
wall times of different games.

## Benchmark original vs optimized

```bash
cd project/backend
python -m src.evaluation.compare_alphabeta --depth 2 --positions 5 --repeats 3
```

The CLI prints original seconds, optimized seconds, nodes, speedup ratio, and
`same_move` for every position, followed by JSON summary. Comparing by
fixed seed and positions avoids differences in game length. More positions
and repetitions produce a more stable estimate; `--depth 3` can be expensive.

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Resumable dataset collection

```bash
cd project/backend
python -u -m src batch \
  --red alphabeta --black alphabeta --red-depth 2 --black-depth 2 \
  --games 50 --max-plies 240 --opening-plies 2 --seed 42 \
  --output /path/to/drive/alphabeta_depth2_seed42_max240_v1.jsonl \
  --resume --progress-every 1 --ply-progress-every 20 --workers 1
```

To resume after interruption, **run exactly the same command**. A complete
game survives restart, but the in-progress game must start again. Increasing
`--games` from 10 to 50 is supported: it expands the target count while
preserving finished indices. A new dataset file/run name is mandatory when
changing agent, depth, seed, max plies, opening, or no-progress limit.

Use `--workers 2` to run independent games concurrently on two CPU cores.
Workers only compute; the parent process writes JSONL and updates the sidecar
checkpoint. Avoid starting two notebook instances writing the **same file**.

- The dataset is the authoritative source of completed `game_index` values.
- JSONL is flushed after each game; sidecar checkpoint updated immediately
  afterward via atomic rename.
- A torn last JSONL line is dropped during resume; malformed complete lines,
  duplicate indices, or incompatible configs raise an error.
- Fingerprints detect exact duplicate move sequences but **do not** detect
  shared opening prefixes; global indices and seeds prevent replay in a
  single run.
- A `truncated` game is never silently labelled as a draw.
- Record-level checkpointing is **not** mid-game continuation.

The Colab notebook lives at
`project/backend/notebooks/xiangqi_colab_batch_runner.ipynb`.
Keep `RUN_NAME` unchanged to resume and inspect
`<RUN_NAME>.jsonl.checkpoint.json` alongside the dataset.

## Notes on correctness and performance

This first optimization pass deliberately avoids unsafe transposition caching,
which would need repetition-history- and no-capture-aware keys. GPU does not
accelerate this Python search implementation. Python multiprocessing scales
throughput rather than necessarily reducing time per match.

For comparable strength, benchmark identical positions and confirm
`same_move` and existing unit tests before large data generation.

## Threat-aware evaluation (November 2026)

The positional evaluation now **reduces the Horse development bonus from +10
to +4**, without changing the material values (Horse 40, Cannon 45, Rook 90).
It also adds `src/agents/threat_evaluator.py` for cheap static pseudo-attack
scoring, limited to threatened horses, cannons and rooks:

- Detects horse-leg and elephant-eye blockers, the one-screen cannon attack,
  rook sliding attacks, pawn movement and palace limits.
- Distinguishes unprotected targets from defended pieces, and cheaper attackers
  from equal/more-expensive attackers.
- Weighs risk more heavily when it is **the attacker's turn**, because the
  threatened piece may not have a chance to retreat.
- Avoids legal-move generation or state copies inside evaluation.

This is not quiescence search or full Static Exchange Evaluation (SEE). Pinned
defenders, conditional attacks and complex recapture sequences may still be
misjudged; depth-limited tactical errors remain possible.

Run focused checks:

```bash
python -m unittest -v tests.test_position_evaluator tests.test_threat_evaluator
```

**Benchmark interpretation:** `compare_alphabeta.py` contrasts the original
state/move-generation/search mechanics against optimized mechanics, but both
branches use the **current** evaluator. Therefore the ratio measures engine
speedup under the same threat-aware evaluation; it is not a measurement of
before-vs-after decision quality or of the incremental cost of threat scoring.
Test both changes with the same opening seeds and whole-game metrics before
using them to label a large training dataset. Dataset configurations should
use a fresh run name if the evaluation model changed between runs, even when
other CLI settings remain unchanged.

