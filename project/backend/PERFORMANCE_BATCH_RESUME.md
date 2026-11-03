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


## Structured opening book and Alpha-Beta handoff

A small **versioned, curated template book** now lives in
`src/training/opening_book.py` with **12 legal six-ply opening lines** in
five families: central cannon, pawn advance, flying elephant, horse development
and palace cannon. Red and Black responses are specified together.

The lines are **hand-authored starting templates**, not imported professional
game records and **not certified optimal openings**. Every move is validated
with `GameEngine.legal_moves`, and `validate_opening_book()` checks every
line before batch generation. Extend the book only with replay-validated
sequences; benchmark playing strength separately.

Use `--opening-mode book --opening-plies 6` to choose a uniformly sampled
opening *family* first, then a random variation within that family. The choice
is deterministic for each global `seed + game_index * 1009`, including after
resume, so multiprocessing does not change it. If `opening-plies` exceeds
a line's length, Alpha-Beta takes over when that line ends, never filling
extra moves randomly. A six-ply prefix corresponds to three moves by each side.

```bash
cd project/backend
python -u -m src batch \
  --red alphabeta --black alphabeta \
  --red-depth 3 --black-depth 3 \
  --games 50 --max-plies 240 \
  --opening-mode book --opening-plies 6 \
  --seed 42 --output alphabeta_depth3_book_v1_seed42.jsonl \
  --resume --progress-every 1 --workers 1
```

To retain old behavior, use `--opening-mode random --opening-plies 2`.
Use `--opening-mode none` for Alpha-Beta from the original board. The CLI's
default is `random` for backwards compatibility, while the Colab notebook
now explicitly sets `OPENING_MODE = 'book'`.

Every game record includes:
- `opening_mode`, `opening_book_version`, `opening_family`,
  `opening_id`, `opening_name`, and `opening_applied_plies`.
- Scripted move records have `actor = 'opening_book'`, so they are **not**
  silently labeled as Alpha-Beta demonstrations.
- The actual search begins immediately after the scripted prefix, with
  the full state/history retained so game rules and repetition still work.

**Resume safety:** the batch validates `opening_mode` and book version
against the existing JSONL before appending. Old datasets without an
`opening_mode` field continue to work as legacy `random` datasets. Always
use a *new RUN_NAME* if the opening mode, prefix length, or book version
changes. The book implementation is strictly CPU-only and requires no model.

Run focused tests:

```bash
python -m unittest -v tests.test_opening_book tests.test_batch_games
```

This release deliberately does **not** add Top-K softmax sampling:
uncalibrated evaluation scores can introduce poor moves. A broader,
expert-reviewed opening book and separately evaluated Top-K exploration can
be added once gameplay outcome statistics establish a useful baseline.

