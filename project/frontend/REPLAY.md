# Xiangqi Game Replay (frontend)

Open **/replay** from the left navigation. The viewer works offline: it reads
the selected file with the browser's File API; no replay endpoints or backend
requests are required. The existing ChessBoard renders the reconstructed
snapshots, with the latest source and destination highlighted.

## Supported files

- A single **Alpha-Beta raw** or **Silver** game object in JSON, with
  `initial_board`, `initial_side` and `moves[].action.{source,target}`.
- A JSON object with `moves[].move` containing coordinate notation such as
  `b2e2` (Neural MCTS evaluation logs).
- A JSON array of game objects, a JSON object with a `games` array, or a
  **JSONL** file containing one game object per nonempty line.
- A plain array of move objects is accepted as one game.

Example:

```json
{
  "game_id": "demo",
  "initial_side": "red",
  "moves": [
    {"ply": 1, "action": {"source": 54, "target": 45}},
    {"ply": 2, "action": {"source": 27, "target": 36}}
  ]
}
```

When `initial_board` is omitted, the viewer starts from the standard engine
opening position. Boards have 90 row-major integers: Black starts at row 0
(negative piece codes), Red at row 9 (positive piece codes). The `source`
and `target` fields use engine indices 0–89. For letter-number notation,
`a0` refers to Red's lower-left corner in engine orientation.

The playback controls provide start, previous, play/pause, next and end,
a seek slider, speed selection, game selection and a clickable move history.
Files are limited to 30 MB, with at most 1,000 games and 1,000 plies per game.

**Validation scope:** files are checked for their shape, move coordinates,
turn order, ownership and optional `board`/`next_board` consistency.
The local viewer does *not* implement the full Xiangqi rules (check,
checkmate, repetition, etc.); run a backend replay/validation pipeline
for authoritative legal-move verification. A standalone live-match
snapshot without historical moves cannot be replayed.

## Generate a real AI-vs-AI game for replay

From `project/backend` (no FastAPI server is required):

```bash
# Generate one actual Alpha-Beta (Red) vs Rule-Based (Black) match.
# Change the output filename if you run the command a second time.
python -m src batch --red alphabeta --black rules --games 1 \
  --red-depth 2 --black-depth 2 --max-plies 40 \
  --opening-mode none --output replay-demo.jsonl --quiet
```

Open the frontend's `/replay` page and upload `project/backend/replay-demo.jsonl`.
The batch generator's output already has the proper `initial_board` and
`moves[].action` fields; **no conversion is needed**.

To replay an existing tournament match involving e.g. MCTS or Tabular Q,
extract just one game rather than uploading the large tournament log:

```bash
python -m scripts.export_tournament_replay \
  --input results/tournaments/20261112-091526_501370b5/games.jsonl \
  --red alphabeta --black mcts --output replay-tournament.json
```

The export tool streams the JSONL file, chooses the first non-error match
with those colors, transforms its flat `source`/`target` coordinates into
`moves[].action`, and preserves the original opening and outcome.
Alternatively use `--game-index 0` instead of agent filters to extract
a specific game. The standard initial position is inferred if
`initial_board` is absent. Other available agent names in the
tournament include `random`, `rules`, `heuristic`, `mcts`,
`alphabeta`, and `tabular_q`.

```bash
# Backend exporter tests (from project/backend):
python -m unittest tests.test_export_tournament_replay -v
```

These files contain the **actual decisions made by both agents**, not
generated animations or random playback positions. Training datasets,
tournament logs and full-game API histories are separate concerns.

## Existing two-model game recordings (no generation needed)

These repository files contain completed, engine-validated Neural MCTS vs
classic MCTS games and can be uploaded directly to /replay:

- `../backend/docs/neural_mcts/evidence/phase4/game-large-neural-red-vs-classic.json`
  (Neural as Red, 31 plies, Neural wins).
- `../backend/docs/neural_mcts/evidence/phase4/game-large-neural-black-vs-classic.json`
  (Neural as Black, 16 plies, Neural wins).

Both use `moves[].move` coordinate notation and need no conversion.
The browser tests in `tests/replay.spec.ts` exercise both files.

## Tests

From `project/frontend`, with the Next.js development server running:

```bash
npx playwright test tests/replay.spec.ts
```
