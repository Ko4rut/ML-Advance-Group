# Xiangqi Game Replay (frontend)

Open **/replay** from the left navigation. The viewer works offline: it reads
the selected file with the browser's File API; no replay endpoints or backend
requests are required. The existing ChessBoard renders the reconstructed
snapshots, with the latest source and destination highlighted.

## Supported files

- A single **Alpha-Beta raw** or **Silver** game object in JSON, with
  \`initial_board\`, \`initial_side\` and \`moves[].action.{source,target}\`.
- A JSON object with \`moves[].move\` containing coordinate notation such as
  \`b2e2\` (Neural MCTS evaluation logs).
- A JSON array of game objects, a JSON object with a \`games\` array, or a
  **JSONL** file containing one game object per nonempty line.
- A plain array of move objects is accepted as one game.

Example:

\`\`\`json
{
  "game_id": "demo",
  "initial_side": "red",
  "moves": [
    {"ply": 1, "action": {"source": 54, "target": 45}},
    {"ply": 2, "action": {"source": 27, "target": 36}}
  ]
}
\`\`\`

When \`initial_board\` is omitted, the viewer starts from the standard engine
opening position. Boards have 90 row-major integers: Black starts at row 0
(negative piece codes), Red at row 9 (positive piece codes). The \`source\`
and \`target\` fields use engine indices 0–89. For letter-number notation,
\`a0\` refers to Red's lower-left corner in engine orientation.

The playback controls provide start, previous, play/pause, next and end,
a seek slider, speed selection, game selection and a clickable move history.
Files are limited to 30 MB, with at most 1,000 games and 1,000 plies per game.

**Validation scope:** files are checked for their shape, move coordinates,
turn order, ownership and optional \`board\`/\`next_board\` consistency.
The local viewer does *not* implement the full Xiangqi rules (check,
checkmate, repetition, etc.); run a backend replay/validation pipeline
for authoritative legal-move verification. A standalone live-match
snapshot without historical moves cannot be replayed.

## Tests

From \`project/frontend\`, with the Next.js development server running:

\`\`\`bash
npx playwright test tests/replay.spec.ts
\`\`\`
