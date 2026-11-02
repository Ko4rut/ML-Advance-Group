# Search evaluation improvements (Minimax & Alpha-Beta)

Branch: `feature/search-evaluation-improvements`.

## What changed

The previous heuristic was material-only. The new fast evaluator adds:

- **Material:** unchanged values: king=10000, rook=90, cannon=45, horse=40, advisor=20, elephant=20, pawn=10.
- **Position:** small, symmetrical centralization/development bonuses for horse, rook and cannon, and additional reward for advancing pawns and crossing the river.
- **King protection:** small bonus for nearby friendly advisors and elephants. This is **not** a full tactical king-safety model.
- **Nonterminal repetition:** mild penalty for revisiting a position when non-losing and a mild bonus when behind. Official engine results are still authoritative: terminal checkmates/stalemates/draws are recognized and scored before evaluating leaf nodes.
- **Move ordering:** root move ordering shared by both agents for deterministic tie-breaking; Alpha-Beta additionally orders internal moves to improve pruning.

These are **hand-crafted baseline weights**, not learned weights. Search depth defaults are unchanged (2). Rule handling, A*, frontend and API interfaces are unchanged. The old `material_score()` remains callable for baseline measurement.

## Run tests

From `project/backend`:

```powershell
python -m unittest discover -s tests -p "test_position_evaluator.py" -v
python -m unittest discover -s tests -p "test_minimax_agent.py" -v
python -m unittest discover -s tests -p "test_alphabeta_agent.py" -v
```

## Benchmark protocol

Use the same opening seeds, opponent pairings, depth, max plies, and machine as the baseline. Track draw reasons (especially repetition), truncations, number of unique game fingerprints, average plies, nodes/move, time/move, and wins against the old agent.

**Important:** fewer draws alone does not prove better play. The evaluator may still loop, and some draws are good defensive decisions. Avoid labelling truncated games as draws. Increasing depth may remain expensive because `GameEngine` performs legal move generation at most search nodes.

If you need the terminal batch dataset runner on this branch, merge or cherry-pick the separate `feature/game-data-pipeline` branch after its tests pass. It is intentionally not pulled into this independent evaluator PR.
