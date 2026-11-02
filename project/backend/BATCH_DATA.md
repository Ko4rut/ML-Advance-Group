# CPU-only batch dataset generator

Run from `project/backend`. No frontend, API, Playwright or GPU is needed.

## Quick smoke test

```powershell
python -m src batch --red random --black random --games 3 --max-plies 12 --output data/smoke.jsonl
python -m unittest discover -s tests -p "test_batch_games.py" -v
```

## Sample Alpha-Beta batches

```powershell
python -m src batch --red alphabeta --black alphabeta --red-depth 2 --black-depth 2 --games 10 --max-plies 160 --opening-plies 2 --seed 42 --output data/ab_vs_ab_001.jsonl
python -m src batch --red alphabeta --black rules --red-depth 3 --games 10 --max-plies 160 --opening-plies 2 --seed 99 --output data/ab_vs_rules_001.jsonl
```

Dataset files are JSONL: one complete game per line. Output paths must be new (won't overwrite existing files). The game file contains the 90-square initial board, each selected legal action, before/after boards, side, acting agent, reward, search node count, timing, result, and a fingerprint for duplicate detection. Records also contain the parameters for reproducibility. Random opening plies have `actor: "opening_random"` and must NOT be treated as strong-agent labels.

The `--no-progress-limit` argument optionally truncates games earlier than the environment's official no-capture rule. Truncated results have status `truncated` and winner `null` and must NOT be treated as legitimate draws or wins when constructing value-network training labels. The `--max-plies` limit follows the same rule.

Do not assume homogeneous deterministic Alpha-Beta games provide a diverse or expert-grade dataset. To produce a useful corpus, vary agent pairings, opening plies, seeds, and search depths, and compare duplicate fingerprints, draw rate, truncated rate and runtime. AlphaZero still requires self-play MCTS policies and outcomes, which this collector does not fabricate.

Start on local CPU: with the current engine and depth-limited Alpha-Beta, CPU search is the dominant work. Adding a GPU or using Colab generally doesn't accelerate these Python search agents. A later iteration can parallelize across worker processes or add optional notebook-compatible runners.

**Current limitations:** this is a first-pass generator (not a production replay storage system); schema version is 1, and there is no command to resume partially generated batch files or run multiprocessing. Training data should undergo replay validation and train/test splitting by whole game. The runner does not modify any agent's evaluation strategy.
