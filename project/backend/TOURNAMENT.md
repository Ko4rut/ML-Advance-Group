# Round-robin tournament between Xiangqi agents

One command makes the agents play each other in a round-robin and writes a raw
game log plus a full statistical report (standings, Elo with confidence
intervals, crosstable, per-pair conclusions, cost, warnings).

```powershell
cd project/backend
python -m src tournament --config configs/tournament.yaml --workers 5
```

All commands below are run from `project/backend`. Paths use PowerShell style
(Windows), but everything also works on Linux/macOS.

## 1. Purpose and design

* **Agents:** `random`, `rules`, `alphabeta`, `heuristic`, `mcts`, `tabular_q`
  (6 agents → 15 pairs → 750 games). If the Tabular Q checkpoint is missing,
  `tabular_q` is excluded with a loud warning (recorded in the manifest and the
  report) and the tournament runs with 5 agents (10 pairs, 500 games).
* **25 openings × 2 colors = 50 games per pair.** Each opening is played twice
  in every pair: once with A as Red, once with A as Black. Balanced colors
  cancel the first-move advantage and any lopsided opening.
* **Why fixed openings?** Search agents (`rules`, `alphabeta`, `tabular_q`) are
  deterministic: replaying a pair from the initial position gives the *same*
  game every time, so "50 games" would be one game repeated. The variety must
  come from the opening. The 25 openings (all with distinct final positions)
  come from the curated book `xiangqi-curated-v2` (`docs/OPENING_BOOK_V2.md`).
* **Truncation:** a game that reaches `max_plies` (300 plies, opening plies
  included) is `truncated`. It is **not** a draw. It is reported separately
  (see Limitations).
* The tournament **measures, it never tunes**: it does not change any agent's
  algorithm or parameters, and it does not use `MatchRunner` (which applies one
  budget to both sides and always starts from the initial position).

### Agents and budgets

Every agent runs with the **defaults the web API uses** (checked by
`tests/evaluation/test_api_parity.py`). The tournament only differs in seeds.

| Agent | Configuration | Deterministic? | Why |
|---|---|---|---|
| `random` | `RandomAgent(seed)`, one seed per game | seeded: exactly replayable | Baseline (Elo anchor = 0). |
| `rules` | `RuleBasedAgent` | yes | Fast, simple baseline. |
| `alphabeta` | `AlphaBetaAgent` default depth (2), **no time limit** | yes | Depth-only budget keeps it independent of machine speed. |
| `heuristic` | `HeuristicAgent` constructor defaults (depth 6, 2000 nodes, **1 s** time limit) | **no (clock-dependent)** | That is what the API serves. It has its own 1-second deadline, so the move can depend on machine speed and load. |
| `mcts` | `configs/mcts.yaml`, `time_limit` = 1 s per move, new seed per game | **no (time- and seed-dependent)** | The experiment is defined by "1 second per move", as before. For exact replay set `max_simulations` in the config. |
| `tabular_q` | Greedy (`epsilon=0`), fallback to `rules`, checkpoint from `XIANGQI_TABULAR_Q_CHECKPOINT` (default `models/tabular_q/q_table.json.gz`) | yes | Same as the API. See Limitations: most moves fall back to `rules`. |

**Why no Minimax?** At the same depth it picks (almost) the same moves as
Alpha-Beta, so it would add a near-duplicate player and 5 more pairs.

**Why does MCTS use wall-clock time?** It mirrors how it is used and how the
earlier experiments (`run_mcts_matches.py`) were run. The cost is that its
strength depends on how many simulations fit in one second, which depends on
the machine and on how many games run in parallel (see Limitations).

Each game builds **fresh** agents inside the worker process; nothing is shared
between games.

## 2. Running (Windows PowerShell)

```powershell
# 1. Quick check (2 openings x 2 colors, max 40 plies): minutes, separate folder smoke_...
python -m src tournament --config configs/tournament.yaml --smoke --workers 5

# 2. Pilot: 1 game per pair at the REAL config; prints a total-time estimate.
python -m src tournament --config configs/tournament.yaml --pilot --workers 5

# 3. The real tournament
python -m src tournament --config configs/tournament.yaml --workers 5

# 4. Continue after Ctrl+C, a crash or a power-off
python -m src tournament --resume results\tournaments\<run_id>

# 5. Regenerate summary\ (CSV, plots, report.md) from the raw log only
python -m src tournament-report results\tournaments\<run_id>
python -m src tournament-report results\tournaments\<run_id> --no-plots
```

Options of `tournament`: `--workers N|auto`, `--output-root DIR`,
`--allow-code-change` and `--retry-errors` (both only with `--resume`).

**Workers.** Do not exceed the number of *physical* cores: MCTS and the
heuristic agent are time-limited, so an over-subscribed machine gives them
fewer simulations/nodes per move and lowers their strength. A laptop with an
Intel i5-13420H has 8 physical cores (12 threads): use **`--workers 5`**. (`auto` =
`cpu_count - 1` counts threads, which is too many for this purpose.)

**Running overnight.**

```powershell
powercfg /change standby-timeout-ac 0     # never sleep while plugged in
powercfg /change monitor-timeout-ac 0     # optional
```

Keep the charger plugged in and close heavy programs. If the machine shuts
down anyway, just run the `--resume` command: finished games are never lost or
repeated (at most the games that were in flight are replayed). Restore your
power settings afterwards, e.g. `powercfg /change standby-timeout-ac 30`.

**Resume rules.**

* The config is re-resolved and its `config_hash` must equal the run's; otherwise
  the resume stops and prints the keys that changed.
* If the git commit changed, resume refuses unless `--allow-code-change` is given;
  the event is stored in `manifest.json` → `resume_history` and warned about in the report.
* A torn last line of `games.jsonl` (crash mid-write) is discarded with a warning;
  a corrupt line elsewhere or a duplicate `game_index` is an error.
* Games with `status="error"` are not re-run automatically; use `--retry-errors`
  (the old records are kept in `games.retried.jsonl`).
* Ctrl+C stops cleanly and prints the resume command.

**Progress** is printed one line per game:
`[123/750] alphabeta vs mcts | opening=… | A=red | mcts wins (checkmate) | 87 plies | 64.2s | ETA 5h12m`.

## 3. Output files

```
results/tournaments/<run_id>/       # run_id = [smoke_|pilot_]YYYYMMDD-HHMMSS_<config_hash[:8]>
├── manifest.json         provenance: git commit/branch/dirty, machine, Python, packages, command,
│                         config + hash, agent specs (incl. is_deterministic), excluded agents,
│                         opening hashes, workers, timings, resume_history, status
├── config.resolved.yaml  config after validation (resolved paths, defaults filled in)
├── games.jsonl           RAW LOG, one game per line: the single source of truth
├── progress.json         completed/total/errors/truncated/ETA (can be rebuilt from games.jsonl)
├── run.log               text log (also printed to the console)
├── git.diff              only if the working tree was dirty at the start
├── games.retried.jsonl   only after --retry-errors
└── summary/              EVERYTHING here is generated from manifest.json + games.jsonl
    ├── standings.csv  pair_results.csv  crosstable.csv  color_breakdown.csv
    ├── cost.csv  openings.csv  errors.csv (only if some game errored)
    ├── report.md         Vietnamese report with all tables and explanations
    └── plots/ crosstable_heatmap.png  elo_ci.png  cost_per_move.png
```

`results/tournaments/` is git-ignored. CSV files and `report.md` contain no
generation timestamp and use fixed number formats and `\n` line endings, so
regenerating them gives **byte-identical** files.

### `games.jsonl` schema (`schema_version: 1`)

One JSON object per line (`ensure_ascii=False`, compact separators):

| Field | Meaning |
|---|---|
| `schema_version`, `run_id`, `config_hash` | Record version and run identity. |
| `game_index`, `game_key` | Position in the schedule (0…N-1) and stable key `A\|B\|opening_id\|a_color`. |
| `pair` | `[A, B]`; A comes first in config order. |
| `red_agent`, `black_agent`, `a_color` | Who plays which side; `a_color` is A's color. |
| `opening` | `{id, family, name, book_version, plies}`. |
| `seeds` | `{game, red, black}`, derived with SHA-256 from `master_seed` and `game_key`. |
| `budgets` | `{red, black}`: the `SearchBudget` given to each side (or `null`). |
| `moves` | List of `{ply, side, actor, source, target, think_seconds, nodes, simulations, supported?}`. `actor` is `"opening"` for book plies (cost 0, excluded from cost stats) or the agent name. `supported` only for `tabular_q` (state found in the Q table). |
| `outcome` | `{status: finished\|truncated\|error, winner: red\|black\|null, reason}`. |
| `plies`, `agent_plies` | Total plies and plies played by each side's agent. |
| `fingerprint` | First 16 hex chars of SHA-256 over the full `[source, target]` list, opening included. |
| `elapsed_seconds`, `worker_pid`, `finished_at` | Wall time, worker process and UTC timestamp. |
| `error` | Only for `status="error"`: `{type, message, traceback, ply, actor, side}`. |

Squares are the engine's 0–89 indices (`row * 9 + col`, row 0 = Black's back rank).

### What the report computes

* Per game, from one agent's view: **W/D/L/T/E**. Win = 1, draw = 0.5, loss = 0.
* **Truncated (T)** is shown two ways: `T_half` (0.5 point, the main figure) and
  `excl_T` (games left out). If the two rank agents differently the report says so.
* **Error (E)** games are excluded from every score and listed in `errors.csv`.
* **Pair conclusion:** Wilson 95% interval on A's score; entirely above 50% → A
  stronger, entirely below → B stronger, otherwise "not enough evidence".
* **Elo:** Bradley–Terry MLE (MM algorithm, Hunter 2004), draws = half a win,
  one virtual draw per pair as a light prior, `random` anchored at 0, scale
  `400·log10(strength)`. 95% CI from 1000 bootstrap resamples of the games within
  each pair (`random.Random(master_seed)`, so reproducible).
* **Cost:** thinking time (mean/median/p95/total), average nodes, average MCTS
  simulations and the Tabular Q fallback rate, counting only moves the agent made.
* **Colors and openings:** score as Red vs Black, overall first-move advantage,
  and per-opening result shares; openings where one side wins ≥ 70% of valid
  games are flagged.

## 4. Reproducibility

To re-run: check out the commit in `manifest.json`, use the same config and
`master_seed`, and run `tournament`. `report.md` section 9 prints the exact commands.

| Agent | Reproducible move-for-move? |
|---|---|
| `rules`, `alphabeta`, `tabular_q` (with the same checkpoint, verified by SHA-256 in the manifest) | **Yes.** |
| `random` | **Yes**, the seed of each game is derived from `master_seed` and the game key. |
| `heuristic` | **Statistically only.** Its 1-second deadline makes node-limited search depend on speed and load. |
| `mcts` | **Statistically only.** Time-limited search + a different number of simulations per run. Set `max_simulations` in the `mcts` entry of the config (it changes the config hash) for an exact replay. |

A tournament that involves `heuristic` or `mcts` will therefore give similar
standings, not identical game records. Results never depend on completion
order; they depend on the worker count only through the time-limited agents.

## 5. Limitations

* **Wilson interval is an approximation** when there are draws (a draw counts as
  half a success, which is not a binomial trial). Conclusions of the 15 pairs are
  not corrected for multiple comparisons.
* **Bradley–Terry prior:** one virtual draw per pair shrinks Elo slightly towards
  0 and keeps ratings finite; Elo is only meaningful as differences between agents
  (the anchor `random` = 0).
* **Truncated games** are neither wins nor draws; `T_half` is a convention. If a
  large share of games is truncated (e.g. weak agents shuffling pieces), rankings
  are sensitive to it. Compare `T_half` and `excl_T`.
* **Parallelism changes MCTS and heuristic.** More workers than physical cores
  means fewer simulations (MCTS) and nodes/depth (heuristic) per second. The report
  lists the mean simulations per MCTS move and the worker count.
* **Tabular Q depends on its checkpoint** and mostly behaves like `rules`: in the
  smoke run only about 4% of its moves were found in the Q table, ~96% fell back
  to the rule-based agent. When two agents behave identically, colour-swapped games
  can be identical (the report flags pairs with `distinct_ratio` < 1).
* **Which checkpoint does the web serve?** Not verified: it is unclear whether the
  deployed web app uses the local `models/tabular_q/q_table.json.gz` or a file
  from Drive. The tournament uses `XIANGQI_TABULAR_Q_CHECKPOINT` from `.env`, else
  the local default, and records its SHA-256 in the manifest.
* The results are only valid for this opening set, `max_plies` and agent configuration.
* A dirty git tree is recorded (`git.diff`) and warned about, but results are then not
  tied to a clean commit.
