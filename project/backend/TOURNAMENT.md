# Round-robin tournament between Xiangqi agents

One command makes the agents play each other in a round-robin and writes a raw
game log plus a full statistical report (standings, Elo with confidence
intervals, crosstable, per-pair conclusions, cost, warnings).

```powershell
cd project/backend
python -m src tournament --config configs/tournament_default.yaml --workers 5      # budget_mode: default
python -m src tournament --config configs/tournament_equal_time.yaml --workers 5   # budget_mode: equal_time
```

All commands below are run from `project/backend`. Paths use PowerShell style
(Windows), but everything also works on Linux/macOS. `neural_mcts` needs PyTorch, so
run with the interpreter that has it installed (on this repo: the virtualenv in the repo root).

## 1. Purpose and design

* **Agents (8):** `random`, `rules`, `alphabeta`, `minimax`, `heuristic`, `mcts`,
  `neural_mcts`, `tabular_q` → **28 pairs**. Optional agents that cannot be built
  (`tabular_q` without its checkpoint, `neural_mcts` without PyTorch or the pinned
  checkpoint) are excluded with a loud warning (log, `manifest.json` →
  `excluded_agents`, and the report) when `required: false`; with `required: true` the run
  stops with an error instead. For a real run set `required: true` so that an agent can never
  disappear silently.
* **`dqn` is not in the tournament.** `src/agents/dqn_agent.py` is an empty skeleton (no
  trained network, no inference code): there is nothing to measure, and a placeholder
  would only add 7 pairs of meaningless games. It is rejected as an unknown agent. Add it to
  `agent_registry.REGISTRY` once it plays.
* **100 games per pair = 50 openings × 2 colors** → **28 pairs × 100 = 2800 games**
  (7 pairs × 100 = 700 games per agent). Each opening is played twice in every pair, once
  with A as Red and once with A as Black, which cancels the first-move advantage and any
  lopsided opening. `min_games_per_pair` (default **100**) makes `resolve_config`
  reject a full run with fewer games per pair; `--smoke` and `--pilot` are exempt, and so is
  `--resume` of an existing run. A deliberately smaller experiment must say so by lowering
  `min_games_per_pair` in its config.
* **Why fixed openings?** Search agents (`rules`, `alphabeta`, `minimax`, `tabular_q`) are
  deterministic: replaying a pair from the initial position gives the *same* game every
  time, so "100 games" would be one game repeated. The variety must come from the openings,
  so **the count may never be reached by repeating openings**: `resolve_config` rejects
  duplicate ids and duplicate final positions.
* **Opening book `xiangqi-curated-v3`** (`docs/OPENING_BOOK_V3.md`): 50 lines, 50 distinct
  final positions, all capture-free with equal material and `|evaluate_position| ≤ 30` for
  Red to move (largest actual value: 24; one horse = 40). It is a new book, because
  published books are immutable: v1 and v2 stay byte-for-byte unchanged (their SHA-256 are
  pinned by tests) so old runs remain reproducible. v3 = 32 v2 lines (after removing
  transposition duplicates and unbalanced finals) + 18 extensions of 2 plies, chosen by a
  deterministic recipe (`python -m scripts.generate_opening_book_v3 [--check|--doc]`, no
  randomness) and frozen as data in `src/training/opening_book.py`.
* **Truncation:** a game that reaches `max_plies` (300 plies, opening plies
  included) is `truncated`. It is **not** a draw. It is reported separately
  (see Limitations).
* The tournament **measures, it never tunes**: it does not change any agent's
  algorithm or parameters (in `equal_time` it only lifts caps so that time is the limit), and it
  does not use `MatchRunner`.

### Two experiments: `budget_mode`

`budget_mode` is a config key; it is part of the config hash (so a run cannot be resumed
under the other mode), and it is written to `manifest.json`, to **every record** of
`games.jsonl`, and to `report.md`.

| | `default` (`configs/tournament_default.yaml`) | `equal_time` (`configs/tournament_equal_time.yaml`) |
|---|---|---|
| Question answered | "Which *shipped* agent is strongest, as the web app serves it?" | "Which *algorithm* is strongest for the same thinking time?" |
| Budget | Each agent keeps its own code/web-API defaults. | Every searching agent gets `SearchBudget(time_limit=T)` per move, `T = time_per_move` (1.0 s). Time is the **only** limit. |
| `alphabeta`, `minimax` | depth 2 (constructor default), no time limit. | `max_depth=64` + `time_limit=T`: iterative deepening goes as deep as the clock allows and keeps the last *completed* depth. |
| `heuristic` | constructor defaults: depth 6, 2000 nodes, 1 s. | `max_depth=64`, `max_nodes=10⁹`, `time_limit=T`. |
| `mcts` | `configs/mcts.yaml`: 1 s per move. | `time_limit=T`, no `max_simulations`. |
| `neural_mcts` | `configs/neural_mcts.yaml`: `c_puct` 1.5, soft `time_limit` 3 s. | `time_limit=T`, no `max_simulations`. |
| `random`, `rules`, `tabular_q` | as before. | **Budget not applicable** (`budget_applicable: false` in the manifest, stated in the report): they do not search, play as usual, and are **never made to sleep**. |
| Determinism | `random`/`rules`/`alphabeta`/`minimax`/`tabular_q` replay move for move. | Only `random`/`rules`/`tabular_q`; the clock decides how deep the others search. |

Both configs share the same opening set, agent list and `master_seed`, so the two runs are
directly comparable. In `equal_time`, per-agent budget keys (`max_depth`, `max_nodes`,
`time_limit`, `max_simulations`) are **rejected** in the agent entries: a per-agent cap would
break the equal-budget design. `default` is the implicit value of old configs and is *not*
written into their hash view, so every run made before the key existed still resumes;
`equal_time` always contributes `budget_mode` and `time_per_move` to the hash.

`default` is also the mode that must mirror the web API, checked by
`tests/evaluation/test_api_parity.py` (agent classes, settings and budgets; `neural_mcts`
too when PyTorch and the checkpoint are present).

**Did the agents keep the budget?** Section 6 of every report has a table of the *measured* think
time per move (mean, p95, max) next to each agent's limit, and section 8 warns for every agent
whose **p95 exceeds 1.10 × the limit**. `neural_mcts` uses a *soft* limit (checked after every
simulation, plus the root's preparation and the simulation in flight), so it can overrun; this is
measured and reported, never hidden.

### Agents

| Agent | Class | `default` configuration | Deterministic in `default`? |
|---|---|---|---|
| `random` | `RandomAgent(seed)`, one seed per game | – | seeded: exactly replayable (Elo anchor = 0) |
| `rules` | `RuleBasedAgent` | – | yes |
| `alphabeta` | `AlphaBetaAgent` | default depth 2, **no time limit** | yes |
| `minimax` | `MinimaxAgent` | default depth 2, no time limit (the constructor's `max_depth`) | yes |
| `heuristic` | `HeuristicAgent` (A*) | constructor defaults, 1 s limit | **no (clock-dependent)** |
| `mcts` | `MCTSAgent` | `configs/mcts.yaml`, 1 s | **no (time- and seed-dependent)** |
| `neural_mcts` | `NeuralMCTSAgent` + `PolicyValueAdapter` | `configs/neural_mcts.yaml`: `c_puct`, soft `time_limit` 3 s; `max_simulations` optional (exact replay) | **no** |
| `tabular_q` | `TabularQAgent` | greedy (`epsilon=0`), fallback to `rules`, checkpoint from `XIANGQI_TABULAR_Q_CHECKPOINT` | yes |

**`neural_mcts` details.**

* The heavy adapter (weights) is loaded **once per worker process** (`_ADAPTER_CACHE`, like the Q-table
  cache); the agent itself is new for every game.
* torch is pinned to **1 thread per worker process** (`threads: 1`, recorded in the manifest;
  `OMP/MKL_NUM_THREADS=1` for workers) so that workers do not fight for CPUs.
* The model name and the checkpoint's SHA-256 (verified against `models/neural/manifest.json`)
  are written to the manifest (`agents.neural_mcts.params.model`, `.checkpoint_sha256`) and to the
  report. The path is machine specific and is not part of the config hash; the SHA-256 is.
* The model, path and device follow the web API: environment variables
  (`XIANGQI_NEURAL_*`, `.env`) win over `configs/neural_mcts.yaml`; `model:` / `checkpoint:` in the
  tournament entry win over both. Provisioning: `models/neural/README.md`.
* `required: false` and torch or the checkpoint missing → excluded with a warning;
  `required: true` → the run stops with the reason.

**Why `minimax` now?** It is the unpruned reference of `alphabeta`. In `default` it picks
(nearly) the same moves at the same depth, so it mostly adds a near-duplicate player (flagged by the
report through `distinct_ratio`); in `equal_time` it is much slower per depth, so it shows what
pruning buys under the same clock.

Each game builds **fresh** agents inside the worker process; only read-only heavy resources (the
Q-table, the neural adapter) are shared inside a process.

## 2. Running

### Google Colab CPU + Google Drive

Open [`notebooks/xiangqi_colab_tournament.ipynb`](notebooks/xiangqi_colab_tournament.ipynb)
in Colab and select a CPU runtime (hardware accelerator: **None**). Choose a
`SESSION_NAME`, then run the cells in order. Leave `TABULAR_Q_DRIVE_PATH` empty
to use `models/tabular_q/q_table.json.gz`, which is tracked in the repository;
set it only to override this with a checkpoint on Drive. An invalid explicit
override stops setup rather than falling back to the repository model.
Neural weights are downloaded and verified by the
existing provisioning script. Selected checkpoint-based agents are required;
missing models stop setup rather than silently reducing the tournament.

Set `BUDGET_MODE = 'equal_time'` (the initial selection) or `'default'` to select
`configs/tournament_equal_time.yaml` or `configs/tournament_default.yaml`.
For a full run the session folder is automatically `equal_time_01` or `default_01`,
respectively. Both modes save the same log/report/ZIP artifacts on Drive. The
name does not increment automatically: reuse it to resume, or change the `01`
suffix for a separate experiment. `MODE = 'full'` is the run size, distinct from
the budget mode; smoke/pilot sessions include that size in their default name.

In `equal_time`, `time_per_move` is the thinking budget **per move**, not a
deadline for the whole tournament. In `default`, each agent keeps its shipped
defaults. Neither adds a whole-run time limit.
`WORKERS` is the number of concurrent games. The notebook defaults to the CPU
count reported by the runtime, limited by process affinity when available: a
2-vCPU runtime selects 2 workers. Override it with an integer when needed; more
workers than available CPUs makes searches compete for CPU time, and the
reported count does not guarantee exclusive compute or linear speed-up.
To change workers mid-experiment, Stop the running cell, wait for the child
process to stop, then rerun the settings and run cells with the same session and
experiment configuration. Saved games are retained; unfinished games restart.
Worker changes are recorded in `manifest.json`'s `resume_history`. Clock-limited
agents may choose different moves when CPU contention changes.
Change `MODE` to `smoke`
or `pilot` and use a separate session name for a setup check or time estimate.

Each session lives in `MyDrive/XiangqiAI/tournaments/<SESSION_NAME>/`. Code and
checkpoints stay on the runtime disk; completed games, run logs, progress,
configuration and metadata are written to Drive. On normal completion or a
handled Stop, the notebook generates `summary/` (CSV, PNG, `report.md`) and writes
`<SESSION_NAME>.zip` plus its SHA-256 alongside the session folder. The ZIP copy
is verified before replacing an existing archive. If reporting fails, it still
archives the raw logs and records the report failure in `export.json`.

Rerun setup with the same session name to fetch its saved source commit and
resume its existing run. Keep checkpoint contents and experiment settings the
same; changed config/model/mode requires a new session. Do not run two runtimes
against the same session. Drive sync can lag or fail, and a reclaimed runtime
cannot execute cleanup: only successfully persisted games survive; unfinished
games are replayed. Reconnect to resume or use the final export-only cell. A
partial report is not a completed balanced tournament. Custom local plot scripts
are not automatically transferred to Colab.

Check [Colab's policies](https://research.google.com/colaboratory/faq.html): free
managed runtimes without a positive compute-unit balance restrict SSH/remote
control and “chess training”; the FAQ does not clarify whether Xiangqi tournament
evaluation falls under the latter. The notebook does not bypass these limits.

### Windows PowerShell

Pick the config; everything else is the same for both modes.

```powershell
# 1. Quick check (2 openings x 2 colors, max 40 plies): separate folder smoke_...
python -m src tournament --config configs/tournament_default.yaml --smoke --workers 5
python -m src tournament --config configs/tournament_equal_time.yaml --smoke --workers 5

# 2. Pilot: 1 game per pair at the REAL config; prints a total-time estimate
python -m src tournament --config configs/tournament_default.yaml --pilot --workers 5
python -m src tournament --config configs/tournament_equal_time.yaml --pilot --workers 5

# 3. The real tournaments (2800 games each)
python -m src tournament --config configs/tournament_default.yaml --workers 5
python -m src tournament --config configs/tournament_equal_time.yaml --workers 5

# 4. Continue after Ctrl+C, a crash or a power-off (the run knows its own config and mode)
python -m src tournament --resume results\tournaments\<run_id>

# 5. Regenerate summary\ (CSV, plots, report.md) from the raw log only
python -m src tournament-report results\tournaments\<run_id>
python -m src tournament-report results\tournaments\<run_id> --no-plots

# 6. Compare the two modes: rank and Elo of every agent in both runs
python -m src tournament-compare results\tournaments\<default_run_id> results\tournaments\<equal_time_run_id>
```

`configs/tournament.yaml` is the **legacy** config (25 openings, 6 agents, 50 games per pair,
`xiangqi-curated-v2`) kept so that run `20261112-091526_501370b5` stays reproducible and
resumable. It no longer passes the `min_games_per_pair` guard for a *new* run; resuming and
reporting its runs work unchanged.

Options of `tournament`: `--workers N|auto`, `--output-root DIR`,
`--allow-code-change` and `--retry-errors` (both only with `--resume`).

**Workers.** Do not exceed the number of *physical* cores: MCTS, the neural agent, the
heuristic agent and (in `equal_time`) every search agent are time-limited, so an over-subscribed
machine gives them less search per move and lowers their strength. A laptop with an Intel
i5-13420H has 8 physical cores (12 threads): use **`--workers 5`**. (`auto` = `cpu_count - 1`
counts threads, which is too many for this purpose.)

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
  the resume stops and prints the keys that changed. A `default` run cannot be resumed
  as `equal_time` or vice versa, and `time_per_move` cannot change.
* If the git commit changed, resume refuses unless `--allow-code-change` is given;
  the event is stored in `manifest.json` → `resume_history` and warned about in the report.
* A torn last line of `games.jsonl` (crash mid-write) is discarded with a warning;
  a corrupt line elsewhere or a duplicate `game_index` is an error.
* Games with `status="error"` are not re-run automatically; use `--retry-errors`
  (the old records are kept in `games.retried.jsonl`).
* Ctrl+C stops cleanly and prints the resume command.
* Reading is tolerant: `games.jsonl` may mix schema 1 and 2 lines (an old run resumed with this
  code), and a missing `budget_mode` means `default`.

**Progress** is printed one line per game:
`[123/2800] alphabeta vs mcts | opening=… | A=red | mcts wins (checkmate) | 87 plies | 64.2s | ETA 5h12m`.

### Time estimate

`--pilot` plays **one game per pair** (first opening, A = Red) under the real budgets, so every
pair, including the new agents and the active `budget_mode`, is in the sample, and prints the
total for `pairs × games per pair` sequentially and divided by the workers, plus the mean think
time of each agent. It is rough (one game per pair, linear speed-up assumed, it also includes
the one-off neural model load). The pilot numbers measured for this change are in the
hand-over notes of the PR; the structure of the cost is:

* **`default`:** `neural_mcts` (soft 3 s per move) and `heuristic`/`mcts` (≈1 s) dominate; the
  7 pairs with `neural_mcts` cost several times the rest.
* **`equal_time`:** every search agent thinks ≈ T = 1 s on *each* of its moves, so a game between two
  search agents lasts about `agent plies × 1 s`; `random`, `rules` and `tabular_q` are fast.
  Expect it to be of the same order as `default` or cheaper (no 3 s neural moves), but the
  `alphabeta`/`minimax` pairs are no longer nearly free.

Both runs are measured in hours, which is why the pilot exists: run it first, with the
workers you will use.

## 3. Output files

```
results/tournaments/<run_id>/       # run_id = [smoke_|pilot_]YYYYMMDD-HHMMSS_<config_hash[:8]>
├── manifest.json         provenance: git commit/branch/dirty, machine, Python, packages, command,
│                         config + hash, budget_mode/time_per_move, agent specs (incl. is_deterministic,
│                         budget_applicable, neural model + sha256 + threads), excluded agents,
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

### `games.jsonl` schema (`schema_version: 2`)

One JSON object per line (`ensure_ascii=False`, compact separators). Schema 2 = schema 1 plus
`budget_mode`; the reader accepts both (a record without `budget_mode` is `default`).

| Field | Meaning |
|---|---|
| `schema_version`, `run_id`, `config_hash` | Record version and run identity. |
| `game_index`, `game_key` | Position in the schedule (0…N-1) and stable key `A\|B\|opening_id\|a_color`. |
| `pair` | `[A, B]`; A comes first in config order. |
| `red_agent`, `black_agent`, `a_color` | Who plays which side; `a_color` is A's color. |
| `opening` | `{id, family, name, book_version, plies}`. |
| `seeds` | `{game, red, black}`, derived with SHA-256 from `master_seed` and `game_key`. |
| `budget_mode` | `default` or `equal_time` (new in schema 2). |
| `budgets` | `{red, black}`: the `SearchBudget` given to each side (or `null`: no budget, e.g. `random`/`rules`/`tabular_q` and the fixed-depth agents in `default`). |
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
* **Cost:** thinking time (mean/median/p95/max/total), average nodes, average MCTS
  simulations and the Tabular Q fallback rate, counting only moves the agent made; and the
  **think time vs. budget** table with the p95 > 1.10 × limit warning.
* **Colors and openings:** score as Red vs Black, overall first-move advantage,
  and per-opening result shares; openings where one side wins ≥ 70% of valid
  games are flagged.
* **`tournament-compare RUN_A RUN_B`:** rank and Elo (with 95% interval) of every agent in each
  run and the differences. Elo is anchored at `random` = 0 *inside each run*, so across runs it is a
  relative strength within that field; read rank changes first, and do not read overlapping intervals
  as a change.

## 4. Reproducibility

To re-run: check out the commit in `manifest.json`, use the same config and
`master_seed`, and run `tournament`. `report.md` section 9 prints the exact commands.

| Agent | Reproducible move-for-move? |
|---|---|
| `rules`, `tabular_q` (with the same checkpoint, verified by SHA-256 in the manifest) | **Yes**, in both modes. |
| `alphabeta`, `minimax` | **Yes in `default`** (fixed depth). **No in `equal_time`** (the clock decides the depth). |
| `random` | **Yes**, the seed of each game is derived from `master_seed` and the game key. |
| `heuristic` | **Statistically only.** Its time limit makes the search depend on speed and load. |
| `mcts`, `neural_mcts` | **Statistically only.** Time-limited search + a different number of simulations per run. Set `max_simulations` in the entry of the config (default mode only; it changes the config hash) for an exact replay of the search. |

A tournament that involves clock-bound agents gives similar standings, not identical game
records. Results never depend on completion order; they depend on the worker count only through
the time-limited agents. Interrupting and resuming a run does not change any game that had been
finished; with deterministic agents the final log is identical to an uninterrupted run (tested).

## 5. Limitations

* **`equal_time` is not "equal everything".** `random`, `rules` and `tabular_q` do not search, so the
  budget cannot apply to them: they play at their natural speed (much faster than T) and are *not*
  slowed down (no sleeping). Their results are not a "same-time" comparison, and the report says so.
* **Equal wall time is not equal compute.** Python alpha-beta, A*, rollout MCTS and a torch network
  turn one second into very different amounts of search, and `neural_mcts` runs on one CPU thread
  per worker. That is what `equal_time` measures, but it is not "equal FLOPs".
* **Budgets can be overrun.** `neural_mcts` uses a soft limit and the others check the clock at
  discrete points; with more workers than physical cores everything is slower. The report shows
  mean/p95/max and flags p95 > 1.10 × T. Treat flagged agents' results with care.
* **Clock-bound agents depend on the machine and the worker count** (`heuristic`, `mcts`,
  `neural_mcts`, and in `equal_time` also `alphabeta`/`minimax`). Run both modes on the same
  machine with the same `--workers`.
* **Wilson interval is an approximation** when there are draws (a draw counts as
  half a success, which is not a binomial trial). Conclusions of the 28 pairs are
  not corrected for multiple comparisons.
* **Bradley–Terry prior:** one virtual draw per pair shrinks Elo slightly towards
  0 and keeps ratings finite; Elo is only meaningful as differences between agents
  (the anchor `random` = 0).
* **Truncated games** are neither wins nor draws; `T_half` is a convention. If a
  large share of games is truncated (e.g. weak agents shuffling pieces), rankings
  are sensitive to it. Compare `T_half` and `excl_T`.
* **Tabular Q depends on its checkpoint** and mostly behaves like `rules`: in the
  smoke run only about 4% of its moves were found in the Q table, ~96% fell back
  to the rule-based agent. When two agents behave identically, colour-swapped games
  can be identical (the report flags pairs with `distinct_ratio` < 1). The same applies to
  `minimax` vs `alphabeta` in `default`.
* **Which checkpoint does the web serve?** Not verified: it is unclear whether the
  deployed web app uses the local `models/tabular_q/q_table.json.gz` or a file
  from Drive. The tournament uses `XIANGQI_TABULAR_Q_CHECKPOINT` from `.env`, else
  the local default, and records its SHA-256 in the manifest.
* **The balance of the openings is a static judgement** (`evaluate_position`, material),
  not a proof: per-opening result shares in the report show which openings are skewed. The
  18 extension lines are engine-chosen, "best-effort" theory (see `docs/OPENING_BOOK_V3.md`).
* The results are only valid for this opening set, `max_plies` and agent configuration.
* A dirty git tree is recorded (`git.diff`) and warned about, but results are then not
  tied to a clean commit.
