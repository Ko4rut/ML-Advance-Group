# Silver → Gold for Xiangqi policy imitation and offline Q-learning

Gold is **task-specific**. The source-neutral Silver JSONL remains unchanged.
Do not train against Bronze or Silver directly. A GoldPipeline takes **exact**
Silver run folders, combines unique canonical games, checks every successor and
observed action against the Xiangqi engine, and creates six split files plus a
report. Choose the CCPD **full** run and Alpha-Beta **full** run; do not include
sample or revalidation copies in the same build.

## Running locally / Colab

From `project/backend` after `pip install -e .`:

```bash
python -m scripts.build_gold \
  --silver-dirs \
    /content/xiangqi_silver_stage/ccpd_full_20261107T080132614436Z \
    /content/xiangqi_silver_stage/alphabeta_full_20261107T074509139310Z \
  --output-dir /content/xiangqi_gold_runs/gold_20261107_v1 \
  --seed 42 --train-ratio 0.8 --val-ratio 0.1 \
  --reward-mode shaped --gamma 0.99 --shaping-weight 0.15
```

On Colab, mount `/content/drive` and stage each source run's `games.jsonl`
and `transitions.jsonl` from
`/content/drive/MyDrive/XiangqiAI/data/02_silver/<run-name>/` into
`/content/xiangqi_silver_stage/<run-name>/` via `shutil.copy2` first; this
reduces Drive's per-file I/O overhead. Then run the CLI above. After verifying
`report.json` and SHA-256 checksums, copy the **new** Gold run directory to
`/content/drive/MyDrive/XiangqiAI/data/03_gold/<new-run-name>/`. Do not
overwrite an existing run and do not delete Silver input runs.

### Output layout

```
03_gold/<run-name>/
    policy_train.jsonl
    policy_val.jsonl
    policy_test.jsonl
    offline_rl_train.jsonl
    offline_rl_val.jsonl
    offline_rl_test.jsonl
    report.json
```

- `PolicySample`: state, side, chosen action, sparse legal action IDs.
- `OfflineRLSample`: state, side, action, sparse legal IDs, reward, next_state,
  next_side, sparse next legal IDs, terminated, truncated, termination_source.
- Action ID is `source_square * 90 + target_square`, in `[0, 8100)`.
- Reward definition `terminal_outcome_sparse_actor_perspective_v1`: each
  intermediate action receives 0; the **last** action in a finished game
  receives +1 if the actor wins, -1 if the actor loses, and 0 for a draw.
  A truncated or unknown game has no outcome reward. Source-side PGN win/draw
  is **not** inferred from the board. `termination_source` distinguishes a
  verified engine terminal from a recorded adjudication/resignation.
- For finished PGN games, an adjudicated result ends the dataset episode even
  when the engine still has legal moves. This is an explicitly recorded
  termination, **not** engine checkmate. For truncated games, `terminated=false`,
  `truncated=true`, and next legal actions remain available.
- **Critical for zero-sum alternating moves:** `reward` belongs to the side
  that played the action. Q values at `next_state` belong to the **opponent**;
  a negamax-style TD backup is `r - gamma * max_a' Q(next_state, a')`, not
  `r + gamma * max_a' Q(next_state, a')`. Apply terminal masking correctly.
- Splitting is stable SHA-256 by canonical **whole game**, seeded (80/10/10
  default). Actual split counts vary; `report.json` records the counts.
- Preserve provenance with `game_id`, `source_dataset`, Silver run names and
  source file SHA-256s in `report.json`. Other Silver run duplicates are skipped
  by canonical game hash. The input directory order determines which duplicate
  source is retained.
- Output is staged and only published when fully validated; existing outputs
  are never overwritten.

This is a **baseline dataset**, not a trained model and not a cure for offline
RL coverage limitations. The existing `src/training/dqn.py` is a skeleton:
Gold output alone does not implement a trainer or self-play loop.

To check the implementation, run `python -m unittest tests.data_pipeline.test_gold_pipeline -v`
from `project/backend`.

## Gold reward profiles (v1 sparse and v2 shaped)

Silver's `recorded_reward` remains source-neutral and is never modified.
Gold's `offline_rl_*.jsonl` reward is assigned by the **explicitly versioned**
reward policy. `policy_*.jsonl`, labels, legal actions, and game splits are
the same for both policies.

**Sparse v1:** `--reward-mode sparse` is the backward-compatible baseline:
only the last finished action receives +1/-1/0 from the actor's perspective.
All intermediate and truncated actions receive 0.

**Shaped v2 (default CLI):** `--reward-mode shaped` uses a bounded,
zero-sum potential `Phi(board, side) = tanh((material + pawn_progress)/300)`.
The king has **zero material weight** because winning is already represented
by the terminal outcome. The symmetric material values use the project's
existing heuristic weights: advisors 20, elephants 20, horses 40, rooks 90,
cannons 45, pawns 10. Pawn progress adds just 1 point per advanced rank
(relative to the player's travel direction).

For the action-taking player, apply

```text
r_v2 = r_terminal + lambda * (gamma * Phi(next_board, actor) - Phi(board, actor))
Phi(next_board, actor) = 0 when terminated=True
defaults: gamma=0.99, lambda=0.15
```

For truncated (nonterminal) episodes, retain the next potential and the
original truncation flag. A recorded PGN win/loss/draw ends its dataset
episode, so its terminal successor potential is zero; the recorded result is
*not* an engine checkmate label. Terminal rewards remain dominant. Fixed
"capture +X" / "check +Y" rewards are deliberately avoided: they may encourage
material chasing or loops and are not necessarily policy-preserving.

**Crucial two-player detail:** the potential uses **the actor's** perspective
on *both* boards. Because the next turn belongs to their opponent and `Phi`
is anti-symmetric, this is equivalent to
`gamma * (-Phi(next_board, opponent)) - Phi(board, actor)`.
For a negamax DQN whose Q-values are from the acting side's perspective,
bootstrap with `r - gamma * max_legal Q(next_state, a')` and mask terminal
transitions. Use the **same gamma** in Gold construction and training; using a
different discount factor invalidates the potential-shaping guarantee.
For transitions marked `truncated`, retain and handle the time-limit bootstrapping
semantics explicitly.

The report records `reward_definition` and `reward_config` so experiments
are reproducible. `lambda=0` makes shaped rewards numerically identical to
sparse rewards while retaining the v2 definition label. No claim of stronger
play is implied until a held-out evaluation against Alpha-Beta is run.

**Do not overwrite Gold v1.** Use a fresh run name (e.g.
`gold_shaped_v2_<timestamp>`) in `data/03_gold`. The updated Colab notebook
selects the new branch and generates its own fresh timestamped run.
To validate just rewards, run

```bash
python -m unittest tests.data_pipeline.test_gold_rewards -v
python -m unittest tests.data_pipeline.test_gold_pipeline -v
```

The sparse dataset is still the essential **control group** when evaluating
whether shaping accelerates learning or changes playing strength.
