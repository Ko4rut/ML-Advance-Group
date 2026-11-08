# Tabular Q-Learning baseline — Xiangqi Gold v2

This is a **real offline tabular Q-Learning baseline**, not a DQN and not an
AlphaZero trainer. It follows the backend's `src/training`, `src/agents`,
`src/data/gold`, `scripts` and `tests` conventions.

## Inputs

Use the **existing** Gold folder:

```
/content/drive/MyDrive/XiangqiAI/data/03_gold/reward_v2_shaped/
  offline_rl_train.jsonl
  offline_rl_val.jsonl
  offline_rl_test.jsonl
  report.json
```

41,084 transitions / 509 games in all splits; **only train** updates Q.
The pipeline checks Gold reward label compatibility, the reported train
SHA-256, and that Q `gamma` matches Gold reward `gamma` (0.99).

## Train

Run from `project/backend` after `python -m pip install -e .`:

```bash
python -m scripts.train_tabular_qlearning \
  --gold-dir /path/to/reward_v2_shaped \
  --output-dir /path/to/model_runs/tabular_gold_v2_seed42 \
  --alpha 0.2 --gamma 0.99 --epochs 12 --seed 42
```

Checkpoint outputs in a **new** model folder:
- `q_table.json.gz`: safe, non-pickle, gzip-compressed sparse Q-table.
- `report.json`: seed, config, Gold input SHA, per-epoch TD error, and split
  coverage and policy agreement metrics.

The model directory must **not exist** before training. It is staged then
published on success. Silver and Gold are read-only. The companion Colab
notebook stages Gold locally, trains locally and publishes a timestamped
version under `XiangqiAI/models/tabular_qlearning/` on Drive.

## Algorithm

`state_key` is **exactly** the 90-square board and side to move. No lossy
state normalization, no material-only reduction. This preserves exactness but
means unseen positions do not generalize; `TabularQAgent` chooses a legal
TRAIN-supported best-Q action when available and **random legal fallback**
for a state not present in train.

```text
Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)]
target = r                                if terminated
target = r - gamma * max_supported Q(s',a') otherwise
```

The minus sign is essential: on the next turn, `Q(s',a')` is from the
**opponent's perspective**. Unsupported next-state actions are excluded
from max. When **no** supported continuation exists, skip the TD update
and count it instead of inventing a zero value.

At a truncated final transition, the continuation is unknown, not a draw;
the default is to skip that final TD update. Gold's `reward` is used
without ad hoc scaling (potential reward shaping v2). Shuffle whole games
deterministically and replay each in reverse order to propagate terminal
outcomes efficiently.

Gold state keys do not include game history, no-capture count or repetition
rights; this is a partial-state abstraction for the baseline.

## Evaluation: correct interpretation

`seen_state_fraction` and `seen_state_action_fraction` report how much
of each split is covered by the train Q-table. `behavior_top1_supported`
reports recorded-action agreement **conditional on having observed that
state-action pair in train**.

These numbers are **not** DQN accuracy, not generalization accuracy over all
moves, and **not** an estimated chess win-rate. For playing-strength
evaluation, load `TabularQAgent.from_checkpoint(...)` and run independent
matchups as **both colors** against `RandomAgent` and `AlphaBetaAgent`.
Also record `last_action_supported` to distinguish model choices from
random fallbacks.

## Verified baseline on Gold v2

Local smoke run with the existing Gold v2 files:
- 34,176 train, 3,708 validation, 3,200 test transitions.
- Config: `alpha=0.2, gamma=0.99, epochs=12, seed=42`.
- Q-table: **29,101 states**, **29,831 state-action pairs**.
- Last epoch TD RMSE before update: **0.044707**.
- Validation known-state coverage: **21.06%**.
- Test known-state coverage: **16.03%**.
- Test known state-action coverage: **14.47%**.
- Test action agreement conditional on known state-action: **52.70%**.

This coverage shows why tabular Q is a baseline, not the planned final Xiangqi
agent. It does not prove playing strength.

Run unit tests from `project/backend`:

```bash
python -m unittest tests.training.test_tabular_qlearning -v
```
