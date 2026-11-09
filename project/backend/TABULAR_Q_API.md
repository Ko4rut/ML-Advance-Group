# Tabular Q-Learning API inference

The offline Gold v2 Q-table can now play through the existing
`/api/v1/matches` endpoints. **This is inference only.** The API does not
train or expand the Q-table, and it does not download a checkpoint from Drive
automatically.

## Configure the checkpoint

The backend needs **local filesystem access** to the model checkpoint.
Download `q_table.json.gz` from the chosen
`MyDrive/XiangqiAI/models/tabular_qlearning/<run>/` checkpoint directory
to your backend machine, or mount the Google Drive directory where the API
process runs.

From `project/backend`, in PowerShell:

```powershell
$env:XIANGQI_TABULAR_Q_CHECKPOINT="E:\\xiangqi-models\\q_table.json.gz"
fastapi dev src/api/main.py
```

On Linux/Colab with a Drive mount:

```bash
export XIANGQI_TABULAR_Q_CHECKPOINT="/content/drive/MyDrive/XiangqiAI/models/tabular_qlearning/<run>/q_table.json.gz"
uvicorn src.api.main:app --port 8000
```

The path must point to the **.json.gz file**, not the containing directory.
Restart the backend after changing the checkpoint file or environment variable.

If no checkpoint is configured, missing or invalid, `/api/v1/agents`
advertises `tabular_q` as `available: false` and attempting to create
a Tabular Q match returns HTTP 503 with a configuration/load error. Other
agents continue working.

## Endpoints

The routes follow the existing API contracts; no new frontend or route
shape is necessary.

```http
GET /api/v1/agents
```

When loaded, includes:

```json
{"id": "tabular_q", "name": "Tabular Q-Learning", "available": true}
```

Create a match:

```http
POST /api/v1/matches
Content-Type: application/json

{
  "red": {"type": "agent", "agent_id": "tabular_q"},
  "black": {"type": "human"},
  "opening_mode": "none"
}
```

Advance the Tabular Q side using the normal step route:

```http
POST /api/v1/matches/{match_id}/step
Content-Type: application/json

{"expected_ply": 0}
```

## Playing behavior

- **epsilon is hard-set to 0.0** by `TabularQService.make_agent`. No
  epsilon-greedy exploration occurs during API matches.
- For an exact known board+side with legal TRAIN-supported actions, choose
  the highest-Q action greedily (deterministic tie-break by action ID).
- For an unseen board or a position with no supported legal action, use
  `RuleBasedAgent` instead of random fallback. This is a heuristic move,
  **not** a learned Q estimate.
- Uses the existing engine to guarantee legal moves.
- The initial opening-book mode defaults to `none` when Tabular Q participates,
  so the book does not silently replace its early decisions. Clients can still
  explicitly request `opening_mode: "book"`.
- Q-table is read once per API process, shared read-only among matches.
  Checkpoint files are never modified through the API.

The model is an exact-state sparse Q-table, not a generalizing neural
network. A fallback is required because nearly every possible Xiangqi
position cannot be enumerated in memory. Measured held-out test state
coverage from the initial Gold v2 baseline is about 16%, not a win rate.

## Verification

Run from `project/backend` with the API/test extras installed:

```bash
python -m pip install -e ".[api,test]"
python -m unittest tests.api.test_tabular_q_api -v
python -m unittest tests.api.test_matches -v
```
