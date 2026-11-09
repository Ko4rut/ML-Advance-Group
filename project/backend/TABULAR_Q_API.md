# Tabular Q-Learning API inference

The offline Gold v2 Q-table can now play through the existing
`/api/v1/matches` endpoints. **This is inference only.** The API does not
train or expand the Q-table. If a Google Drive file ID is configured, the
service automatically downloads the checkpoint on the first readiness check.

## Download or import the model with one command

The standalone CLI uses the backend's existing Drive download implementation,
validates the Tabular Q checkpoint, and installs it into
`project/backend/models/tabular_q/q_table.json.gz` by default. It does **not**
train the model. You only need to run it once per model version.

**Easy option: download manually in the browser** from
[the trained Q-table](https://drive.google.com/file/d/1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c/view),
then from the `project/backend` PowerShell directory:

```powershell
python -m scripts.download_tabular_q_model --from-file "$env:USERPROFILE\Downloads\q_table.json.gz"
```

This mode requires **no Google CLI, ADC or API credentials**. The script checks
the model structure and prints the path to configure in FastAPI.

**Automatic private Drive download** (requires Google ADC configured as
described below):

```powershell
python -m scripts.download_tabular_q_model
```

The default file ID is the existing Gold v2 Q-table above. To choose another
model version, use `--drive-file-id OTHER_ID` or set a different `--output`.
Existing model files are protected from accidental overwrite. Explicit
`--replace` validates the new model first, then atomically replaces the old
file. Keep the API stopped during a model replacement and restart it afterward.

**Use the imported local checkpoint for API inference:**

```powershell
Remove-Item Env:XIANGQI_TABULAR_Q_DRIVE_FILE_ID -ErrorAction SilentlyContinue
$env:XIANGQI_TABULAR_Q_CHECKPOINT=(Resolve-Path "./models/tabular_q/q_table.json.gz").Path
fastapi dev src/api/main.py
```

Then check `http://127.0.0.1:8000/api/v1/models/tabular-q/status`; its
`available` field should become `true`. Refresh the frontend so the Tabular
Q choice becomes enabled.

## Configure the checkpoint

The service supports either a locally accessible checkpoint or a private
Google Drive file ID. For Drive mode, the API host authenticates with Google
Application Default Credentials (ADC) and caches the downloaded file locally.
Gold/Drive data is never modified by inference.

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
The local path configuration takes precedence over Drive when both are set.

### Google Drive mode (no manual model download)

Install the optional Drive dependencies:

```bash
python -m pip install -e ".[api,drive,test]"
```

The existing trained checkpoint is:
- [Q-table on Drive](https://drive.google.com/file/d/1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c/view)
- Drive file ID: `1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c`

For local development, authenticate once using Google Cloud SDK
(`gcloud auth application-default login --scopes=https://www.googleapis.com/auth/drive.readonly,https://www.googleapis.com/auth/cloud-platform`).
The signed-in Google identity must have read permission on the file, and the
Google Cloud project must be authorized to use the Drive API.

```powershell
Remove-Item Env:XIANGQI_TABULAR_Q_CHECKPOINT -ErrorAction SilentlyContinue
$env:XIANGQI_TABULAR_Q_DRIVE_FILE_ID="1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c"
$env:XIANGQI_TABULAR_Q_CACHE_DIR="E:\\xiangqi-models\\cache"
fastapi dev src/api/main.py
```

On Colab, you can instead use the **local Drive mount** config:
`XIANGQI_TABULAR_Q_CHECKPOINT=/content/drive/MyDrive/XiangqiAI/models/tabular_qlearning/<run>/q_table.json.gz`.
Do not use Drive API mode unless the API process has valid ADC with Drive read access.

The cache is a single file, keyed by Drive file ID. On first access
(`GET /api/v1/agents` or creating the first Tabular Q match), the service
downloads it and loads it into memory. Further matches reuse that shared table.
If you replace the checkpoint under the **same Drive file ID**, delete its
cache file and restart the API process to reload it. For a new immutable model
run, point the configuration to the new file ID.

Restart the backend after changing any checkpoint configuration. If the Drive
API is inaccessible, invalid credentials are supplied, or the file is corrupt,
Tabular Q is advertised as unavailable; other agents continue operating.

If no checkpoint is configured, missing or invalid, `/api/v1/agents`
advertises `tabular_q` as `available: false` and attempting to create
a Tabular Q match returns HTTP 503 with a configuration/load error. Other
agents continue working.

## Endpoints

The existing match routes remain compatible with the frontend. A separate,
read-only model status endpoint is now available:

```http
GET /api/v1/models/tabular-q/status
```

When the trained Gold v2 checkpoint is loaded, the response reports:

```json
{
  "agent_id": "tabular_q",
  "available": true,
  "source": "drive",
  "model_version": "tabular_qlearning_exact_board_v1",
  "known_states": 29101,
  "known_state_action_pairs": 29831,
  "epsilon": 0.0,
  "unknown_state_fallback": "rule_based"
}
```

The values above describe the initial checkpoint; future checkpoints will
report their own table sizes. A missing checkpoint returns `available: false`,
zero counts and `model_version: null` with HTTP 200. This endpoint does not
expose credentials, Drive IDs or filesystem paths. It reads the **same loaded
service** used for creating live matches and can trigger the first Drive
download.

The routes follow the existing API contracts; no new frontend or match request
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

Run from `project/backend` with the API/Drive/test extras installed:

```bash
python -m pip install -e ".[api,drive,test]"
python -m unittest tests.api.test_download_tabular_q_model -v
python -m unittest tests.api.test_tabular_q_status_api -v
python -m unittest tests.api.test_tabular_q_drive -v
python -m unittest tests.api.test_tabular_q_api -v
python -m unittest tests.api.test_matches -v
```
