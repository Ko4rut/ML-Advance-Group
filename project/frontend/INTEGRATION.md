# Xiangqi Arena — Frontend ↔ FastAPI

This integration branch connects the Next.js arena to the existing API in `project/backend/src/api`.

## Local setup

Backend terminal:

```powershell
cd project/backend
python -m pip install -e ".[api]"
fastapi dev src/api/main.py
```

Alternatively: `python -m uvicorn src.api.main:app --reload --port 8000`.
Swagger: http://localhost:8000/docs — health: http://localhost:8000/api/v1/health.

Frontend terminal:

```powershell
cd project/frontend
npm ci
npm run dev
```

The client defaults to `http://localhost:8000/api/v1`.
To override, set `NEXT_PUBLIC_XIANGQI_API_URL` in `project/frontend/.env.local`, restart Next.js.
The backend CORS middleware permits `http://localhost:3000` and `http://127.0.0.1:3000`.

## Play flow

1. Choose one participant per side (top=red, bottom=black). The frontend shows each known agent and enables it only when `/api/v1/agents` reports `available=true`. Tabular Q appears as its own option, separate from DQN.
2. Click **Tạo trận đấu** to `POST /api/v1/matches`. The returned snapshot is the single source of board state.
3. On a human turn, click a legal source piece. The frontend filters snapshot `legal_moves`, showing small dots on empty destinations and thin circles on capturable pieces.
4. Click a highlighted destination. Client `POST /matches/{id}/moves` with `expected_ply`; the backend uses XiangqiEnv to validate and apply.
5. On an AI turn, **Chạy AI** starts repeated `POST /step` calls, stopping on a human turn or game end. You can pause between moves. After a human move an AI reply is triggered automatically.
6. **Đặt lại** calls backend `POST /reset`. Statistics represent only the current match (not historical aggregates).

## Board orientation

UI red on top, black below. Engine row 0 is black, row 9 is red.
Every engine index is converted by `displayIndex = (9 - engineRow) * 9 + engineCol`. The same transform converts display squares back for POST moves.
Do **not** modify the engine's board ordering or its legal move indices.

## Known limitations

- Match state is held in the backend process memory; restarting server invalidates existing match IDs.
- No WebSocket, server-run automatic tournament, database, or persisted statistics yet.
- AI vs AI autoplay uses client-side sequential steps with a short delay. Closing the tab stops autoplay.
- If API is offline, a connection warning appears. CORS and the HTTP address must match your environment.
- Agent availability is fetched once on page load. After adding or changing a model checkpoint, restart the backend and reload the frontend page to refresh the available agents.

## Tabular Q-Learning — model loading from Google Drive

**Run the backend** to fetch/load the trained Q-table, not the Colab training
notebook. The backend's `src/api/services/tabular_q.py` calls
`tabular_q_checkpoint.py` when `GET /api/v1/agents` checks availability
(or the first Tabular Q match is created).

The existing trained model on Drive has file ID
`1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c`.
For automatic private Drive retrieval, the backend computer must have
**Google Application Default Credentials (ADC)** with read permission to the
model and access to the Google Drive API. In a PowerShell terminal:

```powershell
cd project/backend
python -m pip install -e ".[api,drive]"
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/drive.readonly,https://www.googleapis.com/auth/cloud-platform
Remove-Item Env:XIANGQI_TABULAR_Q_CHECKPOINT -ErrorAction SilentlyContinue
$env:XIANGQI_TABULAR_Q_DRIVE_FILE_ID="1pU8pcxCwWLCaKj44OT8ZvWOy1Cd8AA9c"
fastapi dev src/api/main.py
```

Then visit `http://localhost:8000/api/v1/agents` and verify
`tabular_q` has `available: true`. On the first request, the backend
downloads the checkpoint to its local cache. To **avoid Google auth**,
download the Q-table manually and point
`XIANGQI_TABULAR_Q_CHECKPOINT` at its absolute file path instead.
The local file takes precedence over Drive mode. Both modes are
inference-only, never modify Gold or the model.

**Run the frontend** in a separate terminal:

```powershell
cd project/frontend
npm ci
npm run dev
```

Open `http://localhost:3000`, select **Tabular Q**, then create a match.
The Arena already calls the standard `POST /api/v1/matches` and
`POST /api/v1/matches/{id}/step`; no separate Q-Learning endpoint is needed.
Tabular Q plays with **epsilon=0**, choosing a learned greedy action when
available, or a deterministic Rule-Based fallback on unseen positions.
It is **not** the DQN model.

For API details, see `project/backend/TABULAR_Q_API.md`.
