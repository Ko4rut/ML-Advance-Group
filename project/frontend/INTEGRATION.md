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

1. Choose one participant per side (top=red, bottom=black). Only Random/Rule-Based are enabled AI agents, along with humans.
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
- The other five agent types are disabled until backend reports them available.
