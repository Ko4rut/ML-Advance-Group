# Xiangqi API — MVP

FastAPI exposes the **existing** Python `XiangqiEnv` and implemented agents. The legal-move engine is not duplicated.

## Setup (PowerShell)

From the repository root:

```powershell
cd project/backend
python -m pip install -e ".[api,test]"
python -m uvicorn src.api.main:app --reload --port 8000
```

Visit [Swagger UI](http://localhost:8000/docs) and test the endpoints under `/api/v1`. Frontend development origin `http://localhost:3000` is allowed by default. Override via `XIANGQI_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000`.

## Layout

```text
src/api/
  main.py                     # FastAPI factory, CORS, domain error handler
  models/match.py             # Pydantic request/response contracts
  routes/matches.py           # Versioned HTTP routes
  services/matches.py         # Game sessions, XiangqiEnv, agent registry
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/agents` | Seven agents; `available=true` for Random, Rule-Based, Minimax, and Alpha-Beta (depth 2 by default) |
| POST | `/api/v1/matches` | Create game with red and black participants |
| GET | `/api/v1/matches/{id}` | Board, current turn, full legal moves, outcome |
| GET | `/api/v1/matches/{id}/legal-moves?source=54` | Optionally filter moves by source |
| POST | `/api/v1/matches/{id}/moves` | Apply **human** move |
| POST | `/api/v1/matches/{id}/step` | Let current **AI** play exactly one move |
| POST | `/api/v1/matches/{id}/reset` | Reset same match to initial position |

### Create Player vs Random

```json
{"red":{"type":"human"},"black":{"type":"agent","agent_id":"random"}}
```

### Move a pawn (first red move)

```json
{"source":54,"target":45,"expected_ply":0}
```

### Ask AI to reply

```json
{"expected_ply":1}
```

Every snapshot includes `match_id`, `red`, `black`, a 90-integer `board`, `side_to_move`, `ply`, `status`, `winner`, `reason`, `last_move`, and **`legal_moves`**. Store `match_id` on the client; use the most recent `ply` on each move/step. Do **not** request `step` for a human turn. `GET /legal-moves` is optional because each snapshot already includes legal moves.

**Coordinates:** indices 0–89, row-major `row*9+col`; **row 0 is Black's back rank and row 9 is Red's**. Signed piece IDs: `1..7` red; `-1..-7` black; `0` empty. Red moves first. If the UI visually displays red on top, translate each index `display_row -> engine_row = 9 - display_row` on both board and moves.

**Errors:** 404 unknown match; 409 stale `expected_ply`, wrong type of turn, or game already finished; 422 invalid body, invalid move or unknown agent ID; 503 agent not implemented. Error body is `{"detail":"..."}`. No operation mutates the board when a request is rejected.

## Boundaries

- MVP matches are stored in **process memory**; restarting clears them. Run **one Uvicorn worker**. Do not use multiple workers until storage/locking has been centralized.
- No authentication, persistence, rate limiting, WebSocket, auto-play loop, or statistics database yet. Expose only on a trusted development environment.
- Random, Rule-Based, Minimax, and Alpha-Beta are enabled. Remaining agents are unavailable until implemented.
- Frontend can instantly show legal move dots from a snapshot; backend remains the source of truth for all moves.

## Tests

With the optional test dependencies installed, from the repository root:

```powershell
python -m unittest project.backend.tests.api.test_matches -v
```

Or from `project/backend`: `python -m unittest discover -s tests -v` (requires the repository root on `PYTHONPATH` for current legacy tests).