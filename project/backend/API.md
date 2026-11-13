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
  services/mcts.py            # MCTS config loading and per-move search budget
  services/neural_mcts.py     # Optional validated neural model, readiness and budget
  services/opening_book.py    # Per-match curated opening state and AI handoff
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/agents` | List implemented and planned agents |
| GET | `/api/v1/openings` | List curated opening ids/families/names |
| POST | `/api/v1/matches` | Create game; automatically sample opening book when an AI participates unless disabled |
| GET | `/api/v1/matches/{id}` | Board, current turn, full legal moves, outcome |
| GET | `/api/v1/matches/{id}/legal-moves?source=54` | Optionally filter moves by source |
| POST | `/api/v1/matches/{id}/moves` | Apply **human** move |
| POST | `/api/v1/matches/{id}/step` | Let current **AI** play exactly one move |
| POST | `/api/v1/matches/{id}/reset` | Reset same match to initial position |

### Create Player vs Random

```json
{"red":{"type":"human"},"black":{"type":"agent","agent_id":"random"}}
```

Use `"agent_id":"astar"` to create a match against the A* agent. It searches for a
shortest predicted checkmate against a deterministic one-ply opponent model, so it
does not guarantee a win against a human who responds differently.

### Play against MCTS

MCTS uses the same match API as other agents; no separate endpoint is required.
Choose `mcts` as a participant and call `POST /api/v1/matches/{id}/step`
only when it is that agent's turn:

```json
{
  "red": {"type": "human"},
  "black": {"type": "agent", "agent_id": "mcts"},
  "opening_mode": "none"
}
```

The MCTS implementation uses UCT with a rollout/static position evaluator.
Settings are loaded from `configs/mcts.yaml` relative to the backend, not the
current working directory. `time_limit` (default: 1 second) limits search per
AI move; optional `max_simulations` can limit simulations as well. The limit
is checked between simulations, so an individual simulation may run over the
time limit. A match's opening-book moves still take precedence over search.
Install with `pip install -e ".[api,test]"` to include the YAML reader.

### Play against Neural MCTS

Use `"agent_id":"neural"` with the same match/move/step/reset routes, for either color.
Provision the real checkpoint and optional CPU runtime using [models/neural/README.md](models/neural/README.md).
Default model: `distill-v4-large`; alternative: `XIANGQI_NEURAL_MODEL=distill-v4`.
`configs/neural_mcts.yaml` supplies `c_puct`, soft `time_limit: 3.0`, and optional
`max_simulations`. No request-supplied model paths or new endpoints are needed.

`GET /agents` loads and probes the configured model on first use. It reports neural
`available=true` only after success; optional `unavailable_reason` explains failures.
Startup and classic match creation do not load weights. The adapter is shared across
matches and moves. Neural requests fail with HTTP 503 on bad configuration, missing
runtime/checkpoints or inference failure; the engine state stays unchanged. There is
no fallback. Restart after provisioning/config changes; set `XIANGQI_NEURAL_ENABLED=false`
and restart to disable the feature. Neural matches default to no opening book.

### Configure Opening Book

When at least one participant is AI, the API randomly selects **one opening line per match** by default, except matches containing `neural` or `tabular_q`, which default to `none`. Human-versus-human games also default to `none`. Override with `opening_mode`.

```json
{
  "red": {"type": "human"},
  "black": {"type": "agent", "agent_id": "alphabeta"},
  "opening_mode": "book",
  "opening_plies": 6,
  "opening_seed": 42
}
```

- `opening_mode`: `book` or `none`. Omit to apply the automatic default.
- `opening_plies`: 1–6 scripted plies; after these, normal agent search starts.
- `opening_seed`: optional reproducible nonnegative integer; omit for a fresh randomized line.
- `GET /api/v1/openings` lists the five families and twelve curated variants.
- The API **never moves human pieces automatically**. Human moves that differ from the book deactivate the line.
- The opening book is shared with the batch collector through `src/training/opening_book.py`, so chess rules and opening definitions are not duplicated.

### Move a pawn (first red move)

```json
{"source":54,"target":45,"expected_ply":0}
```

### Ask AI to reply

```json
{"expected_ply":1}
```

Every snapshot also includes `opening_mode` and `opening` metadata (`id`, `family`, `name`, `book_version`, `seed`, `planned_plies`, `completed_plies`, `active`). The book move is chosen at each eligible **AI** `step`, not applied to the board when creating a game. When a human plays a different move, the book is disabled for that match and the agent uses normal search. Once the prefix ends, the agent takes over. A seeded reset replays the same opening; an unseeded reset samples a fresh one.

Every snapshot includes `match_id`, `red`, `black`, a 90-integer `board`, `side_to_move`, `ply`, `status`, `winner`, `reason`, `last_move`, and **`legal_moves`**. Store `match_id` on the client; use the most recent `ply` on each move/step. Do **not** request `step` for a human turn. `GET /legal-moves` is optional because each snapshot already includes legal moves.

**Coordinates:** indices 0–89, row-major `row*9+col`; **row 0 is Black's back rank and row 9 is Red's**. Signed piece IDs: `1..7` red; `-1..-7` black; `0` empty. Red moves first. If the UI visually displays red on top, translate each index `display_row -> engine_row = 9 - display_row` on both board and moves.

**Errors:** 404 unknown match; 409 stale `expected_ply`, wrong type of turn, or game already finished; 422 invalid body, invalid move or unknown agent ID; 503 agent unavailable, including model configuration/load/inference errors. Error body is `{"detail":"..."}`. No operation mutates the board when a request is rejected.

## Boundaries

- MVP matches are stored in **process memory**; restarting clears them. Run **one Uvicorn worker**. Do not use multiple workers until storage/locking has been centralized.
- No authentication, persistence, rate limiting, WebSocket, auto-play loop, or statistics database yet. Expose only on a trusted development environment.
- Random, Rule-Based, A*, Minimax, Alpha-Beta, and MCTS are enabled. Tabular Q and Neural MCTS depend on their configured model readiness; DQN remains unavailable.
- Frontend can instantly show legal move dots from a snapshot; backend remains the source of truth for all moves.

## Tests

With the optional test dependencies installed, from the repository root:

```powershell
python -m unittest project.backend.tests.api.test_matches -v
```

Or from `project/backend`: `python -m unittest discover -s tests -v` (requires the repository root on `PYTHONPATH` for current legacy tests). To check the MCTS endpoint specifically: `python -m unittest tests.api.test_mcts_api -v`.
