# Phase 5 — Neural MCTS application integration

Date: 2026-11-13 (Asia/Ho_Chi_Minh). **Phase 5: PASS for application integration.**
This is a development application gate, not a production security approval or a playing-strength claim.

## A. Context recovery

- Branch: `agent/mcts_nn`. HEAD before/after: `eed4887841763a02f645b1cbf85a9888227894e4`.
  The starting working tree was clean. No commit, push, reset, stash or discard was performed.
- Read the supplied `/home/thien408/Downloads/NEURAL_MCTS_MASTER_PLAN.md` completely, the
  implementation breakdown, POC report, encoding contract, model card, search lifecycle,
  implementation, API/evaluation registries, match workflow, configuration and test conventions.
  The user's current approval authorizes Phase 5; the plan's original Phase 0 startup text is historical.
- Phases 1–4 are committed: `868d8c5` qualification, `ab33de0` encoding, `4f5c4db` pretrained
  loader, `eed4887` neural search/agents/tests. Phase 4 is present, not a missing local handoff.
- Re-ran the Phase 4 fake/classic/real search gate: **51 tests passed, 0 skipped**
  ([log](evidence/phase5/phase4-core-baseline.log)). Full Phase 5 validation includes all earlier tests.
- Both ignored real checkpoint files were already available. The historical `/tmp` runtime was gone;
  restored declared API dependencies and CPU torch into `/tmp/neural-mcts-phase5/`, without changing
  the shared Python environment or production dependency declarations. No weights were downloaded.

## B. Implemented architecture and behavior

```text
Existing Next.js selector -> GET /api/v1/agents
  -> lazy NeuralMCTSService: settings -> Phase 3 manifest/hash/strict loader
  -> one readiness forward -> available or explicit unavailable_reason

Create match: agent_id="neural" -> MatchService -> fresh NeuralMCTSAgent
  -> shared validated PolicyValueAdapter (one per application service)

Human click -> /moves -> authoritative XiangqiEnv state
AI /step -> neural service SearchBudget -> Phase 4 NeuralMCTSSearch / PUCT
  -> real pretrained P/V -> legal Move -> XiangqiEnv.step -> updated snapshot
```

### Backend / API

- `NeuralMCTSService` lazily loads and probes the model once under a lock. Successful adapters
  and failed readiness attempts are cached. Every match participant receives a fresh search agent;
  all share the adapter. Reset and additional moves/matches do not reload weights.
- Application startup, health checks, and classic match creation do not load torch/weights or download
  anything. The first catalog request or neural match performs readiness validation. A broken optional
  neural installation leaves the rest of the application usable.
- API catalog id remains **`neural`**. Added the optional response field `unavailable_reason`
  (null when ready). Existing match requests/routes/snapshot contracts remain compatible.
- Neural dispatch precedes `isinstance(MCTSAgent)`, because the neural agent inherits that class.
  Thus neural moves receive their own 3-second budget, while classic MCTS keeps its existing config.
- Server-side invalid configuration, missing runtime/checkpoint, bad integrity, failed readiness or
  inference produce understandable catalog reasons and HTTP **503** for neural operations. An
  inference failure disables further neural search until restart. No fallback or board mutation follows.
  Unknown agent/body/move remains 422; stale/wrong turn/finished remains 409.
- Human-vs-neural in either color, neural-vs-neural, and neural-vs-classic use existing match routes.
- Neural matches default to `opening_mode=none`, ensuring the first AI action invokes search.
  Explicit opening books remain supported and hand off to the model after the prefix. Classic
  opening behavior is unchanged. Engine terminal outcomes stay authoritative.
- The application registry is integrated. The separate offline tournament registry/config is unchanged;
  no Phase 6 tournament, strength comparison, parameter tuning or optimization was started.

### Configuration

[configs/neural_mcts.yaml](../../configs/neural_mcts.yaml): enabled by default, `distill-v4-large`,
CPU, `c_puct: 1.5`, soft `time_limit: 3.0`, optional `max_simulations` and seed. Either limit may
stop search. Model fields reuse Phase 3 `NeuralModelSettings`, with `XIANGQI_NEURAL_*` environment
values overriding YAML; file paths remain anchored to backend root. `distill-v4` uses the identical
loader/search path. The pinned manifest's size/SHA-256 validation is always required; YAML/env SHA
is an additional assertion, never an integrity bypass. Invalid numeric values/unknown settings fail.

Rollback: **`XIANGQI_NEURAL_ENABLED=false`**, then restart the API. Changes to configuration,
checkpoints or dependencies require a backend restart and browser reload; there is no hot reload.

### Frontend

- Reused the existing Neural MCTS option and selection/match components.
- Availability enables the option; unavailable options show a status label, and a neural failure reason
  appears beneath the selector and in its tooltip. Existing HTTP errors remain visible below the board.
- Fixed two lint errors in the existing match hook while retaining its workflow: derive catalog error
  text directly, and end AI autoplay on the HTTP response that hands the turn to a human.
- Added Playwright tests with fake catalog/error responses and actual browser clicks through the real
  backend. No frontend redesign or local chess rules were introduced.

## C. Reproduce setup and verification

From repository root, with a suitable Python virtual environment activated:

```sh
cd project/backend
python -m pip install -e '.[api,test]'
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0'
# Explicit provisioning only; do not add these commands to application startup.
python scripts/fetch_neural_checkpoint.py --model distill-v4-large
python scripts/fetch_neural_checkpoint.py --model distill-v4
# Optional .env entries: XIANGQI_NEURAL_MODEL, _MODEL_PATH, _MODEL_SHA256,
# _DEVICE, _THREADS, _ENABLED; see .env.example and models/neural/README.md.
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd project/frontend
npm ci
npx playwright install chromium
npm run lint
npm run build
npm run start -- --hostname 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000`, select Neural MCTS for either participant, create a match,
make legal moves using the board markers. If AI is Red, press **Chạy AI** for its first turn;
a human move triggers the next AI reply automatically. Both AI participants can autoplay.

API example:

```sh
curl -s http://127.0.0.1:8000/api/v1/agents
curl -s -H 'Content-Type: application/json' http://127.0.0.1:8000/api/v1/matches \
  -d '{"red":{"type":"human"},"black":{"type":"agent","agent_id":"neural"},"opening_mode":"none"}'
# POST /matches/<match_id>/moves {"source":54,"target":45,"expected_ply":0}
# POST /matches/<match_id>/step {"expected_ply":1}
```

Validation commands (from `project/backend`):

```sh
export PYTHONPATH="$(git rev-parse --show-toplevel):$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest tests.api.test_neural_mcts_api tests.api.test_mcts_api -v
XIANGQI_NEURAL_REQUIRE_REAL=1 python -m unittest tests.api.test_neural_mcts_real -v
XIANGQI_NEURAL_REQUIRE_REAL=1 python -m unittest discover -s tests -v
# A standard install without torch runs the same suite with explicit real-model skips.
python experiments/neural_mcts/phase5/application_smoke.py \
  --model distill-v4-large --mode game --output /tmp/neural-api-game.json
python experiments/neural_mcts/phase5/application_smoke.py \
  --model distill-v4-large --mode budget --output /tmp/neural-api-large-budget.json
python experiments/neural_mcts/phase5/application_smoke.py \
  --model distill-v4 --mode budget --output /tmp/neural-api-v4-budget.json
```

Browser tests (from `project/frontend`, servers already running):

```sh
npm run test:e2e  # 3 fixture tests; 2 real-backend cases skip explicitly
XIANGQI_NEURAL_REQUIRE_REAL=1 npm run test:e2e  # all 5 cases required
```

Use the default backend model to reproduce the browser run; both variants are independently exercised
by the real API tests. Override `XIANGQI_FRONTEND_URL` for a different frontend address. Set
`NEXT_PUBLIC_XIANGQI_API_URL` **before building** for a non-default API address.

### Actual execution environment and results

Python 3.14.9, torch 2.11.0+cpu, FastAPI 0.162.6, Starlette 1.10.0, httpx 0.28.1,
Next 16.4.0, Node 26.13.0, npm 12.2.2, Playwright 1.65.0/Chromium.
See [environment.json](evidence/phase5/environment.json) for package/checkpoint identities.
Session commands used the repository `.venv/bin/python` with temporary `api-deps` / `torch-deps`
prepended to `PYTHONPATH`; standard checks omitted `torch-deps`.

| Executed check | Observed result | Evidence |
| --- | --- | --- |
| Phase 4 prerequisite: classic/fake/real neural search | 51 passed, 0 skipped, 32.693 s | [log](evidence/phase5/phase4-core-baseline.log) |
| New isolated neural API tests + classic MCTS API | 25 passed, 1.578 s | [log](evidence/phase5/api-isolated.log) |
| Initial real-checkpoint HTTP tests | 4 passed, both models, 3.622 s | [log](evidence/phase5/api-real.log) |
| Final full backend, real checkpoints REQUIRED | **471 passed, 0 skipped, 101.018 s**; includes 6 new real API cases (2 added corruption checks) | [log](evidence/phase5/backend-validation.log) |
| Full backend without torch | **431 run, 408 passed, 23 explicit skips, 0 failures**, 63.553 s | [log](evidence/phase5/backend-standard.log) |
| Real API complete game: large neural Red vs classic Black, 100 simulations each | **47 plies, engine checkmate, Red wins; all moves legal; reset and post-terminal 409 pass**; one model load, 2,290 network calls | [JSON](evidence/phase5/game-distill-v4-large.json) |
| Six 3-second API moves, each model | Legal moves; network invoked; one load per service | [large](evidence/phase5/budget-distill-v4-large.json), [v4](evidence/phase5/budget-distill-v4.json) |
| Frontend lint | PASS, exit 0 | [log](evidence/phase5/frontend-lint.log) |
| Production frontend build | PASS on retry, exit 0; existing security notice described below | [log](evidence/phase5/frontend-build.log) |
| Playwright against production frontend and real backend | **5 passed, 0 skipped**, 14.4 s; real Red/Black play, human move, legal replies, turn handoff and reset; unavailable/create/connection errors | [log](evidence/phase5/frontend-e2e.log) |
| `git diff --check`; checkpoint tracking check | PASS; both `.pt` files ignored and no tracked `.pt`/`.pth` | final review |

The first sandboxed API suite hung in an unchanged minimal FastAPI TestClient request. The same
minimal request succeeded outside the sandbox, and all reported completed API/regression runs used
that environment. [sandbox-testclient-hang.log](evidence/phase5/sandbox-testclient-hang.log) is an
incomplete failed environment attempt, not a passing baseline. Initial dependency installs similarly
needed network-enabled execution. No test assertion was relaxed to bypass these restrictions.

## D. Performance and reuse

Intel Core i5-13420H, 8 torch threads, batch 1. Six consecutive API moves from the starting position:

| Model | HTTP move median / range | Simulations median / range | Cold catalog readiness |
| --- | --- | --- | --- |
| distill-v4-large | 3.033 s / 3.020–3.040 s | 211 / 112–222 | 1.062 s |
| distill-v4 | 3.030 s / 3.023–3.190 s | 281.5 / 26–438 | 0.856 s |

These are small application smoke measurements on a shared desktop, with other validation work
running concurrently. They are not isolated/fair benchmarks or evidence of comparative strength.
HTTP time includes engine/root prework and response work. The search checks its deadline between
simulations; root prework, an in-flight forward, CPU contention and request queueing can exceed
3 seconds. There is **no strict real-time guarantee**.

The complete-game smoke's 100-simulation neural moves had 1.443 s median / 1.672 s maximum HTTP
latency (excluding the final immediate-win shortcut). Readiness and weights loaded exactly once;
additional moves/participants/reset reuse the same adapter. Startup itself loads no neural weights.

## E. Changed files and limitations

Production/config/docs:

- `src/api/services/neural_mcts.py` (new): config, lazy readiness, adapter reuse, error state and budget.
- `src/api/services/matches.py`: catalog/factory/dispatch and neural opening defaults.
- `src/api/models/match.py`: additive `unavailable_reason` field.
- `configs/neural_mcts.yaml` (new), `.env.example`: approved defaults and rollback flag.
- `models/neural/README.md`, `API.md`: provisioning, API and application usage.
- Frontend `useAgentCatalog.ts`, `useArenaMatch.ts`, `AgentSelector.tsx`, API `types.ts`,
  `page.tsx`, `globals.css`, `MatchControls.tsx`: status/error plumbing, turn handoff and help text.
- Frontend `package.json` / lock, `.gitignore`, `playwright.config.ts`,
  `tests/neural-mcts.spec.ts` (new): reproducible browser checks and ignored generated reports.
- `tests/api/test_neural_mcts_api.py`, `test_neural_mcts_real.py` (new);
  `test_mcts_api.py`: isolated API coverage, real model coverage, classic test independence from optional models.
- `experiments/neural_mcts/phase5/application_smoke.py` (new), this report, `evidence/phase5/`,
  frontend `INTEGRATION.md`: reproducibility and raw evidence.

Limitations / material risks:

1. **Existing frontend dependency security findings:** the first Next build stopped on its security
   notice; retry completed successfully. `npm audit --json` reports **9 affected packages: 1 critical,
   1 high, 7 moderate**, including Next.js. [Raw audit](evidence/phase5/frontend-audit.json).
   Framework versions were not upgraded in this scoped integration. Upgrade and revalidate before
   any public production deployment. No deployment was performed or approved by this PASS.
2. Existing in-memory matches and global match lock remain: use **one worker**. Concurrent matches
   can queue behind another search. A restart clears matches and refreshes cached readiness.
3. CPU only; checkpoints must be provisioned explicitly. Torch threads are process-global.
4. Catalog refresh requires a page reload; an inference failure during play is shown through the
   existing notice. Failed readiness is cached to avoid repeated expensive loads; restart to retry.
5. Known checkpoint calibration/rule-history limitations from the model card persist. The engine
   decides all outcomes. The raw model's strength/superiority is unproven.
6. No interior mate detection, proven-win propagation, tuning, tree reuse, major optimization, batching,
   training, fine-tuning or Phase 6 benchmark changes. Classic search/agents and the pretrained loader
   were not edited. Offline tournament registration remains a later benchmark integration task.

## F. Recommendation and next gate

**Phase 5: PASS.** All twelve application acceptance criteria have executed evidence: accessible API,
selectable frontend opponent, real pretrained weights, both variants, legal moves, authoritative
terminals, clear failure paths, preserved agents/API, unit/integration tests, real-model evidence,
frontend checks and reproducible setup. No critical application item remains unverified.

No manager decision is needed to finish Phase 5. A separate dependency security remediation is needed
before public deployment. Phase 6 may start only with explicit manager approval: choose deployment
hardware/resource limits and a paired benchmark protocol, integrate the offline tournament registry,
and define equal-time/equal-simulation experiments. No playing-strength result is claimed here.

**STOP: Phase 6 was not started. Changes are uncommitted and unpushed.**
