# Phase 0 — Repository audit

Audit date: **2026-11-12 (Asia/Ho_Chi_Minh)**. Scope: architecture, existing tests, pretrained-model research, risks, and planning only.

**Acceptance: PASS for the Phase 0 evidence gate. Recommendation: conditional GO to Phase 1 qualification, after approval; NO-GO for production integration or a compatibility/strength claim.** The initial baseline was 269 passes and one test-helper error. The separately authorized [baseline repair](00_BASELINE_REPAIR.md) resolves that error; the full suite now passes **270/270**. Phase 1 has not started.

## Authority, repository identity, and changes

The user's request authorizes Phase 0 only and prohibits production-code changes. Both supplied documents were read completely. Their later-phase instructions are planning material, not authorization to execute them. No neural training, fine-tuning, checkpoint download, model forward pass, production implementation, commit, push, or tournament benchmark was performed. Existing lightweight backend tests, including tabular-learning tests, were exercised as part of baseline discovery.

| Item | Observed |
| --- | --- |
| HEAD before / after | `d5ad26067aa29b72fe9a0337e7cf4500593b4619` / unchanged |
| Branch | `main`; no branch switch, commit, or PR |
| Initial working tree | Clean (`git status --short` produced no entries) |
| Initial audit changes | Documentation and evidence only in `project/backend/docs/neural_mcts/` |
| Authorized follow-up change | One-line repair in `tests/data_pipeline/test_silver_pipeline.py`; no production-code change |
| Plan reference | `e60bb6c`; the checked-out code is newer and is authoritative |

Evidence labels throughout this directory: **VERIFIED** means directly inspected code, metadata, or an executed check; **INFERRED** means a proposed consequence/design; **UNKNOWN** means not established. Source availability does not prove checkpoint contents, inference correctness, licensing approval, or playing strength.

Files created, with purpose:

- `00_REPO_AUDIT.md`: checked-out architecture, call flow, contracts and recommendation.
- `00_BASELINE_TESTS.md`: commands, environment corrections, actual results and coverage gaps.
- `00_MODEL_SHORTLIST.md`: pinned pretrained candidates and qualification limits.
- `00_RISK_REGISTER.md`: unresolved risks, controls and decisions.
- `00_IMPLEMENTATION_BREAKDOWN.md`: later-phase tasks and acceptance gates, not execution.
- `evidence/`: initial/corrected targeted logs, full regression log, diagnostic probes, compact model metadata, checkpoint header observations, runtime snapshot and SHA-256 manifest. No source or checkpoint artifacts were copied into the repository.

## Verified file map

Paths below are relative to `project/backend/`; line references describe this HEAD.

| Responsibility | File and evidence |
| --- | --- |
| Domain types | [types.py](../../src/environment/types.py), lines 8–41: `Side.RED=1`, `BLACK=-1`, piece IDs 1–7, frozen/hashable `Move(source,target)` in 0–89 |
| State and board | [state.py](../../src/environment/state.py), lines 8–18, 27–70, 100–130: 10×9, row-major, black back rank at row 0, immutable board/history, red positive, black negative; trusted `moved()` returns a new state |
| Move legality | [movegen.py](../../src/environment/movegen.py), lines 14–53: geometry then simulation/check filtering; [rules.py](../../src/environment/rules.py), lines 38–167: palace, river, horse leg, elephant eye, cannon screen, facing kings |
| Terminal authority | [engine.py](../../src/environment/engine.py), lines 19–49: validates moves and gives no-move loss priority over repetition/no-capture draws; [rules.py](../../src/environment/rules.py), lines 9–20: thresholds 3 / 120 |
| Environment | [env.py](../../src/environment/env.py), lines 31–57: terminal rejection, reward from the mover's perspective, terminal-aware legal actions, immutable clone |
| Agent protocol | [base.py](../../src/agents/base.py), lines 10–43: `SearchBudget`, `AgentDecision(move,nodes,simulations,policy)` |
| Baseline agent | [mcts_agent.py](../../src/agents/mcts_agent.py), lines 23–63, 90–92: UCT + rollout factory, `(N,Q)` root choice, visit policy, reset |
| Node | [node.py](../../src/search/mcts/node.py), lines 25–104: lazy legality/terminal generation, capture-ordered untried moves, configurable edge prior, N/W/Q |
| Search | [search.py](../../src/search/mcts/search.py), lines 54–155: fresh tree, immediate-win shortcut, one-child expansion, scalar evaluation, alternating backup, visit distribution |
| Selection | [selection.py](../../src/search/mcts/selection.py), lines 13–58: implemented UCT and PUCT formulas; PUCT consumes `child.prior` |
| Leaf evaluation | [evaluators.py](../../src/search/mcts/evaluators.py), lines 16–101: working rollout/static evaluator, unimplemented neural evaluator |
| Neural placeholders | [neural_mcts_agent.py](../../src/agents/neural_mcts_agent.py): both methods `pass`; [actions.py](../../src/encoding/actions.py), [observations.py](../../src/encoding/observations.py), [backbone.py](../../src/networks/backbone.py), [policy_value.py](../../src/networks/policy_value.py): stubs |
| Training/storage placeholders | [alphazero.py](../../src/training/alphazero.py), [self_play.py](../../src/training/self_play.py), [replay_buffer.py](../../src/training/replay_buffer.py), [checkpoints.py](../../src/storage/checkpoints.py), [game_record.py](../../src/storage/game_record.py): stubs; unnecessary for inference MVP |
| API wiring | [matches.py](../../src/api/services/matches.py), lines 34–43, 99–127, 193–215; [mcts.py](../../src/api/services/mcts.py), lines 26–46; [routes](../../src/api/routes/matches.py), lines 22–24, 56–58 |
| Tournament wiring | [agent_registry.py](../../src/evaluation/agent_registry.py), lines 147–182, 259–275, 326–341; [tournament.py](../../src/evaluation/tournament.py), lines 261–296; [match.py](../../src/evaluation/match.py), lines 35–77 |
| Gold contracts | [models.py](../../src/data/03_gold/models.py), lines 11–42, 45–80; [pipeline.py](../../src/data/03_gold/data_pipeline/pipeline.py), lines 82–94, 123–138, 211–242; [rewards.py](../../src/data/03_gold/data_pipeline/rewards.py), lines 106–142 |
| Package/config/docs | [pyproject.toml](../../pyproject.toml), [mcts.yaml](../../configs/mcts.yaml), [alphazero.yaml](../../configs/alphazero.yaml), root/backend README, [TOURNAMENT.md](../../TOURNAMENT.md), [TOURNAMENT_HANDOVER.md](../../TOURNAMENT_HANDOVER.md) |
| Frontend surface (read-only search) | `project/frontend/src/features/arena/constants.ts:12`, `types.ts:11`: existing `neural` option/type. Browser behavior not tested. |

Tests inspected and run include `tests/search/test_mcts.py`, `tests/environment/test_environment.py`, `tests/test_foundation.py`, `tests/api/test_mcts_api.py`, `tests/evaluation/test_api_parity.py`, `tests/evaluation/test_tournament.py`, and Gold pipeline/reward tests. Full backend discovery also ran; see [baseline evidence](00_BASELINE_TESTS.md).

## Current call flow

```text
POST /api/v1/matches -> MatchService._make_agent
  -> MCTSService.make_agent -> MCTSAgent.from_config
     -> MCTSSearch(GameEngine, UCTSelection, RolloutEvaluator)

POST /api/v1/matches/{id}/step -> MatchService.step (under RLock)
  -> terminal / turn / expected-ply checks
  -> opening-book move, if available (may bypass search)
  -> MCTSService.select_move -> SearchBudget(time_limit,max_simulations)
  -> MCTSAgent.select_action -> validate_search_root
  -> MCTSSearch.search -> fresh MCTSNode -> generate legal moves + outcome
     -> immediate winning move? expand uniform-prior child; record 1 simulation
     -> otherwise loop:
          select only through fully expanded nonterminal nodes
          expand one capture-ordered untried move with P=1/num_legal
          exact terminal reward OR scalar rollout/static value
          backup: flip sign BEFORE each node update
     -> root child max(N,Q) -> normalized child-visit policy -> AgentDecision
  -> validate move against engine legal moves -> env.step -> API snapshot

Tournament -> agent_registry._make_mcts -> same MCTSAgent.from_config
  -> per-side budget_for -> agent.select_action -> engine transition + game log
Standalone run_mcts_matches -> MCTSAgent.from_config -> MatchRunner
  -> same budget passed to both agents, always starts from initial state
```

A repository-wide Python reference search found production MCTS construction in `MCTSAgent.from_config`, called by API `MCTSService`, tournament `agent_registry`, and `evaluation/run_mcts_matches.py`. Direct constructor use also occurs in `tests/search/test_mcts.py`. CLI `tournament` reaches the registry indirectly. No production construction of `PUCTSelection` or use of `NeuralEvaluator` was found.

## Policy/value and expansion findings

**VERIFIED: supplying a neural scalar evaluator alone cannot provide neural policy guidance.** `search.py:111–117` pops one untried move and assigns `1/num_legal`; `search.py:62–71` also uses a uniform prior in the immediate-win shortcut. The evaluator receives only state and returns a scalar (`search.py:119–125`). There is no implemented policy-result channel to expansion.

**VERIFIED: PUCT selection is bypassed while a node still has untried moves.** `select()` stops at such a node (`search.py:101–109`). The initial board has 44 legal moves. Diagnostic runs with PUCT produced:

| Simulations | Root children | Untried | Every child prior | Root child visits |
| --- | --- | --- | --- | --- |
| 2 | 2 | 42 | 1/44 | 1 each |
| 44 | 44 | 0 | 1/44 | 1 each |
| 45 | 44 | 0 | 1/44 | one child 2, others 1 |

The formula itself selects prior 0.99 over 0.01 when manually supplied children have equal Q/N. That diagnostic establishes the wiring gap; it is not neural inference evidence. See [probe output](evidence/search-probes.log).

**INFERRED design direction:** retain the classic lifecycle and inject a separately testable neural expansion/evaluation strategy, or a separate neural search driver reusing nodes/PUCT/backup. On a neural node's first evaluation, obtain P/V together and establish legal children with those priors before selection. Decide root evaluation, simulation accounting, zero-visit ties, terminal precedence, and memory semantics explicitly. Do not globally replace classic one-child expansion. A final interface depends on the qualified checkpoint; [breakdown](00_IMPLEMENTATION_BREAKDOWN.md).

## Signs, rules, and budgets to preserve

- **VERIFIED:** evaluator value is from `state.side_to_move`; each child Q is from the parent mover's view. Backup negates before the leaf update and again at each ancestor (`node.py:3–13`, `search.py:127–133`). Root W is the opponent's view and is not the root move-selection score. Do not copy another engine's backup sign convention blindly.
- **VERIFIED:** no legal move loses even without check. Repetition is board + side, three occurrences; 120 consecutive non-capturing plies draw. A pawn move does **not** reset this repo's no-capture counter. Engine outcome overrides any NN prediction; `GameEngine.legal_moves` alone does not suppress draw-terminal moves, whereas environment legal actions do.
- **VERIFIED:** `SearchBudget.max_nodes` and `max_depth` are not consumed by MCTS. `max_simulations` and `time_limit` are consumed; absent both, default is 400 simulations. `from_config` does not store a time limit; callers provide it.
- **VERIFIED:** root generation and the immediate-win scan precede deadline initialization, and the loop guarantees at least one simulation. Limits are soft and checked between simulations. A 1 ns requested limit took approximately 10.3 ms in one diagnostic, including agent/root work. This is not a stable latency benchmark; NN load/forward can increase overshoot.
- **VERIFIED:** default API config is capture-first depth-2 rollout, c=1.4, eval_scale=100, time_limit=1 second, seed=0. A bare factory defaults to random/static depth 0. Tests often use static evaluation and simulation budgets; preserve both modes.
- **VERIFIED:** full-rollout mode capped at 300 plies returns 0 (`evaluators.py:65–67`). That is existing search-evaluation behavior; it must not become a terminal draw or a training label. No change proposed in Phase 0.
- **VERIFIED:** API catalog keeps neural unavailable. MatchService holds a service-wide lock during search, and opening-book moves can bypass the agent. Later NN-invocation tests must disable book openings. An inherited `MCTSAgent` could be dispatched through the classic service budget by the current `isinstance` branch; design Phase 5 dispatch deliberately.

## Gold data and dependency boundaries

**VERIFIED:** Gold uses `source*90+target`, 8100 IDs, `from_to_8100_v1`; its policy samples contain a selected action, not an MCTS visit distribution. Legal masks exist in Gold even though `ActionEncoder` is empty. Game-hash split and truncation/recorded-terminal distinctions already exist. Shaped offline-RL rewards are actor-perspective transitions, not pretrained value targets. Gold board rows omit the complete engine rule history; do not substitute them for full terminal-aware search states.

**VERIFIED:** `pyproject.toml` requires Python >=3.10 and Pydantic, with optional API/Drive/test extras. It declares no ML framework. Current `.venv` uses Python 3.14.9 and initially lacked FastAPI and dotenv; those already-declared packages were installed into `/tmp` only for tests. Torch, TensorFlow, Keras, h5py, and ONNX Runtime are absent. No package or config file was changed.

**INFERRED:** future inference dependencies should remain optional/lazy, with CPU support and explicit checkpoint validation. The inference MVP does not require filling training/storage skeletons or modifying Gold's ID contract.

## Environment and existing experiments

**VERIFIED local host:** Intel Core i5-13420H, 12 logical CPUs / 8 cores reported by `lscpu`; about 15 GiB RAM total and 5.3 GiB available at initial inspection. `nvidia-smi` exists but fails to communicate with the driver (exit 9). A usable CUDA device is **UNKNOWN**; no GPU capacity is assumed. This is the audit machine, not a confirmed deployment specification. A final [runtime snapshot](evidence/runtime.txt) records repeat observations; available RAM varies with system load.

**VERIFIED existing artifact:** `results/tournaments/20261112-091526_501370b5/summary/report.md` reports 750 games, 748 finished, 2 truncated, 0 errors, using commit `22d8943e37` on `feature/round-robin-tournament`. It was read, not regenerated or replay-validated here. It predates this audit HEAD and includes no neural agent. It cannot support a neural strength claim. Current tournament defaults also use different per-agent budgets, so Phase 6 needs explicit equal-time and equal-simulation protocols.

Documentation drift: backend README uses old `project` execution paths and says search/evaluation are skeletons; actual MCTS/tournament code is implemented. TOURNAMENT_HANDOVER describes a not-yet-run full tournament and CI additions, but a full-run report exists and no `.github` workflow directory exists at this checkout. These are historical notes, not current acceptance evidence.

## Results, unknowns, and decision

See [tests](00_BASELINE_TESTS.md), [model shortlist](00_MODEL_SHORTLIST.md), [risks](00_RISK_REGISTER.md), and [implementation gates](00_IMPLEMENTATION_BREAKDOWN.md).

**VERIFIED:** targeted 78/78 pass; regression 269/270 pass, one pre-existing Silver test-helper TypeError. No pretrained checkpoint is present in the inspected backend model directory: its artifact is a Tabular Q table, not a policy-value NN. No NN was loaded or benchmarked.

**UNKNOWN / decisions deferred:** model and weight-use approval; actual CPU latency/RSS; checkpoint bytes/hash/load correctness; exact conversion parity for the legacy route; deployment Python/hardware and acceptable move latency; rule mismatch impact. These do not prevent an isolated qualification POC, but prevent claiming production readiness. The Silver test error has since been repaired under separate user authorization, with 270/270 passing; see the follow-up report.

**Next task, only after approval:** Phase 1 CPU checkpoint qualification, first considering the pinned cc-lite distilled checkpoint and comparing it with the pinned CCZero option. Validate rights, bytes/hash, actual loading, P/V conventions, source parity, and resource cost; stop at its gate. Do not start Phase 2, change core MCTS, or train/fine-tune.
