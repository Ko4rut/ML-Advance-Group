# Phase 0 — Proposed implementation breakdown

This is planning, not implemented functionality. Only Phase 0 was authorized and executed. Later tasks require their individual phase approvals. No from-scratch training; fine-tuning requires explicit approval.

## Proposed boundaries

```text
Existing GameState / GameEngine (authoritative legality, terminal, history)
   -> checkpoint-specific observation + action adapter
   -> optional pretrained CPU inference runtime
   -> validated legal priors + side-to-move scalar value
   -> neural search lifecycle using PUCT / existing mover-view backup
   -> root visits -> AgentDecision
   -> explicit neural API / tournament registration

Classic MCTSAgent.from_config -> UCT + existing rollout/static lifecycle
```

**INFERRED contract:** one internal `predict(state, legal_moves)` result containing legal `Move -> finite nonnegative prior` and finite `value in [-1,1]` from side-to-move. Terminal engine outcomes bypass inference. The checkpoint determines the tensor/vocabulary; Gold IDs remain `from_to_8100_v1`. Distinguish raw logits from probabilities. Validate invalid/nonfinite outputs and define observable error/zero-mass policy; never hide an unavailable model behind a classic fallback labeled neural.

## Phase 1 — Qualification only, next proposed task

1. Review pinned source/weight rights and deployment/runtime constraints. Prepare an isolated CPU environment without changing backend production dependencies. Estimate framework download/disk requirements before installing a large runtime.
2. Compare cc-lite v4 and CCZero on runtime, native codec, licensing/provenance, and expected resources using the [shortlist](00_MODEL_SHORTLIST.md). Fetch only permitted checkpoint(s), one at a time. Suggested artifact sizes: 15.5 MB and 30.3 MB; compute actual SHA-256, record source revision and immutable URL. No datasets are needed.
3. Create a standalone POC and fixtures outside the core. Strict-load actual weights, set eval/inference mode, prohibit random-weight or heuristic fallback. Record tensor names/shapes, architecture payload and all framework versions. No training, BN recalibration, self-play, or fine-tuning.
4. Run source-matched initial and asymmetric positions with both sides to move. Compare tensor orientation, piece order, policy labels and value convention with source implementation. For a conversion, compare actual P/V arrays within stated tolerances; shape agreement is insufficient. Confirm original source backend and checkpoint identity.
5. Check each fixture's complete engine legal-move set maps uniquely; inspect legal mass and normalization, finite values, terminal precedence and sign sanity. Record load time, warm forward time, and peak/process RSS. Do not interpret a plausible top move or mate position's approximate NN value as proof of strength.
6. Choose a candidate only with evidence; stop with NO-GO/reselect if loading, rights, parity, codec, or resource constraints fail. Do not implement the production bridge in this phase.

**Outputs:** the master plan's `01_MODEL_CANDIDATES.md`, `01_COMPATIBILITY_MATRIX.md`, `01_POC_REPORT.md`, isolated reproducible POC/fixture evidence. **Gate:** actual pretrained bytes load and forward on CPU; source contracts/parity and legal mapping demonstrated; intended rights approved; explicit limitations/resource measurements. No compatibility guarantee was supplied by Phase 0.

## Phase 2 — Encoding contracts

Candidate-specific encoder/codec and `02_CONTRACT.md`: board orientation, piece/side/check planes, dtype/shape, native output order, logits/probs, value signs and history limitations. Prefer a model-specific adapter; do not fill generic stubs merely because they exist. Reuse repo Move/GameEngine and Gold ID definitions where they truly match.

**Tests/gate:** initial + asymmetric boards, both sides, mirrored coordinates, action round trips, every legal move in curated tactical fixtures and sampled legal trajectories mapped exactly once, illegal moves masked, finite normalization, source-tensor parity, cannon/horse/facing-kings and terminal/draw states. The actual native vocabulary must be preserved for 2086-output models; no forced 8100 head.

## Phase 3 — Inference adapter

Optional/lazy runtime, CPU default, config-driven checkpoint identity/path/device, safe strict load, computed checksum, inference/eval mode, finite P/V validation, clear missing/bad model errors and model card. Keep network/training skeletons untouched if a qualified external architecture adapter is sufficient. Handle cache/lazy-load concurrency only as needed; defer batching until measured.

**Tests/gate:** fake model validates contract/error invariants; separate real-checkpoint integration tests may skip in ordinary CI with an explicit reason but must pass in the qualification environment. Both-color multistate deterministic inference, model identity/hash, no silent fallback, measured latency. Ordinary classic imports/tests must work without the ML extra.

## Phase 4 — Neural MCTS integration

Retain classic `select/one-child-expand/rollout/backup` behavior. Add an explicit neural lifecycle that evaluates a new node once for P/V and installs legal priors before PUCT decisions. Existing `MCTSNode.expand` accepts mappings and can be reused, but trusts callers and does not validate legal/probability invariants. If expanding all children, define unvisited Q/ties, N=0 PUCT exploration, search accounting, and per-node memory. If progressive, prove priors affect selection before the old full-expansion delay. Do not just exchange UCT for PUCT and fill `NeuralEvaluator.evaluate`.

**Tests/gate:** fake policy bias changes visits, including deeper nodes; fake values affect decisions with correct signs; no repeated redundant inference on an initialized node; terminal bypasses NN; both-color mate-in-one; 1/2/100 simulation budgets, deadline overshoot documented, legal selected action and normalized visit policy; reset/reproducibility; classic seeded results/regressions unchanged. Include actual NN invocation and an approved small legal-move game. Clarify immediate-win shortcut semantics and nodes vs simulations.

## Phase 5 — Config, API and evaluation registration

Separate neural config/service and optional dependency docs; neural availability only after valid configured checkpoint load. Extend API factory/dispatch and tournament registry deliberately. Existing `isinstance(MCTSAgent)` routing and API opening book need explicit tests. Use opening_mode=none to demonstrate neural decisions. Retain unavailable with clear reason for missing/invalid checkpoint; avoid startup crash for classic users.

**Tests/gate:** catalog, both-color steps, reset, cold load/error paths, legal decisions, configured budgets, book/no-book paths, real-model smoke, rollback via feature/config flag. No frontend rewrite; existing option gets accurate availability.

## Phase 6 — Benchmark and release decision

Reuse current tournament opening/provenance/log/metrics infrastructure after adding model identity/hash. Define two independent experiments: equal per-move wall time and equal simulations. Current default mixed-agent budgets are not the fairness protocol. Include paired color swaps/openings/seeds, process/thread limits, CPU/runtime/model metadata, p50/p95 timing, simulations, load time/RSS and W/D/L/truncated/errors with score intervals. Complete-pair treatment should preserve pairing in uncertainty estimates. Do not count truncated games as engine draws.

**Tests/gate:** bounded approved smoke before any long run, raw games/replay evidence, reproducible commands/config, terminal/legal failures investigated, fair comparisons with Random/Rule-Based/Alpha-Beta/classic MCTS. Release correctness and measured strength are separate decisions; no predeclared promise of improvement.

## Phases 7–8 — Separate later authorization

Phase 7 training/fine-tuning/self-play is optional and outside the inference-first MVP. It requires explicit approval, data/model rights and a resource budget; Gold selected actions and shaped rewards are not MCTS policy/terminal value targets. Phase 8 handover covers reproducible installation, model manifest/card, tests, rollback, CI without automatic large checkpoint downloads, and separate real-checkpoint validation.

## Phase 0 handoff

Initial audit created five reports plus retained read-only evidence, without source/test/config edits. Targeted 78/78 passed; initial full baseline was 269/270 with one Silver test-helper TypeError. The subsequently authorized [one-line test repair](00_BASELINE_REPAIR.md) resolves it: **270/270 backend tests pass**, with production code and HEAD unchanged. **Conditional GO for approved Phase 1 qualification; NO-GO for integration now. STOP here and await approval.**
