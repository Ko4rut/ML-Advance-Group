# Model card — cc-lite `distill-v4-large` / `distill-v4` (Phase 3 inference adapter)

Written for: engineers integrating Neural MCTS (Phase 4) and reviewers of the model/licence basis.

Date: 2026-11-13. Branch `agent/mcts_nn`, HEAD at start `ab33de0`, nothing committed. Tags: **VERIFIED** = observed in a command run this phase (evidence in [evidence/phase3/](evidence/phase3/)); **INFERRED**; **UNKNOWN**.

## 1. Identity and provenance

| | `distill-v4-large` (default) | `distill-v4` (fallback) |
| --- | --- | --- |
| Source | <https://github.com/hskl18/cc-lite>, revision `7d305c0e5adbfae4e23db40997603d8dc86b078b`, `evidence/distill-v4-large/train/checkpoint.pt` | same revision, `evidence/distill-v4/train/checkpoint.pt` |
| SHA-256 (pinned in [manifest.json](../../models/neural/manifest.json)) | `05927ccd874c4d97588ba7d06266fda9ba17700d436fdc03ff6844a0d61d0dff` | `1be52664b216529ce56240043406878a93f95f57e4153ff73c4d63afd772fba4` |
| Size | 38,033,337 B | 15,500,553 B |
| Architecture | residual tower, 192 ch × 12 blocks, 9,484,219 params | 128 ch × 8 blocks, 3,860,347 params |
| Training (upstream metadata in the payload) | supervised distillation from Fairy-Stockfish labels, 3,320,673 samples, seed 42; training-run metrics recorded in the payload: legal-move acc. 0.971, target acc. 0.351 | same data; legal-move acc. 0.965, target acc. 0.330 (same metrics) |
| Format | `torch.save` payload `{model_state, model_config, step, metadata}`, loaded with `weights_only=True` | same |
| Licence basis | repository Apache-2.0; upstream states labels came from Fairy-Stockfish with a CC0 network (DESIGN.md). No separate weights licence. Approved by the manager on 2026-11-12 ([01_POC_REPORT §9b](01_POC_REPORT.md)). This is an engineering reading, not legal advice | same |

The network definition in [torch_backend.py](../../src/networks/pretrained/torch_backend.py) is a re-implementation of upstream `src/model/network.py` (Apache-2.0) with identical layer layout; this is credited in the module docstring. Nothing from cc-lite is vendored. **VERIFIED:** raw logits of the production backend are bit-identical to the upstream wrapper's, and values are identical, on 467 states / 17,066 legal moves for both checkpoints ([evidence](evidence/phase3/live-upstream-parity-distill-v4-large.log), [v4](evidence/phase3/live-upstream-parity-distill-v4.log)).

## 2. Runtime

Python ≥ 3.10 (validated 3.14.9); optional `torch` (validated `2.11.0+cpu`, declared as the `neural` extra, `torch>=2.11,<2.12`). No other ML dependency (no numpy needed by the adapter). Importing `src.environment`, `src.encoding`, `src.search`, `src.agents.mcts_agent` and `src.networks.pretrained` never imports torch (tested with torch import blocked). CPU only: `XIANGQI_NEURAL_DEVICE` accepts only `cpu`; GPU was not validated and is rejected explicitly.

## 3. Inference contract

`PolicyValueAdapter.predict(state, legal_moves=None, *, verify_legal_moves=False) -> PolicyValueResult(priors, value, terminal)` ([adapter.py](../../src/networks/pretrained/adapter.py)). Encoding and action mapping are the Phase 2 contract ([02_CONTRACT.md](02_CONTRACT.md)).

| Item | Definition |
| --- | --- |
| Input | `GameState` → `ObservationEncoder` (`absolute_planes16_v1`, float32 `(1,16,10,9)`, absolute board, plane 14 = Red to move, plane 15 = side to move in check) |
| Raw outputs | policy `(1,8100)` logits (native index = `source*90+target`, via `CCLITE_VOCABULARY`); value `(1,)` tanh |
| `priors` | `{legal Move: p}` over exactly the supplied legal moves, finite, ≥ 0, sum 1 (softmax over legal logits only). Read-only mapping, in the caller's move order |
| `value` | finite, in [-1, 1], **side-to-move perspective**, passed through unchanged (no colour-dependent flip) |
| `legal_moves` | defaults to `engine.legal_moves(state)`; a caller that has it (MCTS) passes it. Trusted unless `verify_legal_moves=True`. Duplicates / non-`Move` entries always raise |
| Terminal precedence | If `engine.outcome(state, legal_moves)` is not `None` (no legal move = checkmate/stalemate loss, threefold repetition, 120-ply no-capture draw): **the network is not called**, `terminal` = the engine's `GameResult`, `priors` is empty, `value = result.reward(side_to_move)` ∈ {-1, 0}. An empty `legal_moves` list is therefore a terminal state, as in `GameEngine.outcome` |
| Invalid network output | `InvalidModelOutputError`: wrong shape/dtype, NaN/inf anywhere in the policy (even on illegal slots), value non-finite/out of range/non-numeric, no finite legal mass. No uniform-prior or heuristic fallback exists |
| Determinism | same input → bit-identical priors and value (tested over repeated calls; on CPU with the default or 1 thread) |
| Concurrency | forward passes are serialised by a lock; `adapter.inference_calls` / `terminal_shortcuts` count network calls and terminal shortcuts (lets Phase 4 tests prove the network is really invoked) |

Loading ([manifest.py](../../src/networks/pretrained/manifest.py), [torch_backend.py](../../src/networks/pretrained/torch_backend.py)): the file is read once, its size and SHA-256 are checked against the manifest, and **those same bytes** are deserialised (`weights_only=True`, so pickled code is never executed — tested with a booby-trapped payload). The payload's `model_config` must equal the manifest architecture, the state dict must load with `strict=True`, the parameter count must match and every weight must be finite. The model is put in `eval()`, parameters get `requires_grad=False`, inference runs under `torch.inference_mode()`, and a model found in training mode refuses to infer. Weights are never modified (tested bitwise).

## 4. Configuration

Environment variables, `.env.example` entries and provisioning steps: [models/neural/README.md](../../models/neural/README.md). Checkpoints live outside git (`.gitignore` rule, tested); `scripts/fetch_neural_checkpoint.py` is an explicit, hash-verified download of the pinned upstream revision (the two URLs were HEAD-checked on 2026-11-13: HTTP 200, `content-length` equal to the manifest sizes). Nothing is downloaded at startup.

## 5. What the model is and is not (carried from Phase 1/2, unchanged)

- Upstream describes this as "a careful research setup, not a strong engine": v4-large scored 14/60 and v4 5.5/60 against Pikafish at depths 1–4. Their difference is not statistically established. We did **not** reproduce these numbers; Phase 1 only showed policy-only play beats a seeded random mover 17W-3D-0L (20 games, both models). Strength inside our MCTS is **unmeasured** (Phase 6).
- The raw policy does not reliably prefer a mate-in-one (mating moves hold 17–28 % of the prior mass in Phase 1). The engine's terminal detection stays authoritative.
- Not exactly colour-rotation invariant; the observation is absolute with a turn plane.
- Both checkpoints predate upstream's BatchNorm-statistics fix ([cc-lite issue #21](https://github.com/hskl18/cc-lite/issues/21)); value magnitudes may be under-reported, `distill-v4` more than `distill-v4-large` (initial position, Red to move: −0.149 vs +0.126, reproduced by the Phase 3 tests).
- Rule mismatch: the source adjudicates perpetual check and resets its no-progress counter on pawn moves; our engine draws every threefold repetition and resets only on capture. The 16 planes carry no history or counters. Engine results always win; the value quality in positions the source engine would have adjudicated differently is unmeasured.
- Positions with the two kings facing on an open file are outside the contract (they cannot arise in legal play).
- Coverage of the equivalence evidence is finite (472 golden states + 59-state stored reference per model), not a proof for all positions.

## 6. CPU performance (VERIFIED, method in [benchmark.py](../../experiments/neural_mcts/phase3/benchmark.py))

Machine: Intel Core i5-13420H (4P+4E cores, 12 logical; hybrid, so single-thread timings drift between runs), Linux 7.2.14, Python 3.14.9, torch 2.11.0+cpu, a desktop that was not isolated for benchmarking. 200 non-terminal states from seeded random walks, 10 warm-up calls, legal moves supplied, batch size 1. Raw JSON: [evidence/phase3/benchmark-*.json](evidence/phase3/).

| Model | Threads | `predict` p50 / p95 (ms) | raw forward p50 (ms) | adapter overhead p50 (ms) |
| --- | --- | --- | --- | --- |
| distill-v4-large | 1 | 19.9 / 20.3 | 19.6 | 0.33 |
| distill-v4-large | 4 | 7.6 / 14.2 | 7.3 | 0.37 |
| distill-v4-large | 8 (torch default) | 10.2 / 10.5 | 9.8 | 0.36 |
| distill-v4 | 1 | 10.3 / 10.5 | 9.9 | 0.47 |
| distill-v4 | 4 | 3.1 / 5.6 | 2.9 | 0.36 |
| distill-v4 | 8 (torch default) | 3.7 / 3.9 | 3.3 | 0.35 |

| | distill-v4-large | distill-v4 |
| --- | --- | --- |
| Load: SHA-256 + deserialise + build + validate (torch already imported) | 0.11–0.13 s | 0.05–0.06 s |
| First `import torch` (once per process) | ≈0.7 s | ≈0.7 s |
| RSS: after importing torch → after load → process peak | 232 → 279 → 322 MiB | 231 → 273 → 279 MiB |
| Memory attributable to load (RSS delta) | ≈47 MiB | ≈42 MiB |

Observations and limits: encoding, legal-prior softmax and validation add ≈0.35 ms (≈3–12 % of a forward); the forward dominates. Timings vary between runs on this hybrid CPU: an earlier, non-interleaved run of the same script gave 1-thread `predict` p50 of 7.1 ms (v4) and 29 ms (v4-large), and Phase 1 measured 6.75 / 20.0 ms. Treat them as ±40 % at 1 thread. 4 threads were faster than torch's 8-thread default on this machine for batch 1. Engine move generation (≈1.8 ms/node, Phase 1) is not included here. These are per-position costs, not MCTS throughput: the 3 s/move budget is a Phase 4/6 question (Phase 1 §9b estimated ≈120–330 nodes for v4-large, 330–590 for v4 at batch 1). No batching, caching or GPU work was done; nothing measured here shows it is needed before Phase 4 profiles real search.

## 7. Verification summary

| Check | Result |
| --- | --- |
| Standard env (no torch), `python -m unittest discover -s tests` | 374 tests OK, 19 skipped — all skips are torch/real-weight tests with an explicit reason ([log](evidence/phase3/backend-full-suite-standard.log)) |
| Validation env (torch + `XIANGQI_NEURAL_REQUIRE_REAL=1`) | 394 tests OK, 0 skipped; includes 20 real-weight tests ([log](evidence/phase3/backend-full-suite-validation.log)) |
| Baseline (Phase 2 HEAD) | 313 tests OK before any change; unchanged after (the 313 are a subset of both runs above) |
| Live upstream parity, both checkpoints | 467 states / 17,066 legal moves; logits bit-identical, value error 0, max prior error 1.6e-7 |
| Mutation checks | 14 injected bugs (unsafe unpickling, hash/size check off, sign flip, terminal shortcut off, uniform fallback, value range, training mode, non-strict load, architecture check, SHA assertion, finite check, GPU device, mask) — all detected, baseline green ([log](evidence/phase3/mutation-checks.log)) |

Reproduce: [experiments/neural_mcts/phase3/README.md](../../experiments/neural_mcts/phase3/README.md) and `demo.py` (loads the configured model, prints priors/value for both sides, a mate-in-one, a terminal state, then 20 plies of argmax self-play with every move checked legal).

## 8. Open items for Phase 4+

1. *(Superseded by Phase 4, see [04_SEARCH_LIFECYCLE.md](04_SEARCH_LIFECYCLE.md).)* Search integration was untouched at the end of Phase 3: `MCTSSearch.expand` still assigns uniform priors, `NeuralEvaluator`/`NeuralMCTSAgent` remain stubs, the API registry still reports `neural` unavailable.
2. Single-observation inference only; a batched `infer` is a small extension of `InferenceBackend` if Phase 4 profiling shows leaf batching pays (Phase 1 measured a ≈1.6–1.7× per-position gain at batch 8 on 4 threads).
3. Thread count is process-global in torch (`XIANGQI_NEURAL_THREADS`); a server hosting several models or workers must choose it once.
4. GPU is rejected, not merely untested-by-default; enabling it needs its own parity and determinism validation.
5. The Phase 1 runtime and source live in `/tmp` and vanish on reboot; recreate them with the Phase 1 README for the live-parity test. The stored-reference tests need neither.
