# Phase 1 — POC report: pretrained Xiangqi policy-value checkpoint qualification

- Repository HEAD: `c75f0597f859e32eea8c55e70c8e957a2c517b40`, branch `agent/mcts_nn`. Nothing committed; no `src/`, `tests/`, API, agent-registry or dependency file was modified (`git status` shows only untracked `docs/neural_mcts/01_*`, `docs/neural_mcts/evidence/phase1/`, `experiments/`).
- Dates: Codex produced the harness and first run on 2026-11-12; Claude Code re-ran, extended and wrote this report on 2026-11-12.
- Tags: **VERIFIED** = observed in an executed command or hash-checked file this phase. **INFERRED** = reasoning from verified facts. **ASSUMPTION** / **UNRESOLVED** / **BLOCKER** as named.

## 1. Outcome

| Gate item (Phase 1 definition of done) | Status |
| --- | --- |
| 1. Real pretrained Xiangqi policy-value checkpoint identified and verified | **VERIFIED** — two: cc-lite `distill-v4` and `distill-v4-large`, SHA-256 checked |
| 2. Loads | **VERIFIED** — original upstream architecture, strict `model_state` load, `weights_only=True`, CPU |
| 3. Real inference on representative states | **VERIFIED** — initial, opening line, midgame, check, mate-in-one, checkmated, draw positions; both colors |
| 4. I/O contracts documented | **VERIFIED** — [compatibility matrix](01_COMPATIBILITY_MATRIX.md) |
| 5. Policy → legal-move mapping feasible | **VERIFIED** — all legal moves of 52 states (fixtures + seeded trajectory) match the source codec exactly |
| 6. Value perspective and sign | **VERIFIED** — side-to-move, no color sign flip (§5) |
| 7. License/provenance evaluated | Evaluated: Apache-2.0 + documented CC0-network labels (engineering assessment, not legal advice). **Manager sign-off still required** |
| 8. Remaining integration requirements | Documented (§8) |
| 9. Reproducible evidence | **VERIFIED** — raw JSON/logs in [evidence/phase1/](evidence/phase1/), commands in §3 |

**Phase 1 status: PASS (technical), conditional on manager approval of the model choice and its license basis.** No strength claim is made: upstream reports both checkpoints lose most games to a real engine (§7).

## 2. Recovery of the previous agent's work

Codex had finished: Phase 0 docs and baseline repair; Phase 1 research docs `01_MODEL_CANDIDATES.md`, `01_COMPATIBILITY_MATRIX.md`; the isolated harness [experiments/neural_mcts/phase1/](../../experiments/neural_mcts/phase1/) (`fetch_artifacts.py`, `contracts.py`, `qualify.py`, `test_poc.py`, hash manifests); and a working CPU runtime, pinned sources and the v4 checkpoint under `/tmp/neural-mcts-phase1`. Codex's reports cited `evidence/phase1/result.json` and a `01_POC_REPORT.md` that **did not exist**.

Claude Code in this session:

1. Re-hashed every artifact and re-ran Codex's tests and qualification independently. Result JSON is **identical** to Codex's except timings (§3, run 2).
2. Preserved the raw evidence in the repo (it previously lived only in `/tmp`).
3. Added `sanity_probe.py` (semantic checks Codex did not have: opening line, material value-sign probe, policy-only play).
4. Added a second real checkpoint, `distill-v4-large`, through a `--variant` option (default behavior unchanged), to make the alternatives comparison an executed one.
5. Re-ran the full backend suite to confirm the baseline is unchanged.
6. Wrote this report and revised the two `01_*` docs where this session's evidence changed the recommendation.

## 3. Commands and results (all executed)

Environment: Python 3.14.9, torch 2.11.0+cpu (hash-locked in [requirements-cpu.txt](../../experiments/neural_mcts/phase1/requirements-cpu.txt)), numpy 2.5.3, 1 CPU thread, Linux 7.2.14, Intel i5-13420H host. Runtime installed with `--target` under `/tmp/neural-mcts-phase1/deps` only (not the shared `.venv`). Variables: `W=/tmp/neural-mcts-phase1`, `PY=../../.venv/bin/python`, run from `project/backend`, `PYTHONPATH="$W/deps:$PWD:$PWD/experiments/neural_mcts/phase1"`.

| # | Command | Result |
| --- | --- | --- |
| 1 | `sha256sum` of both checkpoints | v4 `1be52664…fba4`, v4-large `05927ccd…0dff`; **match** pinned manifests; sizes 15,500,553 / 38,033,337 B |
| 2 | `XIANGQI_PHASE1_SOURCE=$W/source XIANGQI_PHASE1_CHECKPOINT=…/distill-v4/train/checkpoint.pt $PY -m unittest discover -s experiments/neural_mcts/phase1 -p test_poc.py -v` | **21 tests OK, 0 skipped** (19 contract tests + 2 real-checkpoint tests). Re-run after my `--variant` edit: 21 OK. Log: [poc-tests.log](evidence/phase1/poc-tests.log) |
| 3 | `$PY experiments/neural_mcts/phase1/qualify.py --source $W/source --checkpoint <v4> --output …` | Completed, all internal assertions held; [result-distill-v4.json](evidence/phase1/result-distill-v4.json) equals Codex's `result.json` except timings |
| 4 | same with `--variant distill-v4-large` and its checkpoint | Completed, all assertions held; [result-distill-v4-large.json](evidence/phase1/result-distill-v4-large.json) |
| 5 | `sanity_probe.py` for each variant (`--games-per-side 10`) | Completed; [v4](evidence/phase1/sanity-distill-v4.json), [v4-large](evidence/phase1/sanity-distill-v4-large.json) |
| 6 | `PYTHONPATH=/tmp/neural-mcts-phase0/deps:$REPO $PY -m unittest discover -s tests -v` (backend baseline) | **270 tests OK**; [log](evidence/phase1/backend-full-suite.log) |

A first attempt at run 4 failed with `ValueError: Unexpected size` — my own bug (a loop variable shadowed the checkpoint record in the new `--variant` code). The harness's size check caught it; fixed and re-run, runs 2–4 above are post-fix. A first attempt at run 6 omitted `$REPO` from `PYTHONPATH` and produced 9 import errors; with the documented command the suite is green.

`qualify.py` asserts, per state: observation equals the upstream encoder bit-for-bit; engine-legal action IDs equal the upstream codec's legal set; engine-masked priors equal the upstream `TorchEvaluator` priors (≤1e-6) and value (≤1e-6); two repeated forwards are bit-identical; terminal states bypass the network; model weights/BN buffers are unchanged after all calls.

## 4. Real inference observations

Shapes/dtypes (**VERIFIED**): input `float32 (1,16,10,9)`; policy `(1,8100)` raw logits (finite); value `(1,)` tanh in [-1,1]. Priors after legal masking: finite, ≥0, sum 1 (error <1e-12), only legal moves present; unit tests prove an enormous logit on an illegal action has no effect and zero/NaN/Inf/out-of-range inputs raise rather than fall back.

Parity vs upstream wrapper (**VERIFIED**, 52 states, 1,984 legal-move occurrences, both models): max prior error 4.5e-8 (v4-large worst case), value error 0.

| State | v4 value (side to move) | v4 top-3 policy (UCCI) | v4-large value | v4-large top-3 |
| --- | --- | --- | --- | --- |
| Initial, Red | −0.149 | g3g4 .19, c3c4 .16, **h2e2 .15** | +0.126 | b2e2 .25, **h2e2 .21**, c3c4 .11 |
| Initial board, Black to move (artificial) | +0.116 | g6g5 .20 | −0.015 | b7e7 (central cannon) .43 |
| After h2e2 h9g7 h0g2 i9h9 i0h0 b9c7, Red (midgame-ish) | −0.107 | c3c4 .29, g3g4 .19, b0c2 .15 | +0.149 | c3c4 .23, h0h6 .17, g3g4 .15 |
| Mate-in-one, Red (crafted) | +0.9997 | mating moves hold 28.4 % mass; top move **is** a mating move | +0.9998 | mating mass 24.8 %; top move is **not** a mating move |
| Mate-in-one, Black (color-swapped) | +1.000 | mating mass 17.2 %; top move not mating | +0.9994 | mating mass 17.1 %; top move not mating |
| In check, 2 legal replies (Red / Black) | −0.968 / −0.766 | top reply gets .68 / .97 | −0.794 / −0.499 | .86 / .87 |
| Engine-terminal (checkmate / 120-ply draw / threefold) | −1 / 0 / 0, **network not called** | — | same | — |

Opening line ([v4-large probe](evidence/phase1/sanity-distill-v4-large.json), [v4 probe](evidence/phase1/sanity-distill-v4.json)): of the six moves of a standard central-cannon / screen-horse development line (`h2e2 h9g7 h0g2 i9h9 i0h0 b9c7`), all six are in v4-large's top 3 policy moves and five of six in v4's (`i0h0` is 4th or lower); the policy is human-plausible (horse and cannon development, pawn pushes, `h2e2` central cannon). Each position was checked legal by our engine.

**Policy-only play (argmax over legal priors, no search) vs seeded uniform-random mover, 10 games per side, 200-ply cap:** both models 17 wins / 3 draws / 0 losses / 0 unfinished (20 games). Evidence that the policy is meaningful. It is not a strength measurement.

**Observed weakness (VERIFIED):** the raw policy does not reliably prefer a mate-in-one (mating moves hold only 17–28 % of prior mass). The raw network is also not exactly color-rotation invariant (e.g. in-check Red vs Black differ). MCTS must therefore keep engine terminal detection authoritative, which the master plan already requires.

Performance (**VERIFIED**, 1 thread, warm, batch 1, 20 calls; whole-process peak RSS includes imports and verification snapshots, not model-only):

| | v4 | v4-large |
| --- | --- | --- |
| Parameters | 3,860,347 | 9,484,219 |
| Forward p50 / p95 | 6.75 / 6.89 ms | 20.0 / 21.0 ms |
| Load (excl. imports) | 0.035 s | 0.070 s |
| Peak RSS (whole POC) | ~375 MiB | ~462 MiB |

## 5. Value perspective and sign convention

**VERIFIED by three independent lines:**
1. Upstream definition: value target is the Fairy-Stockfish *side-to-move* score/1000 (clipped/squashed) — [DESIGN.md](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/DESIGN.md) "Labels" section (file hash-verified locally).
2. Upstream code: evaluator uses `board.turn`, no sign change (Phase 0/1 source review); our wrapper passes the value through unchanged.
3. Measured on our engine, both colors: with a rook or cannon removed, the value for the **side to move** is positive when that side is ahead and negative when behind, for Red-to-move and Black-to-move alike (e.g. v4: Black loses both rooks → +0.9999 Red-to-move / −0.9982 Black-to-move; Red loses both rooks → −0.9993 / +0.9997; v4-large analogous, see [sanity JSON](evidence/phase1/sanity-distill-v4-large.json)).

Consequence for our MCTS (**INFERRED**, to be tested in Phase 4): the network value is already side-to-move — the same convention as the repo's leaf evaluation (`state.side_to_move`) — so **no color-dependent sign flip at the adapter**; the only sign work is the existing per-ply backup flip.

**Calibration caveat (VERIFIED, upstream-acknowledged):** both checkpoints predate cc-lite's BatchNorm-statistics fix ([issue #21](https://github.com/hskl18/cc-lite/issues/21)). Upstream measured v4's held-out value-sign agreement at 90.9 % vs 96.5 % for v4-large. We observed it: v4 rates the initial position −0.149 for Red (to-move disadvantage), v4-large +0.126. We did not recalibrate or alter weights.

## 6. Rule / observation mismatches (carry into Phase 2)

- Our engine: all threefold repetitions draw; 120 no-capture plies draw (pawn move does not reset). Source: perpetual-check adjudication, 120 plies without capture **or pawn move**. The 16 input planes carry no repetition/counter state. **Engine terminal results always take precedence**; the network sees positions the source engine would have adjudicated differently. Effect on play quality unmeasured.
- Observation is absolute (no rotation to side-to-move): planes 0–6 Red K,A,E,H,R,C,P, 7–13 Black, plane 14 = 1 if Red to move, plane 15 = side-to-move in check. Our engine `abs(piece)` IDs 1–7 are exactly that order, so the encoder is ~10 lines (reference implementation in `contracts.py`, not production).
- Action IDs `source*90+target` coincide with our Gold ID convention and our `Move.source/target` (row-major, row 0 = Black's back rank). This was **verified by exact legal-set equality with the source codec**, not assumed from matching sizes. Gold-dataset policy labels are a separate concern (not used here).

## 7. Strength, license and provenance (what is and is not established)

- **Upstream's own evidence** (README/DESIGN, locally hash-verified): v4 beats scripted baselines, beats v3 38–2–8 in paired games, scored 5.5/60 against Pikafish at depths 1–4; v4-large scored 14/60 and agrees with the engine on 36.7 % of held-out positions vs 33.6 %. Upstream says: "a careful research setup, not a strong engine." v4 vs v4-large paired score 0.594 (CI 0.458–0.719) — **not a demonstrated difference**. We did not reproduce any of these numbers; we only reproduced 20 policy-only games vs random.
- **License:** repository Apache-2.0 (LICENSE hash-verified). Labels came from Fairy-Stockfish with a CC0 xiangqi network specifically to avoid Pikafish's restricted weights (DESIGN.md lines 118–119). This is the upstream author's statement; we did not regenerate labels or obtain separate legal review. The checkpoint directory carries no separate weight license. Nothing from cc-lite is vendored into this repository.
- **Provenance:** checkpoint hash equals the hash in the upstream training manifest in the same repository at the same pinned commit. That proves we have the file upstream describes; it is not an independent third-party attestation.

## 8. Remaining integration requirements (not started — Phase 2+)

1. Production encoder (`GameState → (16,10,9)`) and codec with tests in the repo, matching `contracts.py` behavior; verify all legal moves mapped for a much larger position set.
2. Adapter returning `(priors over legal moves, side-to-move value)`; terminal results override NN; fail loud on missing/bad checkpoint, no heuristic fallback.
3. Optional-dependency story for torch (≈201 MB wheels, ≈870 MiB runtime directory) so classic MCTS users are unaffected; checkpoint stays outside git, loaded by path + SHA-256.
4. MCTS lifecycle change (priors at every expanded node, value at leaves) — `MCTSSearch.expand` currently uses uniform `1/num_legal`.
5. Latency budget: at batch 1 CPU, v4 ≈148 evals/s and v4-large ≈50 evals/s per thread (derived from p50); batching/caching are later work.
6. Model choice (§9) and license sign-off.

## 9. Recommendation

- **Primary: `distill-v4-large`** — more sensible value calibration on our engine (initial position +0.126 for Red, not −0.149), better held-out agreement (upstream), identical I/O contract so the adapter is checkpoint-agnostic. Cost: ~3× slower per forward and 2.5× more parameters.
- **Fallback: `distill-v4`** — same code path, ~3× faster; use if the per-move latency budget is tight. Because contracts are identical, choosing between them is a config value, not a re-implementation.
- Decision for the manager: accept the model and the license basis, and set the latency target (the choice between the two follows from it). The v4/v4-large strength difference is not statistically established upstream; the preference for v4-large rests on value calibration and cost, and should be confirmed in Phase 6 by our own benchmark.
- **Not selected: NeymarL/ChineseChess-AlphaZero (CCZero)** — real H5 checkpoint exists but needs TF1/Keras 2.0.8 (Python 3.6 era) or an unverified H5→torch port, a 2086-label orientation-flipping codec, and GPL-3.0 with no separate weight terms. **Not executed this phase**: no download, load, conversion or parity test; only source/config/provenance review. Reserve option only.

## 10. Unresolved / not done

- Real forward pass for CCZero: **not attempted** (would need h5py + a port; GPL review first).
- GPU behavior: untested; CPU-only wheel used.
- Multi-thread/batched latency, memory model-only: unmeasured.
- Strength vs our baselines (UCT+rollout, alpha-beta, tournament agents): **unmeasured**, Phase 6.
- Effect of rule mismatch (repetition/perpetual-check) on value quality: unmeasured.
- Coverage is finite (52 parity states + 12 curated + 20 games). It does not prove parity over all legal Xiangqi positions; Phase 2 should widen it.
- The runtime and source live in `/tmp` and will vanish on reboot; recreate with the commands in the experiment [README](../../experiments/neural_mcts/phase1/README.md).

STOP: Phase 2 has not been started and awaits explicit approval.
