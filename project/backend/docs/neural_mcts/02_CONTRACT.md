# Phase 2 — Encoding contract and verification

- Repository HEAD at start: `868d8c5` (Phase 1 committed), branch `agent/mcts_nn`. Nothing committed in this phase.
- Scope: observation encoder, action codec, legal-mask/normalization, contract document, tests. **Not** in scope and not touched: model loading/inference service (Phase 3), MCTS/PUCT/agent changes (Phase 4), API/registry (Phase 5), `src/search`, `src/agents`, `src/api`, dependencies.
- Models covered: cc-lite `distill-v4-large` (primary) and `distill-v4` (fallback), approved in Phase 1. Both share this one contract.
- Tags: **VERIFIED** = observed in a command run this phase; **INFERRED**; **DECISION**; **UNRESOLVED**.

## 1. The contract

All definitions below are implemented in `src/encoding/` (pure standard library; importing it never imports numpy/torch — tested in a subprocess).

### 1.1 Observation — `ObservationEncoder` ([observations.py](../../src/encoding/observations.py))

| Item | Definition |
| --- | --- |
| Name | `absolute_planes16_v1` |
| Shape / dtype | `(16, 10, 9)`, float32, plane-major then row-major; returned as a flat `array.array('f')` of 1440 values (zero-copy into numpy/torch via `frombuffer`); `encode_batch` returns a flat buffer for `(B,16,10,9)` |
| Board coordinates | The engine's own: index `row*9+col`, **row 0 = Black's back rank**, row 9 = Red's. No flipping or rotation, ever |
| Planes 0–6 | Red king, advisor, elephant, horse, rook, cannon, pawn (= engine `abs(piece)-1` for positive pieces) |
| Planes 7–13 | Black, same order (`abs(piece)-1+7` for negative pieces) |
| Plane 14 | all 1.0 if **Red** to move, all 0.0 if Black to move |
| Plane 15 | all 1.0 if the **side to move** is in check (rule validator), else 0.0 |
| Not encoded | repetition history, `no_capture_count`, `ply_count`. States differing only in those encode identically (tested) |
| Perspective | **Absolute.** The board is not rotated for Black; only plane 14 carries side to move |

`encode(state, in_check=None)`: a caller may pass an already-known check flag; it is then trusted (caller's responsibility). Optional `to_numpy` / `batch_to_numpy` (lazy numpy import) return writable float32 copies.

### 1.2 Actions — `ActionEncoder`, `CCLiteVocabulary` ([actions.py](../../src/encoding/actions.py))

Two integer spaces, deliberately separate types:

| Space | Type | Definition |
| --- | --- | --- |
| Internal Gold action ID, `from_to_8100_v1` | `ActionEncoder` | `source*90 + target`, `[0, 8100)`; same as Gold pipeline and tabular Q. Size checked equal to Gold's `ACTION_SPACE_SIZE` |
| Native policy index of the cc-lite checkpoints | `CCLITE_VOCABULARY` (`to_native`, `from_native`) | 8100 logits; **currently numerically identical** to the Gold ID |

The identity is a *verified fact about this checkpoint*, not an assumption: every legal move in 472 golden states equals the upstream codec's legal index set (§3). Code that talks to a model must use the vocabulary object, so a checkpoint with another policy layout (e.g. CCZero's 2086 labels) can be added without touching the Gold contract. `decode`/`from_native` are structural inverses only; they accept any in-range index (including `source == target`), and **legality is established only by the engine's legal-move list**.

### 1.3 Policy → legal priors — `legal_priors(scores, moves, kind=...)`

1. Input: native scores, one per index (list/array/ndarray), raw **logits** (the cc-lite output) or probabilities.
2. All 8100 entries must be finite (a NaN on an illegal slot still means a corrupt model output → `ValueError`).
3. Only the legal moves' entries are used. Logits → numerically stable softmax over legal entries; probabilities → renormalized (negative probability → error).
4. Output: `{Move: prior}` over exactly the supplied legal moves; each prior finite, ≥ 0; sum = 1 within 1e-12 (`math.fsum`).
5. **No hidden fallback.** Wrong length, non-finite, empty or duplicated legal list, zero legal probability mass → `ValueError`. All-zero logits give an explicit uniform distribution (that is the softmax, not a fallback).
6. Illegal moves can never appear in the output, regardless of their score (tested with a 1e100 logit on an illegal slot).

### 1.4 Value and terminal rules (documented here; enforced and tested in Phase 3)

- **Value** = network scalar, tanh range `[-1, 1]`, **from the side to move's perspective**, no color-dependent sign flip (Phase 1 §5). Reject non-finite or out-of-range values.
- **Terminal precedence:** if the engine says the state is terminal (checkmate/stalemate loss, threefold repetition draw, 120-ply no-capture draw), use the engine's result (`GameResult.reward(side_to_move)`: −1 / 0) and **do not call the network**. The network never overrides the engine.
- MCTS backup keeps the repo's existing per-ply sign flip; nothing here changes it.

### 1.5 Known rule/feature mismatches (carried from Phase 1; unchanged)

The checkpoints were trained under a rule profile with perpetual-check adjudication and a no-capture counter that pawn moves reset; our engine draws every threefold repetition and resets only on capture. The 16 planes contain no history. Engine results always win; value quality in positions the source engine would have adjudicated differently is unmeasured.

## 2. Files added / changed

| File | Purpose |
| --- | --- |
| `src/encoding/observations.py`, `actions.py`, `__init__.py` | Production contract (replaced `pass` skeletons that had no callers — verified by grep) |
| `tests/encoding/test_observations.py`, `test_actions.py`, `test_golden_parity.py`, `__init__.py` | 43 tests, torch-free |
| `tests/encoding/golden/cclite_v1.json` | 472 golden states (271 KB), expected values from the **upstream** encoder/codec |
| `experiments/neural_mcts/phase2/generate_golden.py` | Deterministic generator (re-run is byte-identical); verifies upstream source hashes first |
| `experiments/neural_mcts/phase2/test_real_checkpoint.py` | Real-checkpoint parity of the production stack (explicit skip without env) |
| `docs/neural_mcts/02_CONTRACT.md`, `evidence/phase2/` | This document and raw logs |

No change to `src/environment`, `src/search`, `src/agents`, `src/api`, `pyproject.toml`, or any existing test.

## 3. Verification (all commands executed)

Run from `project/backend`; `PY=../../.venv/bin/python`; Python 3.14.9. Raw logs: [evidence/phase2/](evidence/phase2/).

| # | Command | Result |
| --- | --- | --- |
| 1 | `$PY -m unittest discover -s tests/encoding -t . -v` | **43 tests OK**, 0 skipped ([log](evidence/phase2/encoding-tests.log)) |
| 2 | same, with `import numpy` forced to fail (`sys.modules['numpy']=None`, `/usr/bin/python3`; [log](evidence/phase2/encoding-tests-numpy-blocked.log)) | OK, 43 run, **3 skipped** (the numpy-dependent ones, each with a reason). The 40 others pass without numpy |
| 3 | Mutation checks ([script](evidence/phase2/mutation_checks.py), [log](evidence/phase2/mutation-checks.log)): swap colors, swap rook/cannon planes, invert turn/check planes, transpose vocabulary, transpose `ActionEncoder`, never set check plane | **All 6 detected** (341–939 failing tests each); baseline 0 failing. Shows the tests can fail |
| 4 | Real checkpoint through the production stack, `XIANGQI_PHASE1_*` set, `$PY -m unittest discover -s experiments/neural_mcts/phase2 -p test_real_checkpoint.py -v` | `distill-v4`: **2 OK**, 467 non-terminal states / 17,066 legal moves; `distill-v4-large`: **2 OK**, same coverage ([v4](evidence/phase2/real-checkpoint-distill-v4.log), [large](evidence/phase2/real-checkpoint-distill-v4-large.log)). Without the env vars the 2 tests report **skipped** |
| 5 | `PYTHONPATH=/tmp/neural-mcts-phase0/deps:$REPO $PY -m unittest discover -s tests -v` (full backend) | **313 tests OK** = 270 baseline + 43 new ([log](evidence/phase2/backend-full-suite.log)) |
| 6 | `generate_golden.py` run twice | byte-identical output; 0 engine-vs-upstream legal-set mismatches |

What the checks establish (VERIFIED):
- **Observation:** production encoder bytes equal the upstream `encode_board` bytes (SHA-256) for all 472 golden states, both sides to move; the check plane equals upstream's flag.
- **Action mapping:** for all 472 states (17,066 legal moves) the engine's legal IDs equal the upstream codec's legal indices, and every upstream index decodes to an engine-legal move. Coverage includes cannon capture over one screen / with no screen / with two screens, horse leg blocked by own and enemy pieces, elephant eye blocked and river limit, advisor/king palace limits, king capturing an adjacent piece, flying-general pins, check by cannon/horse/rook with 2 evasions, stalemate, pawn movement before/after the river and on the last rank for both colors (each special position also color-swapped), mate-in-one and checkmated fixtures, and 6 seeded 80-ply random games. This **also validates our engine's move generator against an independent implementation** on those states.
- **Real inference:** with the real weights, production-encoded inputs yield priors within 1e-6 of upstream's `TorchEvaluator` and identical values on all 467 non-terminal states, for both checkpoints; a batched production encoding gives the same logits as single-state encoding.
- **Cost:** encode ≈ 0.046 ms/state (0.007 ms when the check flag is supplied), `legal_priors` ≈ 0.22 ms, `legal_mask` ≈ 0.023 ms, versus a 7–20 ms forward pass. Negligible against the 3 s/move budget.

## 4. Findings and limits

- **Finding (fixture bug, not an engine bug):** my first draft of one hand-built position had both kings facing on an open file. Upstream allows the flying-general capture there; our engine does not. Such a position cannot arise in legal play (a move that exposes the king is illegal), so it is outside the contract; I corrected the fixture. Positions with facing generals are **unsupported input** for this encoder/codec pairing.
- **Coverage is finite:** 472 states (48 states from 24 hand-built rule positions × 2 colors, 10 Phase 1 fixtures [the 2 history-only draw fixtures are excluded: they have the same board as the initial position], and 414 states from 6 seeded random games, one of which ended early), not all Xiangqi positions. It does not prove rule equivalence for rarer geometry; it does show no divergence on this broad set.
- The golden fixture's expected values come from the pinned upstream revision `7d305c0e5adbfae4e23db40997603d8dc86b078b`; a different upstream revision or checkpoint trained on another layout would need regeneration and review.
- The in-check plane is computed by our engine's validator; the golden check flag (from upstream's encoder) agrees on all 472 states.
- Value perspective/range and terminal precedence are specified (§1.4). *Update (Phase 3):* now implemented and tested in `src/networks/pretrained/adapter.py`; see [03_MODEL_CARD.md](03_MODEL_CARD.md).
- No MCTS ran; no strength claim.

## 5. Gate and next step

**Phase 2 gate: PASS.** Deterministic encoding (byte-identical reruns), exact mapping with 100 % of golden legal moves mapped (17,066/17,066), shape/dtype/legal-mask/normalization/degenerate-input tests, color and side-to-move coverage, checkpoint parity on both approved models, and an unchanged baseline (313/313).

Suggested Phase 3 (needs your explicit start): production `PolicyValueAdapter` — lazy optional-torch loader keyed by path + SHA-256 (fail loud, no fallback, no checkpoint in git), CPU inference with `predict(state, legal_moves) -> (priors, value)` using this encoder/vocabulary, terminal-precedence and value-range checks, fake-model unit tests plus a real-checkpoint integration test, a batched predict for later leaf batching, a model card (`03_MODEL_CARD.md`) and a demo script. Decision for you then: where the checkpoint file is kept and how its path is configured on each team machine.
