# Phase 1 — Model candidates and license review

Checked 2026-11-12. Repository HEAD: `c75f0597f859e32eea8c55e70c8e957a2c517b40`, branch `agent/mcts_nn`. **Selected technical candidates: cc-lite `distill-v4-large` (primary) and `distill-v4` (faster fallback).** Both real checkpoints were downloaded, hash-verified and run on CPU (v4 by Codex and re-run by Claude Code; v4-large by Claude Code). CCZero was compared using pinned source/config/provenance, not executed. Claude Code revised the original v4-only recommendation; see the decision below and the [POC report](01_POC_REPORT.md). No training, distillation, recalibration, conversion, or neural MCTS integration ran.

## 1. cc-lite distill-v4 — qualified on CPU

- Source: [hskl18/cc-lite at 7d305c0](https://github.com/hskl18/cc-lite/tree/7d305c0e5adbfae4e23db40997603d8dc86b078b).
- Checkpoint: [evidence/distill-v4/train/checkpoint.pt](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/evidence/distill-v4/train/checkpoint.pt).
- **VERIFIED locally:** 15,500,553 bytes; SHA-256 `1be52664b216529ce56240043406878a93f95f57e4153ff73c4d63afd772fba4`, matching the [upstream training manifest](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/evidence/distill-v4/train/manifest.json).
- **VERIFIED actual payload:** 128 channels, 8 residual blocks, 16 input planes, 8100 policy outputs; 3,860,347 parameters. `step=3320673`; metadata records 3,320,673 samples, seed 42, and config/data hashes that match the upstream manifest. These are source/payload provenance, not an independently reproduced training run.
- Loader: original pinned [checkpoint.py](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/src/model/checkpoint.py), strict state loading, `weights_only=True`, CPU map, explicit eval/inference mode. Every executable downloaded source file is hash-checked before import. The original architecture is used directly, so no cross-framework conversion claim is needed.
- Runtime actually used: Python 3.14.9, PyTorch 2.11.0+cpu, numpy 2.5.3, one CPU thread. Optional runtime installed only under `/tmp`, not in shared `.venv` or `pyproject.toml`.
- Native contracts and exact fixture results: [compatibility matrix](01_COMPATIBILITY_MATRIX.md), [POC report](01_POC_REPORT.md), [raw results](evidence/phase1/result.json).

## 1b. cc-lite distill-v4-large — qualified on CPU (added by Claude Code)

- Checkpoint: [evidence/distill-v4-large/train/checkpoint.pt](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/evidence/distill-v4-large/train/checkpoint.pt) — **VERIFIED locally** 38,033,337 bytes, SHA-256 `05927ccd874c4d97588ba7d06266fda9ba17700d436fdc03ff6844a0d61d0dff` (matches the pinned upstream manifest; registered in `artifacts.json`).
- **VERIFIED payload:** 192 channels, 12 blocks, same 16 planes / 8100 logits / scalar tanh value contract as v4; 9,484,219 parameters; same data (3,320,673 samples), more capacity.
- Passed the identical `qualify.py` assertions as v4 (exact observation parity, exact legal-set parity, prior error ≤4.5e-8, value error 0, weights unchanged). Warm CPU forward p50 **20.0 ms** (v4: 6.75 ms); whole-POC peak RSS ~462 MiB.
- Upstream claims (not reproduced here): 36.7 % held-out policy agreement vs 33.6 % for v4; 14/60 vs 5.5/60 against Pikafish; paired v4-large vs v4 score 0.594 with CI 0.458–0.719 (no demonstrated difference); value-sign agreement 96.5 % vs v4's 90.9 %. Both predate the BatchNorm-statistics fix.
- Our observation: initial-position value for Red-to-move is +0.126 (v4: −0.149), i.e. a more plausible calibration.

## 2. NeymarL/ChineseChess-AlphaZero — comparison option, not qualified

Source: [pinned 7f45b0c](https://github.com/NeymarL/ChineseChess-AlphaZero/tree/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f). [H5 checkpoint](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/data/model/model_best_weight.h5): metadata/HEAD length 30,290,240 bytes; no checkpoint bytes fetched in this phase. Local weight SHA-256, actual load, performance and conversion parity remain **UNKNOWN**.

The [JSON config](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/data/model/model_best_config.json) and [architecture](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/cchess_alphazero/agent/model.py) specify 14×10×9 float32 input, 2086 softmax outputs and one tanh value. Native planes order pawn/cannon/rook/horse/elephant/advisor/king; current player is canonicalized with board/color flips. [Action labels](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/cchess_alphazero/environment/lookup_tables.py) are four-digit coordinate strings in a generated order, not Gold IDs. It needs a verified native codec and both-color/source parity.

The pinned [requirements](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/requirements.txt) require TF-GPU 1.3.0, Keras 2.0.8 and h5py 2.7.1, conflicting with the current runtime by inspection. A legacy environment or conversion would add installation work and require numerical parity. Download lower bound is the 30.3 MB checkpoint; actual framework/storage/memory/latency is unmeasured. No speculative latency estimate is presented.

The repository is [GPL-3.0](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/LICENSE); separate weight terms and training-data provenance were not established. It is not selected because cc-lite removes the legacy conversion and native-codec obstacles and provides useful manifest evidence.

## 3. Ink conversion path — not adopted

[Pinned Ink revision](https://github.com/Ink-bai5844/ChineseChess-AI/tree/10f2563e3daeee022b2fa05a732e565415ec3f58) has no independently verified PT/H5 checkpoint in its complete tree. It offers an H5-to-Torch teacher port, but added-code licensing is unresolved and generic load success can mean a heuristic fallback. Neither source copying nor port execution occurred. See the [Phase 0 evidence](00_MODEL_SHORTLIST.md) for exact files. This is a possible implementation route for CCZero, not an independently qualified third model.

## License/provenance assessment for this POC

**VERIFIED:** the full pinned cc-lite [LICENSE](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/LICENSE) states Apache-2.0, copyright 2026 The cc-lite authors. The repository manifest includes the checkpoint under that repository; no separate weight restriction was found in the inspected license, README, DESIGN or checkpoint directory records. A root NOTICE URL returned 404; no root NOTICE appeared in the complete source tree inspected in Phase 0.

**INFERRED engineering assessment:** the repository-wide license and absence of a stated weight exception support isolated research inference with the published checkpoint. This assessment permits this POC; it is not a claim that the manager separately approved production redistribution or that every upstream rights claim has been independently proven. No third-party model/source was vendored or redistributed in our repository; the external artifact directory retains the original license.

The provenance chain is better than a bare checkpoint link: [cc-lite DESIGN](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/DESIGN.md) and [distill-v1 records](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/evidence/distill-v1/README.md) identify Fairy-Stockfish engine labels, with the Xiangqi `c07e94a5c7cb` network. The [official network authors](https://github.com/official-pikafish/Networks#nnue-license) explicitly distinguish restricted Pikafish weights from their CC0 Fairy-Stockfish Xiangqi weights. The [named Fairy-Stockfish release](https://github.com/fairy-stockfish/Fairy-Stockfish-NNUE/releases/tag/xiangqi-c07e94a5c7cb) exists and identifies that built-in network. We did not fetch the engine/data or regenerate labels; exact training reproducibility remains unverified.

Before any later redistribution, retain the Apache license/copyright/attribution, mark modified source, and preserve applicable NOTICE content if present in the distributed upstream material. Do not import restricted Pikafish weights or unlicensed Ink code by analogy. **Manager acceptance of cc-lite as the model for subsequent phases, including the license assessment, remains a concrete review decision.**

## Decision

**Technical qualification: PASS for cc-lite `distill-v4` and `distill-v4-large`.** Recommended: **v4-large as primary** (better value calibration observed on our engine, better upstream held-out agreement), **v4 as the low-latency fallback** (~3× faster). Both share one I/O contract, so the choice is a config value. **GO to Phase 2 only after explicit manager approval of the model(s) and the documented license basis**, plus a latency target. Neither is a strong player (upstream: both lose most games to Pikafish; our own check: 17W/3D/0L policy-only vs random for each) and no playing-strength claim against our agents is made. Keep CCZero as an un-executed reserve.
