# Phase 0 — Pretrained Xiangqi model shortlist

Checked **2026-11-12**, against public GitHub API metadata and small text sources pinned below. No checkpoint bytes were downloaded, loaded, or executed. HTTP HEAD success verifies an endpoint and advertised length, not checkpoint integrity. The GitHub connector returned an unavailable-tool error; research used web browsing and approved read-only HTTPS requests to GitHub. Network/DNS failed inside the sandbox; approved network execution allowed the metadata reads.

**Recommendation:** qualify **cc-lite distilled v4** first, compare with **CCZero's actual H5 checkpoint**. Treat **Ink's H5-to-PyTorch path** as a conversion research option for CCZero, not a third independently pretrained model. None is certified compatible. This corrects the plan's older characterization of cc-lite as primarily reference architecture: its current pinned tree includes trained checkpoints and provenance.

## Comparison

| Criterion | cc-lite distilled v4 | NeymarL / CCZero | Ink-bai5844 conversion route |
| --- | --- | --- | --- |
| Artifact evidence | **VERIFIED** tracked `.pt`, 15,500,553 bytes; HTTP HEAD 200 | **VERIFIED** tracked `.h5`, 30,290,240 bytes; HTTP HEAD 200 | **VERIFIED** no `.pt`/`.h5` in complete pinned Git tree; expects separately obtained upstream H5 |
| Input from source | float32 16×10×9, absolute board, side/check planes | float32 14×10×9, source canonical state | Reuses CCZero 14-plane encoding |
| Policy from source | 8100 raw logits, `from*90+to` | 2086 softmax probabilities, ordered four-digit labels | 2086 logits internally, softmax + source mask in teacher wrapper |
| Value from source | tanh scalar, side-to-move convention | tanh scalar, canonical current-player convention indicated by source backup | tanh scalar under source canonical convention |
| Runtime | PyTorch; source requires Python >=3.11, torch>=2.1 | TF1/Keras legacy; documented Python 3.6.3 / TF-GPU 1.3 / Keras 2.0.8 | Torch + numpy + h5py; modern port, parity unverified |
| License signal | Apache-2.0 repository; engine-label provenance documented; weight-use scope still needs approval | GPL-3.0 repository; separate weight terms/training provenance not established | No root license in inspected tree; embedded upstream GPL-3.0 license; added conversion code rights unresolved |
| Main obstacle | Rules differ; older-checkpoint BatchNorm caveat; real CPU latency unknown | Legacy runtime + 2086 codec/orientation + rights | Missing bundled weights, rights, conversion parity, heuristic fallback path |
| Phase 0 outcome | **First POC candidate, conditional** | **Second POC candidate, conditional** | **Hold as research reference** |

Matching dimensions/IDs is necessary but does not prove compatibility. All three actual checkpoint forward passes remain **UNKNOWN**.

## A. hskl18/cc-lite — native PyTorch distilled checkpoints

Pinned revision: `7d305c0e5adbfae4e23db40997603d8dc86b078b` (default branch `main`). [Pinned repository](https://github.com/hskl18/cc-lite/tree/7d305c0e5adbfae4e23db40997603d8dc86b078b), [API tree](https://api.github.com/repos/hskl18/cc-lite/git/trees/7d305c0e5adbfae4e23db40997603d8dc86b078b?recursive=1). The API reported a complete, nontruncated tree.

**VERIFIED artifacts and reported hashes:**

| Artifact at pinned revision | Bytes / Git blob SHA-1 | Upstream manifest SHA-256 (not locally verified) |
| --- | --- | --- |
| [distill-v4/train/checkpoint.pt](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/evidence/distill-v4/train/checkpoint.pt) | 15,500,553 / `4441fc8f321b47af203cd69b623589c6ac4bf880` | `1be52664b216529ce56240043406878a93f95f57e4153ff73c4d63afd772fba4` |
| [distill-v4-large/train/checkpoint.pt](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/evidence/distill-v4-large/train/checkpoint.pt) | 38,033,337 / `ded46b0fdbee04aff3f015dd7465a65fb2c48787` | `05927ccd874c4d97588ba7d06266fda9ba17700d436fdc03ff6844a0d61d0dff` |

The first raw artifact's pinned HTTPS endpoint returned HEAD 200 with Content-Length 15,500,553. The complete tree also contains v1/v2/v3 and ablation checkpoints; no need to download all of them. **Exclude `evidence/debug-v1/checkpoint/selfplay.pt` as the proposed useful pretrained model**: upstream describes it as trained on one eight-ply game, a pipeline/debug artifact.

**VERIFIED source contracts:**

- [Encoding](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/src/xiangqi/encoding.py#L7): planes 0–6 red K,A,E,H,R,C,P; 7–13 black in that order; plane 14 all ones for red turn, zeros for black; plane 15 current side in check. Board is row-major, black at top, red uppercase ([board source](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/src/xiangqi/board.py#L6)). The proposed bridge can compute these using the repo's own board/check validator; importing a second rules engine is unnecessary.
- [Network](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/src/model/network.py#L35): residual CNN, policy logits `(B,8100)`, tanh value `(B,)`; [loader](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/src/model/checkpoint.py#L32) uses `weights_only=True`, config from payload, and `model_state`. It does not itself call `eval()`; the evaluator does. Payload content/shape matching remains to be tested.
- [Inference wrapper](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/src/mcts/search.py#L45) selects legal IDs before softmax and defines value from board.turn. Copying that engine's backup is unsafe: its node value convention differs from this repo's mover-perspective Q.
- [Training manifest](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/evidence/distill-v4/train/manifest.json) identifies source commit `0b9a2b68179d42b52160de7c8d583c16fabf8e7f`, seed 42, Python 3.12.3, Torch 2.11.0+cu128, data/config/checkpoint hashes. This is upstream provenance, not a reproduced run. [v4 report](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/evidence/distill-v4/README.md) describes engine-distilled weights and their limitations. No training is needed to consume those existing weights.

**Risks:** [DESIGN](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/DESIGN.md) describes threefold with perpetual-check adjudication and 120 plies without capture **or pawn move**. Our repo draws all threefold repetitions and resets only on capture. Retain our terminal rules. The 16 planes omit repetition/counter context, so identical boards can have different engine outcomes. The source [README](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/README.md#model) also flags old BatchNorm statistics under-reporting value in eval mode; v4's report discusses that issue. Do not silently recalibrate, fine-tune, or replace the pinned checkpoint in qualification.

**License/provenance boundary:** [LICENSE](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/LICENSE) / [pyproject](https://github.com/hskl18/cc-lite/blob/7d305c0e5adbfae4e23db40997603d8dc86b078b/pyproject.toml) state Apache-2.0. DESIGN says labels use Fairy-Stockfish and discusses CC0 Xiangqi network provenance instead of restricted Pikafish weights. This is an upstream explanation, not an independent rights clearance for every data/model derivative. Confirm intended weight redistribution/use before selection. No third-party code was copied into this repository.

**INFERRED:** closest adapter fit because piece order, absolute board layout, scalar perspective, and 8100 IDs align by inspection. CPU POC still must prove tensors, complete legal mapping, real load, resource use, and useful finite outputs. Upstream game results were not replayed and cannot establish strength against our agents.

## B. NeymarL/ChineseChess-AlphaZero — CCZero H5

Pinned revision: `7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f` (default branch **distributed**, not an assumed master). [Pinned repository](https://github.com/NeymarL/ChineseChess-AlphaZero/tree/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f), [API tree](https://api.github.com/repos/NeymarL/ChineseChess-AlphaZero/git/trees/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f?recursive=1).

**VERIFIED artifact metadata:** [model_best_weight.h5](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/data/model/model_best_weight.h5), 30,290,240 bytes, Git blob SHA-1 `1d87c882be5ebb490f7fb07ba4c306601a558726`. Pinned raw endpoint HEAD returned 200 and the same length. Its HTTP ETag is not treated as a checkpoint SHA-256. Local checkpoint SHA-256, HDF5 contents, trained-weight validity, and playing strength are **UNKNOWN**.

**VERIFIED small configuration:** [model_best_config.json](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/data/model/model_best_config.json) was read (39,454 bytes); local SHA-256 of this **JSON only** is `20d7f417cf56a1935db3f252493c48c7d62ddd721569bbbd71ca87b460f435be`. It specifies input `[None,14,10,9]`, softmax policy 2086, tanh value 1, value dense 256. [Model source](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/cchess_alphazero/agent/model.py#L29) provides residual blocks and load_weights.

**Encoding:** [lookup tables](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/cchess_alphazero/environment/lookup_tables.py#L25) use P,C,R,horse,elephant,advisor,king plane order, unlike engine numeric IDs. Source FEN uses `k` for horse and `s` for king. Policy labels are `[source_col,source_row,target_col,target_row]` strings, not dense from-to IDs; `flip_move` maps x→8−x and y→9−y. [static_env](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/cchess_alphazero/environment/static_env.py) builds 14 float32 planes and flips board/case after a move. Its state string row order and its internal board coordinates differ; do not translate by index alone. [Player backup](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/cchess_alphazero/agent/player.py#L323) indicates a current-player value convention, still requiring both-color parity on actual outputs.

**Runtime/rights:** [README](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/README.md) documents the legacy versions; [requirements](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/requirements.txt) and [GPL-3.0 license](https://github.com/NeymarL/ChineseChess-AlphaZero/blob/7f45b0cd7470f7f8e1da7331fc1a31374d6cc10f/LICENSE) were inspected. No separate weight-use grant or sufficient training-data provenance was established. Compatibility with this project's intended distribution needs approval. Do not use its automatic self-play/model-creation entry points, which can initialize random weights when absent. If choosing a port or ONNX conversion, require output parity with a trustworthy source inference path; source-level transposes alone are insufficient proof.

## C. Ink-bai5844/ChineseChess-AI — H5 inference port, not independent weights

Pinned revision: `10f2563e3daeee022b2fa05a732e565415ec3f58` (`main`). [Pinned tree](https://github.com/Ink-bai5844/ChineseChess-AI/tree/10f2563e3daeee022b2fa05a732e565415ec3f58), [API tree](https://api.github.com/repos/Ink-bai5844/ChineseChess-AI/git/trees/10f2563e3daeee022b2fa05a732e565415ec3f58?recursive=1).

**VERIFIED:** complete tree has no H5/PT checkpoint despite README's expected `ChineseChess-AlphaZero/data/model/...` paths. No independent student checkpoint was verified. [OriginalH5Teacher / OriginalAlphaZeroNet](https://github.com/Ink-bai5844/ChineseChess-AI/blob/10f2563e3daeee022b2fa05a732e565415ec3f58/distill_cchess_alphazero.py#L207) define Torch inference: infer channels/blocks from H5, transpose convolution HWIO→OIHW and dense kernels, copy BN statistics with eps=1e-3, `eval()` and inference_mode, then source legal masking. The separately passed JSON config is not used to reconstruct this port's network; some dimensions are fixed. Check flatten ordering, every tensor, source configuration, and numerical parity before trusting it.

**VERIFIED fallback concern:** [embedded model.load](https://github.com/Ink-bai5844/ChineseChess-AI/blob/10f2563e3daeee022b2fa05a732e565415ec3f58/ChineseChess-AlphaZero/cchess_alphazero/agent/model.py#L250) catches Keras errors, installs a material/heuristic model, and returns true. The dedicated OriginalH5Teacher advertises no fallback, but blindly accepting the generic load success would violate our requirements. Any POC must assert the actual backend and loaded artifact.

**License:** GitHub metadata has no root license; only [embedded upstream GPL-3.0](https://github.com/Ink-bai5844/ChineseChess-AI/blob/10f2563e3daeee022b2fa05a732e565415ec3f58/ChineseChess-AlphaZero/LICENSE) was located. Rights for newly added conversion code and independently supplied weights remain **UNKNOWN**. Do not copy the port on the assumption it is permissively licensed. Its training/distillation commands are outside our scope.

## Qualification boundary

Phase 0 established useful candidates and source contracts; it did **not** satisfy Phase 1. After approval, select permitted artifacts and runtime, fetch one at a time, verify SHA-256 against a pinned manifest where available, load on CPU, run deterministic both-color samples, compare source outputs/mapping, report load/forward time and RSS, and stop with a model GO/NO-GO. Approximate checkpoint downloads are 14.8 MiB (cc-lite v4) and 28.9 MiB (CCZero); framework installation may be much larger and needs a concrete resource plan before execution. Neither download latency nor inference cost has been measured here.

Selected GitHub metadata fields are retained in [model-metadata.json](evidence/model-metadata.json), with [HTTP header observations](evidence/checkpoint-headers.json). Hashes labeled Git blob SHA-1 or upstream manifest SHA-256 are not local verification of checkpoint bytes.
