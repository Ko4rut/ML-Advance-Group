# Phase 1 — Compatibility matrix

Date: 2026-11-12 (revised by Claude Code). The cc-lite column describes `distill-v4`; `distill-v4-large` has an **identical contract** (same planes, 8100 logits, scalar tanh side-to-move value) and passed the same parity checks; only size/latency/calibration differ (see below). **VERIFIED** means code/artifact/observed output; **INFERRED** means proposed implementation or license interpretation; **UNKNOWN** means not executed or established.

| Dimension | Current engine | cc-lite v4: observed qualification | CCZero: source-only comparison |
| --- | --- | --- | --- |
| Board | 90 row-major cells, row 0 black side | Exact absolute layout; encoder parity over all 52 tested states | 10×9, canonical state string and internal coordinates require conversion |
| Piece identity | IDs 1 K,2 A,3 E,4 H,5 R,6 C,7 P; red+, black− | Planes 0–6 red K/A/E/H/R/C/P, 7–13 black; exact match after experimental encoding | Per color P/C/R/horse/elephant/advisor/king; source `k` means horse, `s` king |
| Input | GameState plus history/counters | `(B,16,10,9)`, float32; plane 14 red-turn 1/black-turn 0, plane 15 current-side check; no rotation | `(B,14,10,9)`, float32; board/case flips; history variant exists but is not this config |
| Policy | Move, Gold ID source×90+target | Actual `(B,8100)` raw logits; no conversion or head replacement | Config `(B,2086)` softmax probabilities; native ordered labels required |
| Legal mapping | Repo GameEngine authoritative | Exact equality with source codec/legal sets on 12 fixtures + 40 sampled states: **1984 move occurrences mapped** | Native vocabulary exists; no executable bridge or coverage proof here |
| Legal masking | Must exclude illegal moves | Stable softmax over engine-legal IDs only; illegal scores excluded; finite/shape/duplicate checks | Requires source-native probability masking and renormalization |
| Degenerate scores | No hidden neural fallback | Zero logits → uniform legal policy; zero legal probability mass/NaN/Inf/unknown index → explicit ValueError | Not exercised |
| Value | Search evaluates side-to-move; node Q uses previous mover | Actual `(B,)`, tanh in [-1,1]; source evaluator defines side-to-move and wrapper preserves it without color sign changes | Tanh `(B,1)`; canonical current-player convention inferred from source backup; not measured |
| Terminal precedence | No-move loss, threefold draw, 120 non-capture ply draw | Engine result bypasses NN; both-color checkmate → -1 and draw → 0; forward not invoked | Not exercised with repo engine |
| Rule mismatch | All repetitions draw; pawn move alone does not reset count | Source adjudicates perpetual check and resets no-progress on pawn moves; omitted from NN input. Source history is not used to replace engine rules | Different source rules/canonicalization also need a model card |
| Checkpoint validity | No policy-value checkpoint in production | Actual hash matches upstream manifest; original definition strict-loads real payload; all weights/BN buffers unchanged during POC | Header/artifact metadata only; no load/hash parity |
| Runtime | Python >=3.10, no ML requirement | Python3.14.9 + Torch2.11.0+cpu + numpy2.5.3 works in isolated target; source declares Python>=3.11 | TF1/Keras2/h5py2.7 legacy; current compatibility UNKNOWN |
| Source parity | Required before formal adapter | Observations exact; fixture legal-prior max error **2.21e-8**; value error **0** against original TorchEvaluator; sampled trajectory tolerance ≤1e-6 | Conversion/source parity UNKNOWN; no port copied/executed |
| CPU resource observation | i5-13420H audit host | Warm single-state forward p50 **6.88 ms**, p95 **7.37 ms** (20 calls); whole-POC peak RSS ~**378 MiB** | UNKNOWN; at least30.3 MB checkpoint plus a separate runtime/conversion path |
| Installation/artifacts | Core dependencies retained | ~201.1 MB runtime wheels +15.5 MB checkpoint; runtime directory869 MiB; source/artifact directory~15 MiB | Weights not downloaded; no second runtime installed |
| GPU | Host RTX3050 4 GiB/driver615.78.08 verified outside sandbox | CPU-only build and CPU inference by user choice; GPU performance/compatibility untested | Not tested |
| Rights | No external adoption yet | Apache-2.0 repository, documented Fairy-Stockfish/CC0 provenance; isolated POC supported by engineering assessment; manager adoption sign-off pending | GPL-3.0 repository; independent weight/data terms unresolved |
| Decision | Preserve classic search/API | **Technical PASS; recommend selection subject to approval** | **Not qualified; reserve option** |

Coverage is finite, not a proof over all possible Xiangqi positions. Source parity establishes that our experimental bridge matches the source on tested states; it does not prove engine rule equivalence, value calibration, or playing strength. The raw NN is not exactly invariant under color rotation, and its top policy move does not always choose an available mate; see [results](01_POC_REPORT.md).

No cross-framework conversion was used. Production encoding, inference service, neural search, batching/cache design, and API enablement remain later phases. Sources and licensing details are in [candidates](01_MODEL_CANDIDATES.md); reproduction and raw evidence are in [POC report](01_POC_REPORT.md).

## distill-v4 vs distill-v4-large (VERIFIED, same harness, same 52 states)

| Dimension | distill-v4 | distill-v4-large |
| --- | --- | --- |
| Architecture (from payload) | 128 ch, 8 blocks, 3,860,347 params | 192 ch, 12 blocks, 9,484,219 params |
| Checkpoint SHA-256 | `1be52664…fba4` | `05927ccd…0dff` |
| Max legal-prior error vs upstream wrapper / value error | ≤2.2e-8 / 0 | ≤4.5e-8 / 0 |
| Warm CPU forward p50 (1 thread, batch 1) | 6.75 ms | 20.0 ms |
| Value, initial position, Red to move | −0.149 | +0.126 |
| Value sign when a side is up material (both colors) | correct | correct |
| Policy-only vs random, 10 games/side | 17W 3D 0L | 17W 3D 0L |

## Value-sign and strength evidence added in Phase 1 (see [POC report](01_POC_REPORT.md) §4–§5)

- Value is **side-to-move** (upstream label definition + measured on both colors): no color sign flip at the adapter.
- Raw policy does not reliably prefer a mate-in-one; engine terminal detection must stay authoritative.
- Both checkpoints predate upstream's BatchNorm-statistics fix; value magnitudes may be under-reported (v4 worse).
