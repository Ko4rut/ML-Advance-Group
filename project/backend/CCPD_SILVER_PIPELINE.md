# CCPD Silver Pipeline

## Scope and current status

- Source: **CCPD**, real PGN from `01_bronze/ccpd`. Master-match data is encoded in Big5 (other UTF-8/CP950 sources are accepted if valid).
- Output: validated/deduplicated `02_silver` JSONL, **not** Gold/training data.
- Parsing: headers including FEN and recorded match result; Chinese Xiangqi notation resolved only against unique legal moves from `GameEngine`.
- Sidecar: `<file>.pgn.manifest.json` required by default; verify `file_sha256` against the original Bronze file. Source version and license are inherited from the manifest.
- Keep all Bronze files immutable. Unknown/illegal notation, malformed PGN, manifest mismatch, and contradictory results are quarantined.

### Layout

```text
project/backend/
├── src/data/02_silver/
│   ├── models.py                          # result_source, recorded_game_end
│   └── data_pipeline/
│       ├── replay.py                      # recorded versus engine outcome
│       └── ccpd/
│           ├── decoder.py                 # Big5, UTF-8, CP950
│           ├── pgn_parser.py              # FEN, headers and movetext
│           ├── notation.py                # Chinese move -> legal action
│           ├── adapter.py                 # SilverGame, human source labels
│           └── pipeline.py                # replay, dedup, quarantine, report
├── scripts/build_ccpd_silver.py
├── notebooks/xiangqi_colab_ccpd_silver.ipynb
└── tests/data_pipeline/test_ccpd_pipeline.py
```

## Run locally

From `project/backend`:

```bash
python -m scripts.build_ccpd_silver \
  /path/to/XiangqiAI/data/01_bronze/ccpd \
  --output-dir /path/to/new/ccpd_run_001
```

The destination run must be new. By default the pipeline requires one verified manifest for each PGN.

For exploratory parsing of non-Bronze test PGNs only:

```bash
python -m scripts.build_ccpd_silver /path/to/test.pgn \
  --output-dir /path/to/test_silver_001 --allow-missing-manifest
```

Do not use `--allow-missing-manifest` for production or training source material.

For Google Colab, open `notebooks/xiangqi_colab_ccpd_silver.ipynb`, mount Drive,
run `MODE="sample"` (20 files, fixed seed 42), inspect `report.json` and
`quarantine.jsonl`; when satisfied, change to `MODE="full"` (all 595 PGNs).
The notebook stages source data in `/content` and only publishes results after
copying and checking SHA-256 against the local output.

## Output

- `games.jsonl`: canonical SilverGame records (accepted games only)
- `transitions.jsonl`: one transition per valid **ply**
- `duplicates.jsonl`: duplicate game IDs / SHA-256 keys
- `quarantine.jsonl`: invalid source, error code, optional ply number, preview
- `report.json`: counts, encodings, errors, and recorded/engine results

`GameOutcome.result_source="recorded"` means the PGN result is a match result,
not a verified checkmate. `SilverTransition.recorded_game_end=True` marks the
last move in a completed PGN, while `terminated=True` is reserved for a
terminal position detected by `GameEngine`. The `recorded_reward` field
remains `null` (Gold will handle ML-specific reward/labels). Alpha-Beta
self-play keeps `result_source="engine"` by default and its stricter replay
outcome checks are unchanged.

Professional match records may continue after threefold repetition or long
no-capture sequences, which are project-specific self-play termination rules.
The CCPD pipeline uses a separate GameEngine with high counter thresholds
(`repetition_count` and `no_capture_limit`), but **does not weaken
piece movement legality, check protection or flying-generals rules**.

## Limitations and quality gates

Chinese notation forms that cannot be identified uniquely are sent to
quarantine. Parsing errors should be fixed in adapters, **not by editing Bronze**.
Full CCPD compatibility has not yet been established by a complete 595-game
run on Google Colab.

Verification performed during implementation: 18 actual PGNs from
World Mind Sports Games were replayed end-to-end with 1,697 transitions,
0 duplicates and 0 quarantine; 11 focused synthetic integration tests
cover Big5, FEN, legal moves, manifests, outcomes, deduplication and failure cases.
These local tests used a compatible local engine mirror; the new Colab notebook
is the recommended place to run against the exact checked-out repository backend.
Do not interpret sample pass rates as a guarantee about the full collection.

## Attribution

CCPD: Yu-Han Tseng and Bo-Nian Chen (2026).
https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset
Licensed under **CC BY 4.0**; preserve proper attribution when redistributing results.
