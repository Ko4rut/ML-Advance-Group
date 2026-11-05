# Silver pipeline (Alpha-Beta v1)

The Silver layer provides source-neutral, replay-validated games and transitions.
It consumes immutable Bronze JSONL files and does not generate new self-play.

## Responsibility map

- src/data/bronze/reader.py: typed raw JSONL input (standalone).
- src/data_pipeline/silver/adapters.py: SourceAdapter interface, AlphaBetaAdapter.
  Keep source-specific notation converters here; future CCPDAdapter must
  implement the same convert/audit contract.
- src/data_pipeline/silver/replay.py: SilverReplayValidator, replays moves
  using the project's existing GameEngine, checks result, emits transitions.
- src/data_pipeline/silver/deduplication.py: canonical SHA256 of initial
  board, side, and action sequence. Ignore source-provided fingerprints.
- src/data_pipeline/silver/writers.py: SilverWriter, QuarantineWriter.
- src/data_pipeline/silver/pipeline.py: SilverPipeline orchestration and report.
- src/data/silver/models.py and reader.py: shared validated data contracts.

## Run

From project/backend:

    python -m scripts.build_silver data/bronze/alphabeta/games.jsonl --output-dir data/silver/run_001

Multiple input JSONL files are supported; dedup is applied across them.
Use a new output directory for every run; outputs are never overwritten.

Outputs:

- games.jsonl: one validated SilverGame per unique game
- transitions.jsonl: one SilverTransition per legal move
- duplicates.jsonl: repeated canonical games and original provenance
- quarantine.jsonl: malformed, inconsistent, or illegal source records

The report contains source_records, accepted_games, transitions,
duplicate_games, and quarantined_records.

## Semantics

- Plys are 1-based, board is 90 row-major cells, actions are 0..89 squares.
- The raw side-to-move, board snapshots, final board, and outcome are checked
  against replay when supplied.
- opening_book, opening_random, and alphabeta_exploration remain explicitly
  distinguishable via actor/source_type; opening book is NOT expert Alpha-Beta.
- A truncated game is not a draw. Raw reward is deliberately not normalized
  into an offline RL reward.
- Quarantine records store file SHA256, source line, error code, optional ply,
  and a short raw preview. The full raw record remains in Bronze.
- This strict first implementation covers the existing Alpha-Beta batch-game
  format. It rejects non-engine terminal results such as resignation. A
  source-specific termination policy will be needed for CCPD ingestion.
- Readers validate shape; legal Xiangqi validation happens in this pipeline.
- This implementation uses memory for a run-local hash index. It does not
  support resuming partial Silver runs. Do not treat failed partial outputs as
  complete; start a new output directory.

Run unit tests:

    python -m unittest discover -s tests -v
