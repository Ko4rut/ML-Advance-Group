# Xiangqi Medallion data contracts (v1)

The src/data/ directory contains Python code only. Keep large files OUTSIDE Git:
data/01_bronze, data/02_silver, data/03_gold on Drive or a separate configurable storage root.

| Layer | Pydantic models | Readers |
| --- | --- | --- |
| Bronze | BronzeRecord, AlphaBetaRawGame | BronzeReader, AlphaBetaRawReader |
| Silver | SilverGame, SilverTransition | SilverGameReader, SilverTransitionReader |
| Gold | PolicySample, OfflineRLSample | PolicyReader, OfflineRLReader |

- Use schema_version=1 and extra=forbid for Silver and Gold.
  The raw AlphaBetaRawGame adapter permits unknown fields; original source
  bytes are preserved independently of the BronzeRecord sidecar manifest.
- Board has exactly 90 row-major cells, Black row 0 and Red row 9.
  Pieces are coded -7..7 to match src/environment/state.py.
  Moves use source/target square indices from 0 to 89. Ply is one-based.
- provenance.source identifies a dataset; source_type identifies who
  selected a particular move. Book openings are NOT Alpha-Beta labels.
- Truncated games are NOT draws. Silver recorded_reward is optional and
  is not a normalized learning reward. Offline RL Gold requires an explicit
  reward_definition before reward values can be interpreted.
- Policy Gold stores sparse legal_action_ids and target_action_id using
  the explicit from_to_8100_v1 encoding: source * 90 + target.
  Call dense_legal_action_mask during training to expand an 8100-element mask.
  src/encoding/actions.py and observations.py are currently stubs.
- Canonical duplicate hash must come from initial board, starting side
  and action sequence, NOT the source game ID or untrusted raw fingerprint.
  Split train/val/test by entire game AFTER grouping duplicates.
- Readers only parse/validate JSONL; they do NOT replay GameEngine,
  parse CCPD notation, deduplicate, quarantine, build Gold, or write files.
  Implement these as separate ETL stages in src/data/02_silver/data_pipeline/.
- BronzeRecord tracks the original source file SHA256, dataset version,
  licensing information, and ingestion time. Never mutate raw input files.

Run tests from project/backend:
python -m unittest discover -s tests -v

## Numbered folders

Physical source-code folders are `src/data/01_bronze/`, `src/data/02_silver/`, and `src/data/03_gold/`. Existing Python import paths (`src.data.bronze`, `src.data.silver`, `src.data.gold`) are retained as runtime aliases from `src/data/__init__.py`, because Python import statements cannot use identifiers beginning with digits. Dataset storage paths on Drive are separate from source-code paths and are not moved by this commit.
