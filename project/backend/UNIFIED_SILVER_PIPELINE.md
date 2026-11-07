# Xiangqi AI — Unified Bronze → Silver

Use a single Google Colab runner for both supported source formats:

**Notebook:** `notebooks/xiangqi_colab_data_pipeline.ipynb`

- `SOURCES=["ccpd"]`: CCPD PGN ingestion already completed in Bronze; run `scripts.build_ccpd_silver`.
- `SOURCES=["alphabeta"]`: Alpha-Beta JSONL already in Bronze; run `scripts.build_silver`.
- `SOURCES=["ccpd","alphabeta"]`: process both sources in one Colab execution, but publish separate run directories and per-source reports.
- `MODE="sample"`: deterministic 20 PGNs for CCPD or first 20 JSONL records for Alpha-Beta. The latter is done using `--max-records 20` on the **full Bronze file** to retain correct SHA-256 provenance.
- `MODE="full"`: use all selected Bronze files; CCPD checks expected 595 PGNs by default.

The notebook clones/pulls repository `main`, stages Bronze bytes **and manifest sidecars** to local Colab storage, runs the source-specific Silver CLI, verifies output JSONL line counts and SHA-256 integrity, and finally syncs each run to `MyDrive/XiangqiAI/data/02_silver` using a hidden `.uploading` folder until complete.

All source parsing/validation/replay remains in backend; the notebook is only an orchestrator. Bronze data and previously published Silver runs are never overwritten. Sources have different semantics: PGN recorded match results versus engine-verified Alpha-Beta outcomes. Both are retained in the shared Silver schema.

The previous experimental notebook `xiangqi_colab_ccpd_silver.ipynb` was removed after CCPD sample validation. The separate `xiangqi_colab_bronze_ingestion.ipynb` remains because downloading/ingesting raw Bronze is a different step.

## Running

On Google Colab, open `xiangqi_colab_data_pipeline.ipynb`, set `SOURCES` and `MODE`, then choose Runtime → Run all.

Use `MODE="sample"` first for new sources or parser changes. For the already-tested 595-file CCPD collection, `MODE="full"` is the default. Inspect `report.json` and `quarantine.jsonl` in the newly published Silver run. A full dataset can still contain invalid or duplicate games; those must not be silently added to training.

## Command-line options

CCPD:
```bash
python -m scripts.build_ccpd_silver /path/to/01_bronze/ccpd --output-dir /path/to/new_ccpd_run
```

Alpha-Beta:
```bash
python -m scripts.build_silver /path/to/01_bronze/alphabeta/games.jsonl --output-dir /path/to/new_ab_run
```

Alpha-Beta sample that preserves original full-file hash:
```bash
python -m scripts.build_silver /path/to/01_bronze/alphabeta/games.jsonl \
  --output-dir /path/to/new_ab_sample --max-records 20
```
