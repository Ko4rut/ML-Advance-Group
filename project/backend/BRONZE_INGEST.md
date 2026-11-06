# Bronze ingestion

Bronze **copies** source files exactly; it never parses Xiangqi moves, applies Silver validation, or rewrites dataset content.

## Layout

- Ingestion service: \`src/data/01_bronze/data_pipeline/ingest.py\` (\`BronzeIngester\`)
- CLI: \`scripts/ingest_bronze.py\`
- Colab: \`notebooks/xiangqi_colab_bronze_ingestion.ipynb\`
- Metadata schema: \`BronzeFileManifest\` in \`src/data/01_bronze/models.py\`
- Existing \`BronzeReader\` / \`AlphaBetaRawReader\` stay independent.

## Run from project/backend

\`\`\`bash
python -m scripts.ingest_bronze \
  /path/to/XiangqiAI/datasets \
  --destination-root /path/to/XiangqiAI/data/01_bronze \
  --source alphabeta \
  --source-version colab-batch-v1 \
  --include '*.jsonl'
\`\`\`

The ingester creates \`01_bronze/alphabeta/<relative_source_path>\` and a
\`<filename>.manifest.json\` for every nonempty file. Manifest includes SHA-256,
source, source_version, relative path, original path, size, ingest time and
optional explicitly supplied license.

The default pattern ignores Alpha-Beta checkpoint JSON; checkpoints remain in
\`datasets/\` and should be audited separately. Empty files are counted, not
ingested. Existing files with identical hashes are accepted without modification;
content collisions fail instead of overwriting. No automatic HTTP download is
attempted. For external datasets such as CCPD, download according to the
provider's terms, then point the same CLI at those files with a distinct
\`--source ccpd\` and appropriate \`--include\` patterns.

\`\`\`bash
python -m unittest tests.data.test_bronze_ingest -v
\`\`\`

This task does not run Silver or Gold. Source file content is kept unmodified.
