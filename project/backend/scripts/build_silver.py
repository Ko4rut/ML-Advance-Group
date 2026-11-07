"""Run from project/backend: python -m scripts.build_silver ..."""

import argparse
import json

from src.data.silver.data_pipeline import SilverPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build replay-validated Alpha-Beta Silver JSONL")
    parser.add_argument("inputs", nargs="+", help="Bronze Alpha-Beta JSONL files")
    parser.add_argument("--output-dir", required=True, help="New Silver output directory")
    parser.add_argument("--source-version", default="batch-v1")
    parser.add_argument("--license", default=None)
    parser.add_argument("--max-records", type=int, default=None,
                        help="Limit records across all input files (sample from full raw files, preserving file SHA)")
    args = parser.parse_args()
    report = SilverPipeline().run_alphabeta(
        args.inputs, args.output_dir, source_version=args.source_version,
        license=args.license, max_records=args.max_records
    )
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
