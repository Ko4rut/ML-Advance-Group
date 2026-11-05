"""Run from project/backend: python -m scripts.build_silver ..."""

import argparse
import json

from src.data_pipeline.silver import SilverPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build replay-validated Alpha-Beta Silver JSONL")
    parser.add_argument("inputs", nargs="+", help="Bronze Alpha-Beta JSONL files")
    parser.add_argument("--output-dir", required=True, help="New Silver output directory")
    parser.add_argument("--source-version", default="batch-v1")
    parser.add_argument("--license", default=None)
    args = parser.parse_args()
    report = SilverPipeline().run_alphabeta(
        args.inputs, args.output_dir, source_version=args.source_version, license=args.license
    )
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
