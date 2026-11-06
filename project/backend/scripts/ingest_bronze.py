"""Bronze CLI. Run from project/backend: python -m scripts.ingest_bronze ..."""

import argparse
import json

from src.data.bronze.data_pipeline import BronzeIngester


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy raw dataset files to immutable Bronze storage with SHA-256 manifests"
    )
    parser.add_argument("inputs", nargs="+", help="Local files or dataset directories")
    parser.add_argument("--destination-root", required=True,
                        help="Parent folder containing 01_bronze/source")
    parser.add_argument("--source", required=True,
                        help="Dataset identifier, e.g. alphabeta or ccpd")
    parser.add_argument("--source-version", required=True,
                        help="Dataset version or ingestion batch identifier")
    parser.add_argument("--include", nargs="+", default=["*.jsonl"],
                        help="Filename globs; e.g. *.jsonl or *.pgn *.txt (default: *.jsonl)")
    parser.add_argument("--license", default=None,
                        help="License text supplied by dataset owner; never guessed")
    args = parser.parse_args()

    report = BronzeIngester().ingest(
        args.inputs,
        args.destination_root,
        source=args.source,
        source_version=args.source_version,
        patterns=args.include,
        license=args.license,
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
