"""Run from project/backend: python -m scripts.build_ccpd_silver ..."""

import argparse
import json

from src.data.silver.data_pipeline.ccpd import CCPDPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CCPD PGN and emit canonical Silver")
    parser.add_argument("inputs", nargs="+", help="Bronze CCPD .pgn files/directories")
    parser.add_argument("--output-dir", required=True, help="New output directory")
    parser.add_argument("--source-version", default="ccpd-v1", help="Used only without manifests")
    parser.add_argument("--license", default="CC-BY-4.0", help="Used only without manifests")
    parser.add_argument("--allow-missing-manifest", action="store_true",
                        help="Only for local experiments; no Bronze manifest verification")
    args = parser.parse_args()
    report = CCPDPipeline().run(
        args.inputs, args.output_dir,
        source_version=args.source_version,
        license=args.license,
        require_manifest=not args.allow_missing_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
