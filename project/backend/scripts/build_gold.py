"""Run from project/backend: python -m scripts.build_gold --help"""

import argparse
import json

from src.data.gold.data_pipeline import GoldConfig, GoldPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build split Gold policy/offline RL JSONL from Silver runs")
    parser.add_argument("--silver-dirs", nargs="+", required=True,
                        help="Exact Silver run directories; never select both sample and full of one dataset")
    parser.add_argument("--output-dir", required=True, help="New output directory (must not exist)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    config = GoldConfig(seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    result = GoldPipeline(config).run(args.silver_dirs, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
