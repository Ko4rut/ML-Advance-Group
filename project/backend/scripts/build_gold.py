"""Run from project/backend: python -m scripts.build_gold --help"""

import argparse
import json

from src.data.gold.data_pipeline import GoldConfig, GoldPipeline
from src.data.gold.data_pipeline.rewards import RewardConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build split Gold policy/offline RL JSONL from Silver runs")
    parser.add_argument("--silver-dirs", nargs="+", required=True,
                        help="Exact Silver run directories; never select both sample and full of one dataset")
    parser.add_argument("--output-dir", required=True, help="New output directory (must not exist)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--reward-mode", choices=("sparse", "shaped"), default="shaped",
                        help="shaped is the new DQN reward; sparse reproduces Gold v1")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Must match gamma used by the DQN trainer")
    parser.add_argument("--shaping-weight", type=float, default=0.15,
                        help="Potential shaping strength (0 to 1)")
    parser.add_argument("--material-scale", type=float, default=300.0)
    parser.add_argument("--pawn-advancement-weight", type=float, default=1.0)
    args = parser.parse_args()
    config = GoldConfig(seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    rewards = RewardConfig(
        mode=args.reward_mode,
        gamma=args.gamma,
        shaping_weight=args.shaping_weight,
        material_scale=args.material_scale,
        pawn_advancement_weight=args.pawn_advancement_weight,
    )
    result = GoldPipeline(config, reward_config=rewards).run(args.silver_dirs, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
