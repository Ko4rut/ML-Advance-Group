"""Gold -> offline Tabular Q-Learning CLI.

From project/backend:
python -m scripts.train_tabular_qlearning \
    --gold-dir /path/to/reward_v2_shaped \
    --output-dir /path/to/models/tabular_qlearning_v1
"""

import argparse
import json

from src.training.tabular_qlearning import TabularQConfig, train_from_gold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train exact-board offline Tabular Q-Learning from Gold"
    )
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = TabularQConfig(
        alpha=args.alpha, gamma=args.gamma, epochs=args.epochs, seed=args.seed
    )
    report = train_from_gold(args.gold_dir, args.output_dir, config)
    print(json.dumps({
        "output_dir": args.output_dir,
        "train_games": report["train_games"],
        "train_transitions": report["train_transitions"],
        "known_states": report["known_states"],
        "known_state_action_pairs": report["known_state_action_pairs"],
        "last_epoch": report["history"][-1],
        "splits": report["splits"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
