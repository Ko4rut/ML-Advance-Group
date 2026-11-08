"""Offline exact-board tabular Q-learning for Xiangqi Gold transitions.

Each Q(s,a) belongs to the player whose turn it is. Therefore the
alternating-player, zero-sum target is r - gamma * opponent_best_Q.
Offline bootstrapping uses only actions OBSERVED in train, never arbitrary
legal actions with unsupported, untrained Q estimates.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.data.gold.models import OfflineRLSample
from src.data.gold.reader import OfflineRLReader

MODEL_VERSION = "tabular_qlearning_exact_board_v1"


def state_key(board: Iterable[int], side: str) -> bytes:
    """Exact 90-square board plus side; compact 91-byte dictionary key."""
    squares = tuple(board)
    if len(squares) != 90 or any(
        not isinstance(piece, int) or isinstance(piece, bool) or not -7 <= piece <= 7
        for piece in squares
    ):
        raise ValueError("board must contain 90 piece codes in [-7, 7]")
    if side not in ("red", "black"):
        raise ValueError("side must be red or black")
    return bytes((1 if side == "red" else 0, *(piece + 7 for piece in squares)))


@dataclass(frozen=True, slots=True)
class TabularQConfig:
    alpha: float = 0.2
    gamma: float = 0.99
    epochs: int = 12
    seed: int = 42
    skip_truncated: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("alpha must be finite in (0, 1]")
        if not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be finite in (0, 1]")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")


def _load_games(path: str | Path, expected_split: str) -> list[list[OfflineRLSample]]:
    """Validate Gold schema, sequential episode states and split boundaries."""
    games: list[list[OfflineRLSample]] = []
    current: list[OfflineRLSample] = []
    seen: set[str] = set()
    reward_definition: str | None = None
    for item in OfflineRLReader(path):
        if item.split != expected_split or item.next_side is None:
            raise ValueError(f"{path}: invalid split or absent next_side")
        if not item.legal_action_ids or item.action_id not in item.legal_action_ids:
            raise ValueError(f"{path}: observed action not legal")
        if reward_definition is None:
            reward_definition = item.reward_definition
        elif reward_definition != item.reward_definition:
            raise ValueError("mixed reward definitions in Gold file")
        if not math.isfinite(item.reward):
            raise ValueError("reward must be finite")
        if not current or item.game_id != current[-1].game_id:
            if current:
                games.append(current)
            if item.game_id in seen:
                raise ValueError(f"noncontiguous game: {item.game_id}")
            seen.add(item.game_id)
            if item.ply != 1:
                raise ValueError(f"game {item.game_id}: first ply must be 1")
            current = []
        else:
            previous = current[-1]
            if item.ply != previous.ply + 1:
                raise ValueError(f"game {item.game_id}: nonsequential ply")
            if previous.terminated or previous.truncated:
                raise ValueError(f"game {item.game_id}: transition after end")
            if item.state != previous.next_state or item.side != previous.next_side:
                raise ValueError(f"game {item.game_id}: transition chain is broken")
        current.append(item)
    if current:
        games.append(current)
    if not games:
        raise ValueError(f"Gold file is empty: {path}")
    return games


def load_gold_splits(folder: str | Path) -> dict[str, list[list[OfflineRLSample]]]:
    root = Path(folder)
    return {
        split: _load_games(root / f"offline_rl_{split}.jsonl", split)
        for split in ("train", "val", "test")
    }


class TabularQTable:
    """Sparse exact-state action values and train observation counts."""

    def __init__(self) -> None:
        self.values: dict[bytes, dict[int, float]] = {}
        self.counts: dict[bytes, dict[int, int]] = {}

    def observe(self, key: bytes, action: int) -> None:
        self.values.setdefault(key, {}).setdefault(action, 0.0)
        counts = self.counts.setdefault(key, {})
        counts[action] = counts.get(action, 0) + 1

    def supported_max(self, key: bytes, legal_actions: Iterable[int]) -> float | None:
        values = self.values.get(key)
        if values is None:
            return None
        candidates = [values[action] for action in legal_actions if action in values]
        return max(candidates) if candidates else None

    def select(self, key: bytes, legal_actions: Iterable[int]) -> int | None:
        """Use only observed actions; break Q ties by action ID."""
        values = self.values.get(key)
        if not values:
            return None
        candidates = [action for action in legal_actions if action in values]
        if not candidates:
            return None
        return min(candidates, key=lambda action: (-values[action], action))

    def save(self, destination: str | Path, config: TabularQConfig, metadata: dict) -> None:
        path = Path(destination)
        if path.exists():
            raise FileExistsError(f"model checkpoint exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": MODEL_VERSION,
            "config": {
                "alpha": config.alpha, "gamma": config.gamma, "epochs": config.epochs,
                "seed": config.seed, "skip_truncated": config.skip_truncated,
            },
            "metadata": metadata,
            "table": [
                [key.hex(), [[action, q, self.counts[key][action]]
                             for action, q in sorted(values.items())]]
                for key, values in sorted(self.values.items())
            ],
        }
        with gzip.open(path, "xt", encoding="utf-8", compresslevel=6) as stream:
            json.dump(payload, stream, separators=(",", ":"), allow_nan=False)

    @classmethod
    def load(cls, source: str | Path) -> tuple["TabularQTable", dict]:
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("format") != MODEL_VERSION:
            raise ValueError("unsupported Tabular Q checkpoint format")
        table = cls()
        for key_hex, entries in payload["table"]:
            key = bytes.fromhex(key_hex)
            if len(key) != 91:
                raise ValueError("invalid checkpoint state key")
            table.values[key] = {int(a): float(q) for a, q, _ in entries}
            table.counts[key] = {int(a): int(n) for a, _, n in entries}
        return table, payload


class TabularQTrainer:
    """Replay whole games in reverse, not random isolated transitions."""

    def __init__(self, config: TabularQConfig | None = None) -> None:
        self.config = config or TabularQConfig()
        self.table = TabularQTable()
        self.reward_definition: str | None = None
        self.train_transitions = 0

    def fit(self, games: list[list[OfflineRLSample]]) -> list[dict]:
        if not games:
            raise ValueError("training games are empty")
        for game in games:
            for transition in game:
                if self.reward_definition is None:
                    self.reward_definition = transition.reward_definition
                elif self.reward_definition != transition.reward_definition:
                    raise ValueError("mixed train reward definitions")
                self.table.observe(state_key(transition.state, transition.side),
                                   transition.action_id)
                self.train_transitions += 1

        rng = random.Random(self.config.seed)
        indices = list(range(len(games)))
        history: list[dict] = []
        for epoch in range(1, self.config.epochs + 1):
            rng.shuffle(indices)
            errors = 0.0
            updates = missing = skipped = 0
            for index in indices:
                for step in reversed(games[index]):
                    if step.truncated and self.config.skip_truncated:
                        skipped += 1
                        continue
                    key = state_key(step.state, step.side)
                    target = step.reward
                    if not step.terminated:
                        next_key = state_key(step.next_state, step.next_side)
                        next_best = self.table.supported_max(
                            next_key, step.next_legal_action_ids
                        )
                        if next_best is None:
                            # Do NOT equate unknown/truncated continuation with a draw.
                            missing += 1
                            continue
                        target -= self.config.gamma * next_best
                    old_q = self.table.values[key][step.action_id]
                    td_error = target - old_q
                    self.table.values[key][step.action_id] = (
                        old_q + self.config.alpha * td_error
                    )
                    errors += td_error * td_error
                    updates += 1
            history.append({
                "epoch": epoch,
                "updates": updates,
                "skipped_truncated": skipped,
                "missing_next_action_support": missing,
                "td_rmse": math.sqrt(errors / updates) if updates else 0.0,
            })
        return history

    def evaluate(self, games: list[list[OfflineRLSample]]) -> dict:
        """Held-out action overlap metrics, NOT playing win-rate."""
        total = seen = action_seen = matched = residual_count = 0
        abs_residual = 0.0
        for game in games:
            for step in game:
                total += 1
                key = state_key(step.state, step.side)
                supported = self.table.values.get(key)
                if supported is None:
                    continue
                seen += 1
                if step.action_id not in supported:
                    continue
                action_seen += 1
                if self.table.select(key, step.legal_action_ids) == step.action_id:
                    matched += 1
                if step.truncated:
                    continue
                target = step.reward
                if not step.terminated:
                    best = self.table.supported_max(
                        state_key(step.next_state, step.next_side),
                        step.next_legal_action_ids,
                    )
                    if best is None:
                        continue
                    target -= self.config.gamma * best
                abs_residual += abs(target - supported[step.action_id])
                residual_count += 1
        return {
            "transitions": total,
            "seen_state_rows": seen,
            "seen_state_fraction": seen / total if total else 0.0,
            "seen_state_action_rows": action_seen,
            "seen_state_action_fraction": action_seen / total if total else 0.0,
            "behavior_top1_supported": matched / action_seen if action_seen else None,
            "mean_abs_td_error_supported": abs_residual / residual_count if residual_count else None,
            "td_error_rows": residual_count,
        }


def train_from_gold(gold_dir: str | Path, output_dir: str | Path,
                    config: TabularQConfig | None = None) -> dict:
    """Read existing Gold files, train on train ONLY, save model and report."""
    config = config or TabularQConfig()
    gold = Path(gold_dir)
    data = load_gold_splits(gold)
    info = gold / "report.json"
    source_report = json.loads(info.read_text(encoding="utf-8")) if info.is_file() else {}
    gold_gamma = source_report.get("reward_config", {}).get("gamma")
    if gold_gamma is not None and not math.isclose(
        config.gamma, gold_gamma, abs_tol=1e-12
    ):
        raise ValueError(f"Q gamma {config.gamma} != Gold shaping gamma {gold_gamma}")
    declared = source_report.get("reward_definition")
    if declared is not None and any(
        step.reward_definition != declared
        for split in data.values() for game in split for step in game
    ):
        raise ValueError("Gold report reward_definition does not match samples")

    digest = hashlib.sha256((gold / "offline_rl_train.jsonl").read_bytes()).hexdigest()
    recorded_digest = source_report.get("output_sha256", {}).get("offline_rl_train.jsonl")
    if recorded_digest is not None and digest != recorded_digest:
        raise ValueError("Gold train SHA-256 does not match report")

    trainer = TabularQTrainer(config)
    history = trainer.fit(data["train"])
    report = {
        "algorithm": MODEL_VERSION,
        "gold_directory": gold.name,
        "train_sha256": digest,
        "reward_definition": trainer.reward_definition,
        "config": {
            "alpha": config.alpha, "gamma": config.gamma, "epochs": config.epochs,
            "seed": config.seed, "skip_truncated": config.skip_truncated,
        },
        "train_games": len(data["train"]),
        "train_transitions": trainer.train_transitions,
        "known_states": len(trainer.table.values),
        "known_state_action_pairs": sum(len(a) for a in trainer.table.values.values()),
        "history": history,
        "splits": {
            name: trainer.evaluate(data[name])
            for name in ("train", "val", "test")
        },
        "notes": [
            "Exact board table cannot generalize to unseen states.",
            "Held-out behavior agreement is NOT a playing-strength metric.",
            "Backup uses only observed train actions and r - gamma * opponent_Q.",
            "Board+side omits repetition and no-capture history.",
        ],
    }
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"model output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.tmp-", dir=output.parent) as tmp:
        stage = Path(tmp)
        trainer.table.save(stage / "q_table.json.gz", config, {
            "reward_definition": trainer.reward_definition,
            "train_sha256": digest,
            "gold_directory": gold.name,
        })
        (stage / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        stage.rename(output)
    return report
