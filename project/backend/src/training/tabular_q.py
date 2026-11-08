"""Exact-state offline tabular Q-learning baseline for Xiangqi Gold transitions.

A full 90-square board plus side-to-move is one discrete state. This deliberately
does NOT generalize to unseen boards. Only actions observed in TRAIN are eligible
for max-Q backups; otherwise an offline max over unobserved actions is unsafe.

The Gold reward is from the actor's perspective, while next_state belongs to
their opponent. Consequently the correct alternating zero-sum target is:
    target = reward - gamma * max_{observed legal next actions} Q(next_state, a)
Terminal transitions have target=reward. Truncated *games* are skipped by
default because Gold contains no known continuation for their final states.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.data.gold.reader import OfflineRLReader

BOARD_SIZE = 90
ACTION_SPACE = BOARD_SIZE * BOARD_SIZE
TABLE_SCHEMA_VERSION = 1
STATE_ENCODING = "board_plus7_side_byte_v1"


def encode_state(board: Sequence[int], side: str) -> bytes:
    """Lossless state key; no potentially misleading material bucketing."""
    if len(board) != BOARD_SIZE or side not in ("red", "black"):
        raise ValueError("state must have 90 squares and a red/black side")
    if any(not isinstance(piece, int) or isinstance(piece, bool) or not -7 <= piece <= 7
           for piece in board):
        raise ValueError("invalid Xiangqi piece encoding")
    return bytes(piece + 7 for piece in board) + bytes((1 if side == "red" else 0,))


@dataclass(frozen=True)
class TabularQConfig:
    alpha: float = 0.3
    gamma: float = 0.99
    epochs: int = 25
    seed: int = 42
    skip_truncated_games: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("alpha must be finite and in (0, 1]")
        if not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be finite and in (0, 1]")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")


@dataclass(frozen=True)
class _Step:
    game_id: str
    ply: int
    key: bytes
    action: int
    reward: float
    next_key: bytes
    next_legal: tuple[int, ...]
    terminated: bool
    truncated: bool


class TabularQTable:
    """Sparse Q(s,a) map for exact board+side states and TRAIN-supported actions."""

    def __init__(self) -> None:
        self._values: dict[bytes, dict[int, float]] = {}

    @property
    def state_count(self) -> int:
        return len(self._values)

    @property
    def action_count(self) -> int:
        return sum(len(actions) for actions in self._values.values())

    def register(self, key: bytes, action: int) -> None:
        if not 0 <= action < ACTION_SPACE:
            raise ValueError("action_id out of range")
        self._values.setdefault(key, {}).setdefault(action, 0.0)

    def supported(self, key: bytes, legal: Sequence[int] | None = None) -> dict[int, float]:
        values = self._values.get(key, {})
        if legal is None:
            return dict(values)
        legal_set = set(legal)
        return {action: value for action, value in values.items() if action in legal_set}

    def best_supported_value(self, key: bytes, legal: Sequence[int]) -> float:
        """Zero fallback is explicit when the opponent state/action is unseen."""
        values = self._values.get(key)
        if not values:
            return 0.0
        legal_set = set(legal)
        matches = (value for action, value in values.items() if action in legal_set)
        return max(matches, default=0.0)

    def td_target(self, step: _Step, gamma: float) -> float:
        if step.terminated:
            return step.reward
        opponent_value = self.best_supported_value(step.next_key, step.next_legal)
        return step.reward - gamma * opponent_value

    def update(self, step: _Step, alpha: float, gamma: float) -> float:
        current = self._values[step.key][step.action]
        target = self.td_target(step, gamma)
        difference = target - current
        self._values[step.key][step.action] = current + alpha * difference
        return difference

    def save(self, path: str | Path, metadata: dict[str, object] | None = None) -> None:
        """Gzip JSON, stable ordering, atomic file publication (no pickle)."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": TABLE_SCHEMA_VERSION,
            "state_encoding": STATE_ENCODING,
            "action_encoding": "from_to_8100_v1",
            "metadata": metadata or {},
            "states": [
                [key.hex(), [[action, value] for action, value in sorted(actions.items())]]
                for key, actions in sorted(self._values.items())
            ],
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.",
                suffix=".uploading", delete=False,
            ) as output:
                temporary = Path(output.name)
                with gzip.GzipFile(fileobj=output, mode="wb") as compressed:
                    compressed.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> tuple["TabularQTable", dict[str, object]]:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if (payload.get("schema_version") != TABLE_SCHEMA_VERSION
                or payload.get("state_encoding") != STATE_ENCODING
                or payload.get("action_encoding") != "from_to_8100_v1"):
            raise ValueError("unsupported tabular Q checkpoint schema")
        table = cls()
        for encoded, pairs in payload["states"]:
            key = bytes.fromhex(encoded)
            if len(key) != BOARD_SIZE + 1 or key[-1] not in (0, 1):
                raise ValueError("corrupt state key in Q-table")
            if key in table._values:
                raise ValueError("duplicate state key in Q-table")
            values: dict[int, float] = {}
            for action, value in pairs:
                if (not isinstance(action, int) or isinstance(action, bool)
                        or not 0 <= action < ACTION_SPACE
                        or action in values or not math.isfinite(value)):
                    raise ValueError("invalid Q-table action/value")
                values[action] = float(value)
            table._values[key] = values
        return table, payload.get("metadata", {})


class TabularQTrainer:
    """Deterministic reverse-episode fitted Q updates on Gold TRAIN only."""

    def __init__(self, config: TabularQConfig | None = None) -> None:
        self.config = config or TabularQConfig()
        self.table = TabularQTable()
        self.reward_definition: str | None = None

    def _load_training_games(
        self, path: Path,
    ) -> tuple[list[list[_Step]], dict[str, int]]:
        grouped: dict[str, list[_Step]] = defaultdict(list)
        reward_definitions: set[str] = set()
        for record in OfflineRLReader(path):
            if record.split != "train":
                raise ValueError(f"non-training sample in {path}")
            reward_definitions.add(record.reward_definition)
            if record.action_id not in record.legal_action_ids:
                raise ValueError(f"illegal recorded action in {record.game_id}")
            if record.next_side is None:
                raise ValueError("Gold v2 requires next_side")
            if record.terminated and record.truncated:
                raise ValueError("terminal and truncated cannot coincide")
            grouped[record.game_id].append(_Step(
                game_id=record.game_id,
                ply=record.ply,
                key=encode_state(record.state, record.side),
                action=record.action_id,
                reward=float(record.reward),
                next_key=encode_state(record.next_state, record.next_side),
                next_legal=tuple(record.next_legal_action_ids),
                terminated=record.terminated,
                truncated=record.truncated,
            ))
        if len(reward_definitions) != 1:
            raise ValueError(f"mixed or missing Gold reward definitions: {reward_definitions}")
        self.reward_definition = next(iter(reward_definitions))

        accepted: list[list[_Step]] = []
        skipped_games = 0
        skipped_transitions = 0
        for steps in grouped.values():
            steps.sort(key=lambda step: step.ply)
            if any(right.ply != left.ply + 1 or left.next_key != right.key
                   for left, right in zip(steps, steps[1:])):
                raise ValueError(f"broken transition chain in {steps[0].game_id}")
            if any(step.terminated or step.truncated for step in steps[:-1]):
                raise ValueError(f"premature episode end in {steps[0].game_id}")
            if self.config.skip_truncated_games and steps[-1].truncated:
                skipped_games += 1
                skipped_transitions += len(steps)
                continue
            accepted.append(steps)
            for step in steps:
                self.table.register(step.key, step.action)
        return accepted, {
            "source_games": len(grouped),
            "source_transitions": sum(map(len, grouped.values())),
            "train_games": len(accepted),
            "train_transitions": sum(map(len, accepted)),
            "skipped_truncated_games": skipped_games,
            "skipped_truncated_transitions": skipped_transitions,
        }

    def fit(self, train_path: str | Path) -> dict[str, object]:
        """Register TRAIN action support before learning, then reverse-replay games."""
        if self.table.state_count:
            raise ValueError("trainer has already been fitted")
        games, counts = self._load_training_games(Path(train_path))
        if not games:
            raise ValueError("training dataset contains no complete games")
        rng = random.Random(self.config.seed)
        history = []
        for epoch in range(1, self.config.epochs + 1):
            shuffled = list(games)
            rng.shuffle(shuffled)
            abs_error_sum = 0.0
            updates = 0
            for game in shuffled:
                for step in reversed(game):
                    abs_error_sum += abs(self.table.update(
                        step, self.config.alpha, self.config.gamma,
                    ))
                    updates += 1
            history.append({
                "epoch": epoch,
                "updates": updates,
                "mean_absolute_td_error_before_update": abs_error_sum / updates,
            })
        return {
            **counts,
            "reward_definition": self.reward_definition,
            "q_states": self.table.state_count,
            "q_state_actions": self.table.action_count,
            "config": {
                "alpha": self.config.alpha,
                "gamma": self.config.gamma,
                "epochs": self.config.epochs,
                "seed": self.config.seed,
                "skip_truncated_games": self.config.skip_truncated_games,
            },
            "learning_curve": history,
        }

    def evaluate_split(self, path: str | Path, split: str) -> dict[str, object]:
        """Offline diagnostics; not an estimate of playing strength or win rate."""
        samples = state_hits = action_hits = greedy_hits = 0
        td_abs_error = 0.0
        for row in OfflineRLReader(path):
            if row.split != split:
                raise ValueError(f"unexpected split in {path}")
            if row.reward_definition != self.reward_definition:
                raise ValueError("train/evaluation reward definitions differ")
            samples += 1
            key = encode_state(row.state, row.side)
            candidates = self.table.supported(key, row.legal_action_ids)
            if not candidates:
                continue
            state_hits += 1
            if row.action_id not in candidates:
                continue
            action_hits += 1
            best = max(candidates.values())
            if candidates[row.action_id] == best:
                greedy_hits += 1
            next_key = encode_state(row.next_state, row.next_side)
            step = _Step(
                game_id=row.game_id, ply=row.ply, key=key,
                action=row.action_id, reward=row.reward, next_key=next_key,
                next_legal=tuple(row.next_legal_action_ids),
                terminated=row.terminated, truncated=row.truncated,
            )
            td_abs_error += abs(self.table.td_target(step, self.config.gamma)
                                - candidates[row.action_id])
        return {
            "split": split,
            "samples": samples,
            "known_states": state_hits,
            "known_state_rate": state_hits / samples if samples else 0.0,
            "known_state_actions": action_hits,
            "known_state_action_rate": action_hits / samples if samples else 0.0,
            "greedy_match_given_known_action": greedy_hits / action_hits if action_hits else None,
            "mean_absolute_td_error_given_known_action": td_abs_error / action_hits if action_hits else None,
        }


def verify_gold_report(gold_dir: str | Path, gamma: float) -> dict[str, object]:
    """Refuse mismatched gamma for potential-based reward v2."""
    root = Path(gold_dir)
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("action_encoding") != "from_to_8100_v1":
        raise ValueError("expected from_to_8100_v1 Gold action encoding")
    reward = report.get("reward_definition")
    if not isinstance(reward, str):
        raise ValueError("missing Gold reward definition")
    cfg = report.get("reward_config", {})
    if cfg.get("mode") == "shaped" and not math.isclose(
        gamma, cfg.get("gamma", -1), rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("Tabular gamma must equal Gold reward-shaping gamma")
    expected = report.get("output_sha256", {})
    for name in ("offline_rl_train.jsonl", "offline_rl_val.jsonl", "offline_rl_test.jsonl"):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as reader:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
        if expected.get(name) != digest.hexdigest():
            raise ValueError(f"Gold checksum mismatch for {name}")
    return report
