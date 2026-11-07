"""Deterministic, auditable Silver -> task-specific Gold conversion.

The state/action reward is always measured from the side that made the move.
For two-player alternating zero-sum Q learning the next state's value belongs
TO THE OPPONENT, so an online bootstrap uses reward - gamma * next_value.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Sequence

from src.data.gold.models import OfflineRLSample, PolicySample
from src.data.silver.models import SilverGame, SilverTransition
from src.data.silver.data_pipeline.deduplication import canonical_game_hash
from src.data.silver.reader import SilverGameReader, SilverTransitionReader
from src.environment import GameEngine, GameState, Side

from .rewards import RewardConfig, calculate_reward


SPLITS = ("train", "val", "test")
OUTPUTS = tuple(f"{kind}_{split}.jsonl" for kind in ("policy", "offline_rl") for split in SPLITS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GoldConfig:
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    def __post_init__(self) -> None:
        if not (0.0 < self.train_ratio < 1.0 and 0.0 < self.val_ratio < 1.0):
            raise ValueError("train_ratio and val_ratio must be in (0, 1)")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1")

    def split_for(self, canonical_hash: str) -> str:
        # Uses a stable game fingerprint rather than game_id, which may vary
        # between source datasets. All transitions in a game share its split.
        digest = hashlib.sha256(f"{self.seed}:{canonical_hash}".encode("ascii")).digest()
        number = int.from_bytes(digest[:8], "big") / 2**64
        if number < self.train_ratio:
            return "train"
        if number < self.train_ratio + self.val_ratio:
            return "val"
        return "test"


class GoldPipeline:
    """Read validated Silver runs and produce supervised + offline-RL datasets.

    Source runs should be passed explicitly; do not glob a parent directory
    containing sample/full/revalidation runs because that would double-count.
    """

    def __init__(
        self,
        config: GoldConfig | None = None,
        engine: GameEngine | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.config = config or GoldConfig()
        self.engine = engine or GameEngine()
        self.reward_config = reward_config or RewardConfig(mode="sparse")
        self._legal_cache: dict[tuple[tuple[int, ...], str], list[int]] = {}

    def _legal_ids(self, board: list[int], side: str) -> list[int]:
        key = (tuple(board), side)
        if key in self._legal_cache:
            return self._legal_cache[key]
        side_enum = Side.RED if side == "red" else Side.BLACK
        state = GameState(key[0], side_enum)
        actions = sorted(move.source * 90 + move.target for move in self.engine.legal_moves(state))
        if len(self._legal_cache) >= 4096:
            self._legal_cache.clear()
        self._legal_cache[key] = actions
        return actions

    @staticmethod
    def _validate_steps(game: SilverGame, steps: list[SilverTransition]) -> None:
        if not steps or len(steps) != len(game.moves):
            raise ValueError(f"{game.game_id}: mismatch between game moves and Silver transitions")
        if canonical_game_hash(game.initial_board, game.initial_side, [move.action for move in game.moves]) != game.canonical_game_sha256:
            raise ValueError(f"{game.game_id}: canonical Silver game hash mismatch")
        board = game.initial_board
        side = game.initial_side
        for number, (move, step) in enumerate(zip(game.moves, steps), start=1):
            if (move.ply != number or step.ply != number or step.game_id != game.game_id
                    or step.action != move.action or step.board != board or step.side != side
                    or step.source_type != move.source_type or step.actor != move.actor):
                raise ValueError(f"{game.game_id}: inconsistent Silver transition at ply {number}")
            if (step.terminated or step.truncated or step.recorded_game_end) and number != len(steps):
                raise ValueError(f"{game.game_id}: premature episode end at ply {number}")
            if board[step.action.source] == 0:
                raise ValueError(f"{game.game_id}: moving from an empty square at ply {number}")
            if (board[step.action.source] > 0) != (side == "red"):
                raise ValueError(f"{game.game_id}: moving the wrong side at ply {number}")
            expected = list(board)
            expected[step.action.target] = expected[step.action.source]
            expected[step.action.source] = 0
            next_side = "black" if side == "red" else "red"
            if step.next_board != expected or step.next_side != next_side:
                raise ValueError(f"{game.game_id}: invalid successor at ply {number}")
            board, side = expected, next_side
        final = steps[-1]
        if game.outcome.status == "finished":
            if not (final.terminated or final.recorded_game_end):
                raise ValueError(f"{game.game_id}: finished game lacks terminal evidence")
            if game.outcome.result_source == "engine" and not final.terminated:
                raise ValueError(f"{game.game_id}: engine result is not engine-terminal")
        elif final.terminated or final.recorded_game_end:
            raise ValueError(f"{game.game_id}: terminal Silver row contradicts outcome status")

    @staticmethod
    def _end_flags(game: SilverGame, step: SilverTransition, is_last: bool) -> tuple[bool, bool, str]:
        if not is_last:
            return False, False, "none"
        if game.outcome.status == "finished":
            origin = "engine" if step.terminated else "recorded"
            return True, False, origin
        return False, True, "none"

    def run(self, silver_dirs: Sequence[str | Path], output_dir: str | Path) -> dict[str, object]:
        sources = [Path(path).resolve() for path in silver_dirs]
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("provide unique, explicit Silver run directories")
        output = Path(output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"Gold output already exists: {output}")
        input_metadata: list[dict[str, object]] = []
        for folder in sources:
            if output == folder or folder in output.parents or output in folder.parents:
                raise ValueError("Gold output must be independent from Silver input directories")
            games, transitions = folder / "games.jsonl", folder / "transitions.jsonl"
            if not games.is_file() or not transitions.is_file():
                raise FileNotFoundError(f"Missing Silver games.jsonl/transitions.jsonl: {folder}")
            input_metadata.append({
                "silver_run": folder.name,
                "games_sha256": file_sha256(games),
                "transitions_sha256": file_sha256(transitions),
            })

        report: dict[str, object] = {
            "schema_version": 1,
            "reward_definition": self.reward_config.definition,
            "reward_config": self.reward_config.as_dict(),
            "action_encoding": "from_to_8100_v1",
            "reward_perspective": "side_that_selected_action",
            "bootstrap_note": "For alternating zero-sum states use r - gamma * opponent_next_Q; do not add next_Q.",
            "seed": self.config.seed,
            "ratios": {"train": self.config.train_ratio,
                       "val": self.config.val_ratio,
                       "test": 1 - self.config.train_ratio - self.config.val_ratio},
            "inputs": input_metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        games_by_split: Counter[str] = Counter()
        steps_by_split: Counter[str] = Counter()
        games_by_source: Counter[str] = Counter()
        end_origin: Counter[str] = Counter()
        duplicate_games = 0
        canonical_seen: set[str] = set()
        game_id_seen: set[str] = set()

        output.parent.mkdir(parents=True, exist_ok=True)
        # Stage complete output and publish only if every row passes checks.
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temp_dir:
            staged = Path(temp_dir)
            with ExitStack() as stack:
                streams = {name: stack.enter_context((staged / name).open("x", encoding="utf-8"))
                           for name in OUTPUTS}
                for folder in sources:
                    games = list(SilverGameReader(folder / "games.jsonl"))
                    game_index = {game.game_id: game for game in games}
                    if len(game_index) != len(games):
                        raise ValueError(f"{folder}: duplicate game_id within a Silver games file")
                    local_seen: set[str] = set()
                    transition_iter = SilverTransitionReader(folder / "transitions.jsonl")
                    for game_id, grouped in groupby(transition_iter, key=lambda row: row.game_id):
                        steps = list(grouped)
                        if game_id not in game_index or game_id in local_seen:
                            raise ValueError(f"{folder}: unexpected or noncontiguous game_id {game_id}")
                        local_seen.add(game_id)
                        game = game_index[game_id]
                        self._validate_steps(game, steps)
                        canonical = game.canonical_game_sha256
                        if canonical in canonical_seen:
                            duplicate_games += 1
                            continue
                        if game_id in game_id_seen:
                            raise ValueError(f"{game_id}: ID collision across sources")
                        canonical_seen.add(canonical)
                        game_id_seen.add(game_id)
                        split = self.config.split_for(canonical)
                        games_by_split[split] += 1
                        games_by_source[game.provenance.source] += 1
                        for index, step in enumerate(steps):
                            action_id = step.action.source * 90 + step.action.target
                            legal_actions = self._legal_ids(step.board, step.side)
                            if action_id not in legal_actions:
                                raise ValueError(f"{game_id}: action not legal at ply {step.ply}")
                            terminal, truncated, origin = self._end_flags(
                                game, step, index == len(steps) - 1
                            )
                            next_legal = [] if terminal else self._legal_ids(step.next_board, step.next_side)
                            source = game.provenance.source
                            policy = PolicySample(
                                game_id=game_id, ply=step.ply, split=split,
                                state=step.board, side=step.side,
                                target_action_id=action_id, legal_action_ids=legal_actions,
                                behavior_source_type=step.source_type,
                                source_dataset=source,
                            )
                            rl = OfflineRLSample(
                                game_id=game_id, ply=step.ply, split=split,
                                state=step.board, side=step.side, action_id=action_id,
                                reward=calculate_reward(game, step, terminal, self.reward_config),
                                reward_definition=self.reward_config.definition,
                                next_state=step.next_board, next_side=step.next_side,
                                legal_action_ids=legal_actions,
                                next_legal_action_ids=next_legal,
                                terminated=terminal, truncated=truncated,
                                termination_source=origin,
                                behavior_source_type=step.source_type,
                                source_dataset=source,
                            )
                            for label, sample in (("policy", policy), ("offline_rl", rl)):
                                streams[f"{label}_{split}.jsonl"].write(
                                    json.dumps(sample.model_dump(mode="json"),
                                               ensure_ascii=False, separators=(",", ":")) + "\n"
                                )
                            steps_by_split[split] += 1
                            end_origin[origin] += 1
                    missing = set(game_index) - local_seen
                    if missing:
                        raise ValueError(f"{folder}: {len(missing)} games without transitions")

            report["accepted_games"] = sum(games_by_split.values())
            report["transitions"] = sum(steps_by_split.values())
            report["cross_run_duplicate_games"] = duplicate_games
            report["games_by_split"] = {name: games_by_split[name] for name in SPLITS}
            report["transitions_by_split"] = {name: steps_by_split[name] for name in SPLITS}
            report["games_by_source"] = dict(games_by_source)
            report["termination_sources_per_transition"] = dict(end_origin)
            report["output_sha256"] = {
                name: file_sha256(staged / name) for name in OUTPUTS
            }
            (staged / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            # No output is published if anything above fails.
            if output.exists():
                raise FileExistsError(f"Gold output already exists: {output}")
            staged.rename(output)
        return report
