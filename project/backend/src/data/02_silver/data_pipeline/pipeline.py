"""Streaming Alpha-Beta Bronze -> Silver ETL, never overwriting source data."""

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from src.data.bronze.models import AlphaBetaRawGame
from src.data.models import Provenance

from .adapters import AlphaBetaAdapter
from .deduplication import GameDeduplicator
from .errors import SilverValidationError
from .replay import SilverReplayValidator
from .writers import QuarantineWriter, SilverWriter


@dataclass(slots=True)
class SilverReport:
    source_records: int = 0
    accepted_games: int = 0
    transitions: int = 0
    duplicate_games: int = 0
    quarantined_records: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "source_records": self.source_records,
            "accepted_games": self.accepted_games,
            "transitions": self.transitions,
            "duplicate_games": self.duplicate_games,
            "quarantined_records": self.quarantined_records,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SilverPipeline:
    def __init__(self, replay_validator: SilverReplayValidator | None = None) -> None:
        self.adapter = AlphaBetaAdapter()
        self.validator = replay_validator or SilverReplayValidator()

    def run_alphabeta(
        self, input_files: Sequence[str | Path], output_dir: str | Path,
        *, source_version: str = "batch-v1", license: str | None = None,
        max_records: int | None = None,
    ) -> SilverReport:
        """Build new Silver files and quarantine bad rows without aborting the batch."""
        if max_records is not None and max_records <= 0:
            raise ValueError("max_records must be positive")
        inputs = [Path(path) for path in input_files]
        if not inputs:
            raise ValueError("at least one input file is required")
        if len({path.resolve() for path in inputs}) != len(inputs):
            raise ValueError("input_files contains the same path twice")
        output_root = Path(output_dir)
        filenames = ("games.jsonl", "transitions.jsonl", "duplicates.jsonl", "quarantine.jsonl")
        destinations = [output_root / name for name in filenames]
        for path in destinations:
            if path.exists():
                raise FileExistsError(f"Silver output already exists: {path}")
        for source in inputs:
            if not source.is_file():
                raise FileNotFoundError(f"Bronze input does not exist: {source}")
        output_root.mkdir(parents=True, exist_ok=True)

        report = SilverReport()
        dedup = GameDeduplicator()
        with ExitStack() as stack:
            streams = [stack.enter_context(path.open("x", encoding="utf-8")) for path in destinations]
            writer = SilverWriter(streams[0], streams[1], streams[2])
            quarantine = QuarantineWriter(streams[3])
            for source in inputs:
                digest = _sha256_file(source)
                with source.open("r", encoding="utf-8") as stream:
                    for line_number, raw_line in enumerate(stream, start=1):
                        if max_records is not None and report.source_records >= max_records:
                            break
                        if not raw_line.strip():
                            continue
                        report.source_records += 1
                        try:
                            payload = json.loads(raw_line)
                        except json.JSONDecodeError as exc:
                            quarantine.write(
                                source.name, digest, line_number, "INVALID_JSON", exc.msg, raw_line
                            )
                            report.quarantined_records += 1
                            continue
                        try:
                            raw = AlphaBetaRawGame.model_validate(payload)
                        except ValidationError as exc:
                            quarantine.write(
                                source.name, digest, line_number, "INVALID_SCHEMA", str(exc), raw_line
                            )
                            report.quarantined_records += 1
                            continue
                        try:
                            provenance = Provenance(
                                source="alphabeta", source_version=source_version,
                                source_file=source.name, file_sha256=digest,
                                license=license, source_game_id=raw.game_id,
                            )
                            game = self.adapter.convert(raw, provenance)
                            steps = self.validator.replay(game)
                            self.adapter.audit(raw, steps)
                        except (SilverValidationError, ValidationError) as exc:
                            code = exc.code if isinstance(exc, SilverValidationError) else "INVALID_SCHEMA"
                            ply = exc.ply if isinstance(exc, SilverValidationError) else None
                            quarantine.write(
                                source.name, digest, line_number, code, str(exc), raw_line, ply
                            )
                            report.quarantined_records += 1
                            continue
                        original_id = dedup.register(game)
                        if original_id is not None:
                            writer.write_duplicate(game, original_id, line_number)
                            report.duplicate_games += 1
                            continue
                        writer.write_game(game, steps)
                        report.accepted_games += 1
                        report.transitions += len(steps)
                if max_records is not None and report.source_records >= max_records:
                    break
        return report
