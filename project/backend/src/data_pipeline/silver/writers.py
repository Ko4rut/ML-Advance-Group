"""JSONL Silver, duplicate, and quarantine output sinks."""

import json
from typing import IO, Any

from src.data.silver.models import SilverGame, SilverTransition


def _write_json_line(stream: IO[str], payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


class SilverWriter:
    def __init__(self, games: IO[str], transitions: IO[str], duplicates: IO[str]) -> None:
        self.games = games
        self.transitions = transitions
        self.duplicates = duplicates

    def write_game(self, game: SilverGame, steps: list[SilverTransition]) -> None:
        _write_json_line(self.games, game.model_dump(mode="json"))
        for step in steps:
            _write_json_line(self.transitions, step.model_dump(mode="json"))

    def write_duplicate(self, game: SilverGame, existing_game_id: str, line_number: int) -> None:
        _write_json_line(self.duplicates, {
            "game_id": game.game_id,
            "same_as": existing_game_id,
            "canonical_game_sha256": game.canonical_game_sha256,
            "source_file": game.provenance.source_file,
            "file_sha256": game.provenance.file_sha256,
            "source_game_id": game.provenance.source_game_id,
            "line_number": line_number,
        })


class QuarantineWriter:
    def __init__(self, stream: IO[str]) -> None:
        self.stream = stream

    def write(self, source_file: str, file_sha256: str, line_number: int,
              error_code: str, message: str, raw_line: str,
              ply: int | None = None) -> None:
        _write_json_line(self.stream, {
            "source_file": source_file,
            "file_sha256": file_sha256,
            "line_number": line_number,
            "error_code": error_code,
            "message": message,
            "ply": ply,
            "raw_preview": raw_line[:1024],
            "raw_preview_truncated": len(raw_line) > 1024,
        })
