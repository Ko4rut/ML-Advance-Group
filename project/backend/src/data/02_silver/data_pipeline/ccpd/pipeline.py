"""Bronze CCPD PGN -> replayed/deduplicated Silver JSONL, one file per game."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from src.data.bronze.models import BronzeFileManifest
from src.data.models import Provenance
from src.environment import GameEngine, RulesConfig

from ..deduplication import GameDeduplicator
from ..errors import SilverValidationError
from ..replay import SilverReplayValidator
from ..writers import QuarantineWriter, SilverWriter
from .adapter import CCPDAdapter
from .decoder import decode_pgn
from .pgn_parser import parse_pgn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_inputs(inputs: Sequence[str | Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.pgn")))
        elif path.is_file() and path.suffix.lower() == ".pgn":
            files.append(path)
        else:
            raise FileNotFoundError(f"CCPD PGN file/directory not found: {path}")
    if not files:
        raise ValueError("No .pgn files discovered")
    if len({str(path.resolve()) for path in files}) != len(files):
        raise ValueError("Duplicate input paths are not allowed")
    return sorted(files)


class CCPDPipeline:
    def __init__(self, replay_validator: SilverReplayValidator | None = None) -> None:
        # Professional Xiangqi PGNs may legally continue after a repeated
        # position. The project engine's default threefold-draw threshold is
        # a simplified self-play rule, not tournament adjudication. Keep all
        # move-legality checks, but do not auto-stop recorded master games on
        # repetition/no-capture counters.
        self.validator = replay_validator or SilverReplayValidator(
            GameEngine(RulesConfig(repetition_count=10**9, no_capture_limit=10**9))
        )
        self.adapter = CCPDAdapter(self.validator.engine)

    def run(
        self,
        input_files: Sequence[str | Path],
        output_dir: str | Path,
        *,
        source_version: str = "ccpd-v1",
        license: str = "CC-BY-4.0",
        require_manifest: bool = True,
    ) -> dict[str, object]:
        files = _resolve_inputs(input_files)
        output = Path(output_dir)
        output_names = (
            "games.jsonl", "transitions.jsonl", "duplicates.jsonl",
            "quarantine.jsonl", "report.json",
        )
        if any((output / name).exists() for name in output_names):
            raise FileExistsError(f"Silver output already exists: {output}")
        # No writes to Bronze. Input may be a folder, but output should stay separate.
        for path in files:
            if output.resolve().is_relative_to(path.parent.resolve()):
                raise ValueError("Silver output must not be inside a Bronze PGN directory")
        output.mkdir(parents=True, exist_ok=True)
        report: dict[str, object] = {
            "source": "ccpd", "source_files": len(files),
            "accepted_games": 0, "transitions": 0,
            "duplicate_games": 0, "quarantined_records": 0,
            "engine_terminal_games": 0, "recorded_results": 0,
            "quarantine_error_codes": {}, "encoding_counts": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        errors: Counter[str] = Counter()
        encodings: Counter[str] = Counter()
        dedup = GameDeduplicator()

        with ExitStack() as stack:
            streams = [stack.enter_context((output / name).open("x", encoding="utf-8"))
                       for name in output_names[:4]]
            writer = SilverWriter(*streams[:3])
            quarantine = QuarantineWriter(streams[3])
            for file in files:
                digest = _sha256(file)
                preview = ""
                try:
                    manifest_file = file.with_name(file.name + ".manifest.json")
                    if require_manifest:
                        if not manifest_file.is_file():
                            raise SilverValidationError("MISSING_MANIFEST", "Bronze sidecar missing")
                        try:
                            manifest = BronzeFileManifest.model_validate_json(
                                manifest_file.read_text(encoding="utf-8")
                            )
                        except (OSError, ValidationError, ValueError) as exc:
                            raise SilverValidationError("INVALID_MANIFEST", str(exc)) from exc
                        if manifest.file_sha256 != digest or manifest.source != "ccpd":
                            raise SilverValidationError(
                                "BRONZE_HASH_MISMATCH", "Bronze manifest SHA/source mismatch"
                            )
                        file_version = manifest.source_version
                        file_license = manifest.license
                    else:
                        file_version = source_version
                        file_license = license
                    raw = file.read_bytes()
                    text, codec = decode_pgn(raw)
                    preview = text
                    record = parse_pgn(text)
                    provenance = Provenance(
                        source="ccpd", source_version=file_version,
                        source_file=file.name, file_sha256=digest,
                        license=file_license, source_game_id=file.stem,
                    )
                    game = self.adapter.convert(record, provenance)
                    transitions = self.validator.replay(game)
                except (SilverValidationError, ValidationError) as exc:
                    code = exc.code if isinstance(exc, SilverValidationError) else "INVALID_SCHEMA"
                    ply = exc.ply if isinstance(exc, SilverValidationError) else None
                    quarantine.write(file.name, digest, 1, code, str(exc), preview, ply)
                    report["quarantined_records"] += 1
                    errors[code] += 1
                    continue
                original_id = dedup.register(game)
                if original_id is not None:
                    writer.write_duplicate(game, original_id, 1)
                    report["duplicate_games"] += 1
                    continue
                writer.write_game(game, transitions)
                report["accepted_games"] += 1
                report["transitions"] += len(transitions)
                if transitions[-1].terminated:
                    report["engine_terminal_games"] += 1
                if game.outcome.status == "finished":
                    report["recorded_results"] += 1
                encodings[codec] += 1
        report["quarantine_error_codes"] = dict(errors)
        report["encoding_counts"] = dict(encodings)
        if report["source_files"] != (report["accepted_games"]
                                     + report["duplicate_games"]
                                     + report["quarantined_records"]):
            raise AssertionError("Silver CCPD report has inconsistent totals")
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return report
