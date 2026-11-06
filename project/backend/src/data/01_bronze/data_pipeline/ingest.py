"""Ingest dataset files into immutable Bronze storage with SHA-256 manifests.

No chess parsing, dataset downloading, deduplication or Silver validation belongs
here. Files are copied byte-for-byte and existing content is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Sequence

from src.data.bronze.models import BronzeFileManifest

_CHUNK_BYTES = 1024 * 1024
_SOURCE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(slots=True)
class BronzeIngestReport:
    source: str
    discovered_files: int = 0
    copied_files: int = 0
    unchanged_files: int = 0
    empty_files: int = 0
    copied_bytes: int = 0

    def as_dict(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "discovered_files": self.discovered_files,
            "copied_files": self.copied_files,
            "unchanged_files": self.unchanged_files,
            "empty_files": self.empty_files,
            "copied_bytes": self.copied_bytes,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_files(root: Path, patterns: Sequence[str]) -> list[tuple[Path, Path]]:
    """Return (real source file, relative destination path), in stable order."""
    if root.is_symlink():
        raise ValueError(f"symlink sources are unsupported: {root}")
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        if not any(fnmatch(root.name, pattern) for pattern in patterns):
            return []
        return [(root, Path(root.name))]
    if not root.is_dir():
        raise ValueError(f"unsupported source: {root}")
    discovered: list[tuple[Path, Path]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink sources are unsupported: {path}")
        if path.is_file() and any(fnmatch(path.name, pattern) for pattern in patterns):
            discovered.append((path, path.relative_to(root)))
    return discovered


def _check_manifest(path: Path, expected_sha256: str, expected_source: str) -> None:
    """Never silently accept a manifest that describes another file."""
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = BronzeFileManifest.model_validate(payload)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid Bronze manifest: {path}") from exc
    if manifest.file_sha256 != expected_sha256 or manifest.source != expected_source:
        raise ValueError(f"Bronze manifest/source conflict: {path}")


class BronzeIngester:
    def ingest(
        self,
        inputs: Sequence[str | Path],
        destination_root: str | Path,
        *,
        source: str,
        source_version: str,
        patterns: Sequence[str] = ("*.jsonl",),
        license: str | None = None,
    ) -> BronzeIngestReport:
        """Copy inputs to <destination_root>/<source>/ without destructive writes.

        Files in source directories keep their relative paths. Re-ingesting an
        identical file is idempotent; a name collision with different bytes
        raises rather than replacing the original.
        """
        if not _SOURCE_PATTERN.fullmatch(source):
            raise ValueError("source must match [a-z][a-z0-9_-]*")
        if not isinstance(source_version, str) or not source_version.strip():
            raise ValueError("source_version must be non-empty")
        if not inputs:
            raise ValueError("at least one source file or directory is required")
        if not patterns or any(not pattern.strip() for pattern in patterns):
            raise ValueError("patterns must be non-empty")

        destination = Path(destination_root) / source
        files: list[tuple[Path, Path]] = []
        for raw_input in inputs:
            root = Path(raw_input)
            if root.is_dir() and destination.resolve().is_relative_to(root.resolve()):
                raise ValueError("Bronze destination must not be inside input directory")
            files.extend(_discover_files(root, patterns))

        # Detect duplicate target names before copying any bytes.
        seen: set[Path] = set()
        for _, relative in files:
            if relative in seen:
                raise ValueError(f"multiple inputs map to the same Bronze path: {relative}")
            seen.add(relative)

        report = BronzeIngestReport(source=source, discovered_files=len(files))
        if not files:
            return report

        destination.mkdir(parents=True, exist_ok=True)
        for origin, relative in files:
            if origin.stat().st_size == 0:
                report.empty_files += 1
                continue

            source_sha = sha256_file(origin)
            target = destination / relative
            if origin.resolve() == target.resolve():
                raise ValueError(f"source is already the destination: {origin}")
            target.parent.mkdir(parents=True, exist_ok=True)
            manifest_path = target.with_name(target.name + ".manifest.json")

            if target.exists():
                if not target.is_file() or sha256_file(target) != source_sha:
                    raise ValueError(f"Bronze collision with different content: {target}")
                _check_manifest(manifest_path, source_sha, source)
                report.unchanged_files += 1
            else:
                # 'xb' will fail rather than overwriting files created by another run.
                try:
                    with origin.open("rb") as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, _CHUNK_BYTES)
                    if sha256_file(target) != source_sha:
                        raise IOError(f"Bronze copy checksum mismatch: {target}")
                except FileExistsError as exc:
                    raise ValueError(f"Bronze target created concurrently: {target}") from exc
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
                report.copied_files += 1
                report.copied_bytes += target.stat().st_size

            if not manifest_path.exists():
                record = BronzeFileManifest(
                    source=source,
                    source_version=source_version,
                    source_file=relative.as_posix(),
                    file_sha256=source_sha,
                    ingested_at=datetime.now(timezone.utc),
                    license=license,
                    original_path=str(origin),
                    relative_path=relative.as_posix(),
                    size_bytes=target.stat().st_size,
                )
                with manifest_path.open("x", encoding="utf-8") as stream:
                    stream.write(record.model_dump_json(indent=2) + "\n")

        return report
