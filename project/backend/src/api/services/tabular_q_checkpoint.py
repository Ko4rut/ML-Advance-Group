"""Fetch and cache a Tabular Q checkpoint for API inference.

Checkpoint sources:
1. XIANGQI_TABULAR_Q_CHECKPOINT: direct local file, including mounted Drive.
2. XIANGQI_TABULAR_Q_DRIVE_FILE_ID: Google Drive file ID, downloaded through
   Google Application Default Credentials (ADC) and cached locally.

No OAuth tokens, credentials or Drive URLs are stored in the checkpoint.
The service never writes to the source Drive file.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Callable

_DRIVE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{10,128}$")
_MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_DRIVE_READ_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

CheckpointDownloader = Callable[[str, Path], None]


class TabularQCheckpointSource:
    """Select a local checkpoint or download a Drive checkpoint once per ID."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        drive_file_id: str | None = None,
        cache_dir: str | Path | None = None,
        downloader: CheckpointDownloader | None = None,
    ) -> None:
        configured_path = checkpoint_path
        if configured_path is None:
            configured_path = os.getenv("XIANGQI_TABULAR_Q_CHECKPOINT")
        configured_id = drive_file_id
        if configured_id is None:
            configured_id = os.getenv("XIANGQI_TABULAR_Q_DRIVE_FILE_ID")
        configured_cache = cache_dir
        if configured_cache is None:
            configured_cache = os.getenv("XIANGQI_TABULAR_Q_CACHE_DIR")

        self.checkpoint_path = (
            Path(configured_path).expanduser() if configured_path else None
        )
        self.drive_file_id = configured_id.strip() if configured_id else None
        self.cache_dir = (
            Path(configured_cache).expanduser()
            if configured_cache
            else Path.home() / ".cache" / "xiangqi-ai" / "tabular-q"
        )
        self.downloader = downloader or download_drive_checkpoint

    def resolve(self) -> Path:
        """Return a verified-present path; never silently fall back to a model."""
        if self.checkpoint_path is not None:
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Tabular Q checkpoint not found: {self.checkpoint_path}"
                )
            return self.checkpoint_path

        if not self.drive_file_id:
            raise ValueError(
                "Tabular Q checkpoint not configured; set either "
                "XIANGQI_TABULAR_Q_CHECKPOINT (local file) or "
                "XIANGQI_TABULAR_Q_DRIVE_FILE_ID (private Drive file ID)"
            )
        if _DRIVE_ID_PATTERN.fullmatch(self.drive_file_id) is None:
            raise ValueError("Invalid XIANGQI_TABULAR_Q_DRIVE_FILE_ID")

        cached = self.cache_dir / f"drive_{self.drive_file_id}.json.gz"
        if cached.is_file() and cached.stat().st_size > 0:
            return cached

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.cache_dir,
                prefix=".tabular_q_",
                suffix=".downloading",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
            self.downloader(self.drive_file_id, temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ValueError("Drive checkpoint download was empty")
            if temporary.stat().st_size > _MAX_CHECKPOINT_BYTES:
                raise ValueError("Drive checkpoint exceeds maximum allowed size")
            os.replace(temporary, cached)
            temporary = None
            return cached
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def download_drive_checkpoint(file_id: str, destination: Path) -> None:
    """Read one private Google Drive blob using Google ADC.

    Requires the optional `drive` project extra, and the authenticated
    principal must have read permission on the specific file.
    """
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive authentication is not installed. "
            "Run: python -m pip install -e '.[api,drive]'"
        ) from exc

    try:
        credentials, _ = google.auth.default(scopes=[_DRIVE_READ_SCOPE])
        session = AuthorizedSession(credentials)
    except Exception as exc:
        raise RuntimeError(
            "Google Drive authentication failed. Configure Application Default "
            "Credentials with read access to the checkpoint file."
        ) from exc
    response = None
    try:
        response = session.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"alt": "media", "supportsAllDrives": "true"},
            stream=True,
            timeout=(10, 60),
        )
        response.raise_for_status()
        received = 0
        with destination.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                received += len(chunk)
                if received > _MAX_CHECKPOINT_BYTES:
                    raise ValueError("Drive checkpoint exceeds maximum allowed size")
                stream.write(chunk)
        if received == 0:
            raise ValueError("Downloaded Drive checkpoint is empty")
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Unable to download Tabular Q checkpoint from Google Drive. "
            "Check the file ID, Drive permissions and network connection."
        ) from exc
    finally:
        if response is not None:
            response.close()
        session.close()
