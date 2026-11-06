"""Bronze manifest and minimally inspected Alpha-Beta raw records."""

from __future__ import annotations

from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from ..models import Board, VersionedRecord


class BronzeRecord(VersionedRecord):
    """Sidecar metadata for a raw record or file; preserve original bytes separately."""

    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ingested_at: AwareDatetime
    license: str | None = None
    raw_record: dict[str, Any] | list[Any] | str | None = None


class AlphaBetaRawGame(BaseModel):
    """Loose adapter for the existing batch_games JSONL, not a Silver validation."""

    model_config = ConfigDict(extra="allow")
    schema_version: int = Field(ge=1)
    game_id: str = Field(min_length=1)
    initial_board: Board
    moves: list[dict[str, Any]]
    outcome: dict[str, Any]


class BronzeFileManifest(VersionedRecord):
    """File-level Bronze metadata; accepts legacy sidecars from Colab."""

    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ingested_at: AwareDatetime
    license: str | None = None
    original_path: str | None = None
    relative_path: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
