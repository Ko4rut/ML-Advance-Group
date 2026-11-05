"""Versioned, cross-layer Xiangqi data primitives."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SideName = Literal["red", "black"]
SourceType = Literal["human", "alphabeta", "computer", "opening_book", "random", "unknown"]
Split = Literal["train", "val", "test"]
PieceCode = Annotated[int, Field(strict=True, ge=-7, le=7)]
Board = Annotated[list[PieceCode], Field(min_length=90, max_length=90)]
Square = Annotated[int, Field(strict=True, ge=0, lt=90)]
ActionId = Annotated[int, Field(strict=True, ge=0, lt=8100)]


class StrictModel(BaseModel):
    """Reject unexpected normalized fields; raw Bronze records are exempt."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VersionedRecord(StrictModel):
    schema_version: Literal[1] = 1


class MoveModel(StrictModel):
    source: Square
    target: Square


class Provenance(StrictModel):
    """Trace a normalized game back to an immutable Bronze source file."""

    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str | None = None
    source_game_id: str | None = None
