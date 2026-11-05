"""Streaming JSONL readers shared by all Medallion layers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from pydantic import BaseModel, ValidationError

RecordT = TypeVar("RecordT", bound=BaseModel)


class BaseJsonlReader(ABC, Generic[RecordT]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    @abstractmethod
    def schema_model(self) -> type[RecordT]:
        """Pydantic schema to validate each JSONL row."""

    def read(self) -> Iterator[RecordT]:
        """Stream records and report the source file and line on bad input."""
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                try:
                    yield self.schema_model.model_validate(payload)
                except ValidationError as exc:
                    raise ValueError(
                        f"{self.path}:{line_number}: schema validation failed: {exc}"
                    ) from exc

    def __iter__(self) -> Iterator[RecordT]:
        return self.read()
