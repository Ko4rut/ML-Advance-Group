"""Bronze JSONL readers; external formats will get dedicated adapters."""

from ..base import BaseJsonlReader
from .models import AlphaBetaRawGame, BronzeRecord


class BronzeReader(BaseJsonlReader[BronzeRecord]):
    @property
    def schema_model(self) -> type[BronzeRecord]:
        return BronzeRecord


class AlphaBetaRawReader(BaseJsonlReader[AlphaBetaRawGame]):
    @property
    def schema_model(self) -> type[AlphaBetaRawGame]:
        return AlphaBetaRawGame
