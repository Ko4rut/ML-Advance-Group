"""Stream shared Silver records independently of the generating algorithm."""

from ..base import BaseJsonlReader
from .models import SilverGame, SilverTransition


class SilverGameReader(BaseJsonlReader[SilverGame]):
    @property
    def schema_model(self) -> type[SilverGame]:
        return SilverGame


class SilverTransitionReader(BaseJsonlReader[SilverTransition]):
    @property
    def schema_model(self) -> type[SilverTransition]:
        return SilverTransition
