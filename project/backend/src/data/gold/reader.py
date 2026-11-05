"""Readers for independent ML objectives."""

from ..base import BaseJsonlReader
from .models import OfflineRLSample, PolicySample


class PolicyReader(BaseJsonlReader[PolicySample]):
    @property
    def schema_model(self) -> type[PolicySample]:
        return PolicySample


class OfflineRLReader(BaseJsonlReader[OfflineRLSample]):
    @property
    def schema_model(self) -> type[OfflineRLSample]:
        return OfflineRLSample
