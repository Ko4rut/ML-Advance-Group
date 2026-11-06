"""Immutable dataset ingestion into the Bronze layer."""

from .ingest import BronzeIngester, BronzeIngestReport

__all__ = ["BronzeIngester", "BronzeIngestReport"]
