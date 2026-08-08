"""Corpus sources, fetchers and provenance."""

from .fetch import Document, fetch_source, read_jsonl, write_jsonl
from .sources import SOURCES, SOURCES_BY_ID, Source

__all__ = ["Document", "Source", "SOURCES", "SOURCES_BY_ID", "fetch_source", "read_jsonl", "write_jsonl"]
