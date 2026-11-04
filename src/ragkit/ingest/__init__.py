"""Text extraction for the supported file types."""

from __future__ import annotations

from pathlib import Path

from ragkit.ingest.chunk import chunk_document
from ragkit.ingest.extract import extract_document, iter_documents

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".html", ".htm", ".md", ".markdown", ".txt"}


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


__all__ = ["extract_document", "iter_documents", "chunk_document", "is_supported", "SUPPORTED_SUFFIXES"]
