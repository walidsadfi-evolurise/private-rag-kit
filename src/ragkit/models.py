"""Core data types shared by ingestion, retrieval and generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A source file after text extraction, before chunking."""

    doc_id: str
    source: str  # path or URL, shown in citations
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def make_id(source: str, text: str) -> str:
        return hashlib.sha256(f"{source}\n{text}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit. `heading` is the nearest heading path, e.g. 'Title > Section'."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    heading: str
    text: str
    position: int  # order inside the document, 0-based

    @property
    def citation(self) -> str:
        where = f"{self.title}" + (f" › {self.heading}" if self.heading else "")
        return f"{where} ({self.source}#chunk{self.position})"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    # Where the hit came from; useful for debugging fusion.
    lexical_rank: int | None = None
    vector_rank: int | None = None


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Chunk]
    refused: bool
    model: str
    retrieved: list[Hit]
