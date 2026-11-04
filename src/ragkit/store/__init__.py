"""Vector + lexical stores. The memory store is the reference implementation and the test double."""

from __future__ import annotations

from ragkit.store.base import Store
from ragkit.store.memory import MemoryStore


def make_store(url: str) -> Store:
    if url.startswith("memory://"):
        path = url.removeprefix("memory://")
        return MemoryStore(path or None)
    if url.startswith("postgresql"):
        from ragkit.store.pgvector import PgVectorStore

        return PgVectorStore(url)
    raise ValueError(f"unknown store url: {url}")


__all__ = ["Store", "MemoryStore", "make_store"]
