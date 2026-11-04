from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ragkit.models import Chunk


class Store(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> int: ...

    @abstractmethod
    def vector_search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def lexical_search(self, query: str, k: int) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def documents(self) -> list[tuple[str, str, str, int]]:
        """[(doc_id, source, title, chunk_count)]"""
