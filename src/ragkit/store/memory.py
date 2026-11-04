"""In-process store: numpy for vectors, BM25 for lexical. Optionally persisted to a .npz + .json pair."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ragkit.models import Chunk
from ragkit.store.base import Store

_TOKEN = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class MemoryStore(Store):
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._bm25: BM25Okapi | None = None
        if self.path and self.path.with_suffix(".json").exists():
            self._load()

    # -- persistence -------------------------------------------------------
    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.path.with_suffix(".npy"), self.vectors)
        self.path.with_suffix(".json").write_text(
            json.dumps([c.__dict__ for c in self.chunks]), encoding="utf-8"
        )

    def _load(self) -> None:
        assert self.path is not None
        self.chunks = [
            Chunk(**c) for c in json.loads(self.path.with_suffix(".json").read_text(encoding="utf-8"))
        ]
        npy = self.path.with_suffix(".npy")
        self.vectors = np.load(npy).astype(np.float32) if npy.exists() else np.zeros((0, 0), dtype=np.float32)
        self._bm25 = None

    def _index(self) -> BM25Okapi:
        if self._bm25 is None:
            corpus = [tokenize(f"{c.title} {c.heading} {c.text}") for c in self.chunks] or [["_"]]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    # -- Store -------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        doc_ids = {c.doc_id for c in chunks}
        for d in doc_ids:
            self.delete_document(d)
        self.chunks.extend(chunks)
        self.vectors = (
            vectors.astype(np.float32) if self.vectors.size == 0 else np.vstack([self.vectors, vectors])
        )
        self._bm25 = None
        self._save()

    def delete_document(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if self.vectors.size else self.vectors
            self._bm25 = None
            self._save()
        return removed

    def vector_search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks or self.vectors.size == 0:
            return []
        scores = self.vectors @ query.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def lexical_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scores = self._index().get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]

    def count(self) -> int:
        return len(self.chunks)

    def documents(self) -> list[tuple[str, str, str, int]]:
        seen: dict[str, tuple[str, str, int]] = {}
        for c in self.chunks:
            src, title, n = seen.get(c.doc_id, (c.source, c.title, 0))
            seen[c.doc_id] = (src, title, n + 1)
        return [(d, s, t, n) for d, (s, t, n) in seen.items()]
