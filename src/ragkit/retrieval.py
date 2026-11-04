"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion.

RRF is used instead of score interpolation because BM25 and cosine scores live
on different scales; rank-based fusion needs no tuning and is robust to that.
"""

from __future__ import annotations

from ragkit.embed import Embedder
from ragkit.models import Hit
from ragkit.store import Store


def rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, candidates: int = 20) -> None:
        self.store = store
        self.embedder = embedder
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[Hit]:
        query = query.strip()
        if not query:
            return []
        qvec = self.embedder.embed([query])[0]
        vec_hits = self.store.vector_search(qvec, self.candidates)
        lex_hits = self.store.lexical_search(query, self.candidates)

        by_id = {c.chunk_id: c for c, _ in vec_hits + lex_hits}
        vec_rank = {c.chunk_id: i + 1 for i, (c, _) in enumerate(vec_hits)}
        lex_rank = {c.chunk_id: i + 1 for i, (c, _) in enumerate(lex_hits)}
        fused = rrf([list(vec_rank), list(lex_rank)])

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], by_id[kv[0]].position))
        return [
            Hit(chunk=by_id[cid], score=score, lexical_rank=lex_rank.get(cid), vector_rank=vec_rank.get(cid))
            for cid, score in ordered[:k]
        ]
