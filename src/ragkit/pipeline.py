"""Wires settings → embedder / store / llm → Retriever / Answerer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragkit.answer import Answerer
from ragkit.config import Settings
from ragkit.embed import Embedder, make_embedder
from ragkit.ingest import chunk_document, iter_documents
from ragkit.llm import LLM, make_llm
from ragkit.retrieval import Retriever
from ragkit.store import Store, make_store


@dataclass
class IngestReport:
    documents: int
    chunks: int
    skipped: list[str]


class Pipeline:
    def __init__(self, settings: Settings, store: Store, embedder: Embedder, llm: LLM) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.retriever = Retriever(store, embedder, candidates=settings.candidates)
        self.answerer = Answerer(self.retriever, llm, k=settings.top_k)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Pipeline:
        s = settings or Settings()
        key = s.openai_api_key if s.embed_provider == "openai" else None
        embedder = make_embedder(s.embed_provider, s.embed_model, s.embed_base_url, key)
        store = make_store(s.store_url)
        llm_key = {"openai": s.openai_api_key, "anthropic": s.anthropic_api_key}.get(s.llm_provider)
        llm = make_llm(s.llm_provider, s.llm_model, llm_key, s.llm_base_url)
        return cls(s, store, embedder, llm)

    def ingest(self, path: Path) -> IngestReport:
        n_docs = n_chunks = 0
        skipped: list[str] = []
        for doc in iter_documents(path):
            chunks = chunk_document(doc, self.settings.chunk_chars, self.settings.chunk_overlap)
            if not chunks:
                skipped.append(doc.source)
                continue
            vectors = self.embedder.embed([f"{c.title}\n{c.heading}\n{c.text}" for c in chunks])
            self.store.upsert(chunks, vectors)
            n_docs += 1
            n_chunks += len(chunks)
        return IngestReport(n_docs, n_chunks, skipped)
