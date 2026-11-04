"""Wires settings → embedder / store / llm → Retriever / Answerer."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    unchanged: list[str] = field(default_factory=list)


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

    def ingest(self, path: Path, force: bool = False) -> IngestReport:
        """Index every supported file under `path`.

        Documents are content-addressed (doc_id = hash of source + text), so a
        file that has not changed since the last run is skipped unless `force`.
        """
        n_docs = n_chunks = 0
        skipped: list[str] = []
        unchanged: list[str] = []
        existing = {source: doc_id for doc_id, source, *_ in self.store.documents()}
        for doc in iter_documents(path):
            previous = existing.get(doc.source)
            if previous == doc.doc_id and not force:
                unchanged.append(doc.source)
                continue
            if previous is not None and previous != doc.doc_id:
                # Same file, new content: drop the old version so it is not retrievable next to the new one.
                self.store.delete_document(previous)
            chunks = chunk_document(doc, self.settings.chunk_chars, self.settings.chunk_overlap)
            if not chunks:
                skipped.append(doc.source)
                continue
            vectors = self.embedder.embed([f"{c.title}\n{c.heading}\n{c.text}" for c in chunks])
            self.store.upsert(chunks, vectors)
            n_docs += 1
            n_chunks += len(chunks)
        return IngestReport(n_docs, n_chunks, skipped, unchanged)
