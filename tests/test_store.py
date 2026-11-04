from __future__ import annotations

from pathlib import Path

import numpy as np

from ragkit.embed import HashingEmbedder
from ragkit.ingest import chunk_document, extract_document
from ragkit.store import MemoryStore, make_store

CORPUS = Path(__file__).parent / "fixtures" / "corpus"


def _chunks_and_vectors(name: str):  # type: ignore[no-untyped-def]
    chunks = chunk_document(extract_document(CORPUS / name, source=name))
    return chunks, HashingEmbedder(dim=64).embed([c.text for c in chunks])


def test_upsert_replaces_same_document() -> None:
    s = MemoryStore()
    chunks, vecs = _chunks_and_vectors("sla.md")
    s.upsert(chunks, vecs)
    s.upsert(chunks, vecs)
    assert s.count() == len(chunks)
    assert s.documents() == [(chunks[0].doc_id, "sla.md", "Support SLA", len(chunks))]


def test_delete_document_removes_rows_and_vectors() -> None:
    s = MemoryStore()
    a, va = _chunks_and_vectors("sla.md")
    b, vb = _chunks_and_vectors("refund-policy.md")
    s.upsert(a, va)
    s.upsert(b, vb)
    assert s.delete_document(a[0].doc_id) == len(a)
    assert s.count() == len(b) == s.vectors.shape[0]
    assert s.delete_document("nope") == 0


def test_persistence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "idx"
    s = MemoryStore(str(path))
    chunks, vecs = _chunks_and_vectors("refund-policy.md")
    s.upsert(chunks, vecs)
    again = make_store(f"memory://{path}")
    assert again.count() == len(chunks)
    hits = again.lexical_search("withdraw within 14 days", k=2)
    assert hits and "14 days" in hits[0][0].text
    q = HashingEmbedder(dim=64).embed(["refund original payment method"])[0]
    vhits = again.vector_search(q, k=1)
    assert vhits and isinstance(vhits[0][1], float)


def test_empty_store_searches_return_empty() -> None:
    s = MemoryStore()
    assert s.lexical_search("x", 3) == [] and s.vector_search(np.zeros(8, dtype=np.float32), 3) == []


def test_ingest_skips_unchanged_documents_unless_forced(tmp_path: Path) -> None:
    from ragkit.config import Settings
    from ragkit.llm import FakeLLM
    from ragkit.pipeline import Pipeline

    src = tmp_path / "docs"
    src.mkdir()
    (src / "a.md").write_text("# A\n\nFirst version of the policy.")
    p = Pipeline(Settings(store_url="memory://"), MemoryStore(), HashingEmbedder(), FakeLLM())
    first = p.ingest(src)
    assert first.documents == 1 and first.unchanged == []

    again = p.ingest(src)
    assert again.documents == 0 and again.unchanged == ["a.md"]
    assert p.store.count() == first.chunks

    (src / "a.md").write_text("# A\n\nSecond version of the policy, now longer.")
    changed = p.ingest(src)
    assert changed.documents == 1 and changed.unchanged == []
    assert len(p.store.documents()) == 1  # old version replaced, not duplicated

    forced = p.ingest(src, force=True)
    assert forced.documents == 1 and forced.unchanged == []
