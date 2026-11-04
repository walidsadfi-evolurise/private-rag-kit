from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.ingest import chunk_document, extract_document, is_supported, iter_documents
from ragkit.ingest.chunk import _sections, _split_long
from ragkit.models import Document

CORPUS = Path(__file__).parent / "fixtures" / "corpus"


def test_iter_documents_reads_all_supported_files() -> None:
    docs = list(iter_documents(CORPUS))
    assert sorted(d.source for d in docs) == ["onboarding.html", "refund-policy.md", "sla.md"]
    assert all(d.doc_id and d.title for d in docs)


def test_html_extraction_drops_boilerplate_and_keeps_headings() -> None:
    doc = extract_document(CORPUS / "onboarding.html")
    assert doc.title == "Employee onboarding"
    assert "var x" not in doc.text
    assert "Home · Docs" not in doc.text
    assert "## First day" in doc.text


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.xyz"
    f.write_text("hello")
    assert not is_supported(f)
    with pytest.raises(ValueError):
        extract_document(f)


def test_sections_follow_heading_hierarchy() -> None:
    text = "intro\n\n# A\n\nbody a\n\n## A.1\n\nbody a1\n\n# B\n\nbody b"
    secs = _sections(text)
    assert secs == [("", "intro"), ("A", "body a"), ("A › A.1", "body a1"), ("B", "body b")]


def test_split_long_respects_max_and_hard_splits_giant_sentences() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    parts = _split_long(text, 100)
    assert all(len(p) <= 100 for p in parts)
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")
    giant = "x" * 350
    assert [len(p) for p in _split_long(giant, 100)] == [100, 100, 100, 50]


def test_chunks_keep_heading_and_overlap() -> None:
    body = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(6))
    doc = Document(doc_id="d", source="s", title="T", text=f"# T\n\n## Sec\n\n{body}")
    chunks = chunk_document(doc, max_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    assert all(c.heading == "Sec" for c in chunks)  # title is not repeated in the path
    assert [c.position for c in chunks] == list(range(len(chunks)))
    # overlap: chunk i+1 starts with the tail of chunk i's own text
    assert chunks[1].text.startswith(chunks[0].text[-50:].strip()[:20])
    assert chunks[0].citation == "T › Sec (s#chunk0)"
