"""Structure-aware chunking.

Split on headings first, then on paragraphs, then on sentences, so a chunk
never straddles two sections and rarely cuts a sentence. Chunks carry their
heading path for citations and get a small overlap with the previous chunk so
a fact that sits on a boundary is still retrievable.
"""

from __future__ import annotations

import re

from ragkit.models import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9])")


def _sections(text: str) -> list[tuple[str, str]]:
    """Return [(heading_path, body)] in document order."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    pre = text[: matches[0].start()].strip()
    if pre:
        sections.append(("", pre))
    for i, m in enumerate(matches):
        level, title = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        path = " › ".join(t for _, t in stack)
        if body:
            sections.append((path, body))
    return sections


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    buf = ""
    for sentence in _SENTENCE.split(text):
        if len(buf) + len(sentence) + 1 > max_chars and buf:
            parts.append(buf.strip())
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        parts.append(buf.strip())
    # A single sentence longer than max_chars: hard split.
    out: list[str] = []
    for p in parts:
        out.extend(p[i : i + max_chars] for i in range(0, len(p), max_chars)) if len(
            p
        ) > max_chars else out.append(p)
    return out


def chunk_document(doc: Document, max_chars: int = 1200, overlap_chars: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    position = 0
    for heading, body in _sections(doc.text):
        # The top-level title is already carried by the chunk; don't repeat it in the heading path.
        if heading == doc.title:
            heading = ""
        elif heading.startswith(f"{doc.title} › "):
            heading = heading[len(doc.title) + 3 :]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buf = ""
        pieces: list[str] = []
        for para in paragraphs:
            if len(buf) + len(para) + 2 <= max_chars:
                buf = f"{buf}\n\n{para}".strip()
                continue
            if buf:
                pieces.append(buf)
            buf = ""
            if len(para) > max_chars:
                pieces.extend(_split_long(para, max_chars))
            else:
                buf = para
        if buf:
            pieces.append(buf)

        prev_tail = ""
        for piece in pieces:
            text = f"{prev_tail} {piece}".strip() if prev_tail else piece
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{position}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    heading=heading,
                    text=text,
                    position=position,
                )
            )
            position += 1
            prev_tail = piece[-overlap_chars:] if overlap_chars > 0 and len(piece) > overlap_chars else ""
    return chunks
