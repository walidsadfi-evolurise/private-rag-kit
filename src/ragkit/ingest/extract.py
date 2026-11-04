"""Turn files into `Document`s with a light structural markup.

Every extractor emits Markdown-ish text where headings start with '#'. The
chunker uses those headings to keep sections together and to label chunks,
so citations read "Policy › 3. Refunds" instead of "page 12".
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ragkit.models import Document


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        if txt.strip():
            pages.append(f"\n\n## Page {i}\n\n{txt}")
    return _clean("".join(pages))


def extract_docx(path: Path) -> str:
    import docx  # python-docx

    d = docx.Document(str(path))
    out: list[str] = []
    for p in d.paragraphs:
        style = (p.style.name or "").lower() if p.style is not None else ""
        txt = p.text.strip()
        if not txt:
            continue
        m = re.match(r"heading (\d)", style)
        if m:
            out.append("#" * min(int(m.group(1)), 6) + " " + txt)
        elif style == "title":
            out.append("# " + txt)
        else:
            out.append(txt)
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                out.append(" | ".join(cells))
    return _clean("\n\n".join(out))


def extract_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()
    body = soup.find("main") or soup.find("article") or soup.body or soup
    out: list[str] = []
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "pre", "blockquote"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            out.append("#" * int(el.name[1]) + " " + txt)
        else:
            out.append(txt)
    return _clean("\n\n".join(out))


def extract_text(path: Path) -> str:
    return _clean(path.read_text(encoding="utf-8", errors="replace"))


def title_for(path: Path, text: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return path.stem.replace("_", " ").replace("-", " ").strip() or path.name


def extract_document(path: Path, source: str | None = None) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = extract_pdf(path)
    elif suffix == ".docx":
        text = extract_docx(path)
    elif suffix in {".html", ".htm"}:
        text = extract_html(path)
    elif suffix in {".md", ".markdown", ".txt"}:
        text = extract_text(path)
    else:
        raise ValueError(f"unsupported file type: {path.suffix}")
    src = source or str(path)
    return Document(
        doc_id=Document.make_id(src, text),
        source=src,
        title=title_for(path, text),
        text=text,
        metadata={"suffix": suffix, "bytes": str(path.stat().st_size)},
    )


def iter_documents(root: Path, base: Path | None = None) -> Iterator[Document]:
    """Yield documents for every supported file under `root` (or `root` itself)."""
    from ragkit.ingest import is_supported

    base = base or (root if root.is_dir() else root.parent)
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for p in paths:
        if not is_supported(p):
            continue
        yield extract_document(p, source=str(p.relative_to(base)))
