"""Grounded answering with mandatory citations and an explicit refusal path.

Rules the generator is held to, and that `verify_citations` enforces after the
fact: every sentence must carry at least one [n] marker that points at a
retrieved source; a marker that points nowhere is a hallucination and the
answer is downgraded to a refusal. A model that says NO_ANSWER is believed.
"""

from __future__ import annotations

import re

from ragkit.llm import LLM
from ragkit.models import Answer, Hit
from ragkit.retrieval import Retriever

REFUSAL_SENTINEL = "NO_ANSWER"
REFUSAL_TEXT = {
    "en": "I can't answer that from the documents I have access to.",
    "fr": "Je ne peux pas répondre à cette question à partir des documents disponibles.",
}

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.
Rules:
1. Every sentence you write must end with one or more citations like [1] or [2][4].
2. Cite only source numbers that appear in the prompt. Never invent a number.
3. If the sources do not contain the answer, reply with exactly: NO_ANSWER
4. Do not use outside knowledge. Do not speculate. Quote figures exactly as written.
5. Answer in the language of the question. Be concise."""

_CITE = re.compile(r"\[(\d+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TRAILING_CITE = re.compile(r"([.!?])((?:\s*\[\d+\])+)")


def normalise_citations(text: str) -> str:
    """Move 'sentence. [1]' to 'sentence [1].' so a citation always belongs to the sentence before it."""
    return _TRAILING_CITE.sub(lambda m: m.group(2).strip() and f" {m.group(2).strip()}{m.group(1)}", text)


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        where = c.title + (f" › {c.heading}" if c.heading else "")
        blocks.append(f"[{i}] {where}\n{c.text}")
    sources = "\n\n".join(blocks)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def verify_citations(text: str, n_sources: int) -> tuple[bool, list[int]]:
    """Return (every sentence cited with valid numbers, distinct cited indexes 1-based)."""
    text = normalise_citations(text)
    cited = [int(m) for m in _CITE.findall(text)]
    if not cited:
        return False, []
    if any(n < 1 or n > n_sources for n in cited):
        return False, []
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if not all(_CITE.search(s) for s in sentences):
        return False, []
    return True, sorted(set(cited))


def _lang(question: str) -> str:
    fr_markers = (
        " le ",
        " la ",
        " les ",
        " est ",
        " quel",
        " comment ",
        " pourquoi ",
        "é",
        "è",
        " du ",
        " des ",
    )
    q = f" {question.lower()} "
    return "fr" if sum(m in q for m in fr_markers) >= 2 else "en"


class Answerer:
    def __init__(self, retriever: Retriever, llm: LLM, k: int = 5, min_score: float = 0.0) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.min_score = min_score

    def ask(self, question: str) -> Answer:
        hits = [h for h in self.retriever.search(question, k=self.k) if h.score >= self.min_score]
        lang = _lang(question)
        if not hits:
            return Answer(question, REFUSAL_TEXT[lang], [], True, self.llm.name, [])

        raw = self.llm.complete(SYSTEM_PROMPT, build_prompt(question, hits)).strip()
        if not raw or REFUSAL_SENTINEL in raw:
            return Answer(question, REFUSAL_TEXT[lang], [], True, self.llm.name, hits)

        ok, cited = verify_citations(raw, len(hits))
        if not ok:
            return Answer(question, REFUSAL_TEXT[lang], [], True, self.llm.name, hits)

        text = normalise_citations(raw)
        return Answer(question, text, [hits[i - 1].chunk for i in cited], False, self.llm.name, hits)
