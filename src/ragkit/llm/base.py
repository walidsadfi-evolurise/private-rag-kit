from __future__ import annotations

import re
from abc import ABC, abstractmethod


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ0-9]+", text.lower())


class LLM(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str: ...


class FakeLLM(LLM):
    """Deterministic stand-in for tests.

    It behaves like a well-instructed model: it picks the numbered source that
    shares the most content words with the question, answers with that
    source's most relevant sentence and cites it; when no source shares a
    content word with the question it emits the refusal sentinel.
    """

    name = "fake"

    def __init__(self, canned: str | None = None) -> None:
        self.canned = canned
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        if self.canned is not None:
            return self.canned
        question = user.rsplit("Question:", 1)[-1] if "Question:" in user else user
        qwords = {w for w in _words(question) if len(w) >= 4}
        sources = re.findall(r"^\[(\d+)\]\s.*?\n(.*?)(?=^\[\d+\]\s|\nQuestion:|\Z)", user, flags=re.M | re.S)
        best: tuple[int, str, str] | None = None
        for n, body in sources:
            for sentence in re.split(r"(?<=[.!?])\s+", body.strip()):
                overlap = len(qwords & set(_words(sentence)))
                if overlap and (best is None or overlap > best[0]):
                    best = (overlap, n, sentence)
        if best is None:
            return "NO_ANSWER"
        _, n, sentence = best
        return f"{sentence.rstrip('.!?')} [{n}]."
