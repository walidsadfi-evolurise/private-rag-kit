"""Offline evaluation against a YAML set of questions.

Each case: question, and either `expected_sources` (substrings of chunk
sources that should be retrieved) and/or `expected_answer_contains`
(substrings the answer must contain), or `expect_refusal: true`.

Metrics: hit@k (a relevant source is in the top k), citation rate (answers whose
citations all point to relevant sources), refusal precision/recall, and plain
answer accuracy. These are the numbers to watch when you change chunking,
embedders or prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ragkit.answer import Answerer


@dataclass
class Case:
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_answer_contains: list[str] = field(default_factory=list)
    expect_refusal: bool = False


@dataclass
class CaseResult:
    case: Case
    hit: bool
    answered: bool
    refused: bool
    citations_ok: bool
    answer_ok: bool
    answer: str


@dataclass
class EvalReport:
    results: list[CaseResult]

    @property
    def n(self) -> int:
        return len(self.results)

    def rate(self, pred: Any) -> float:
        rel = [r for r in self.results if pred(r) is not None]
        return sum(1 for r in rel if pred(r)) / len(rel) if rel else 0.0

    def summary(self) -> dict[str, float]:
        retrieval_cases = [r for r in self.results if r.case.expected_sources]
        answer_cases = [r for r in self.results if not r.case.expect_refusal]
        refusal_cases = [r for r in self.results if r.case.expect_refusal]
        answered_cases = [r for r in self.results if not r.refused]

        def mean(items: list[CaseResult], fn: Any) -> float:
            return sum(1 for r in items if fn(r)) / len(items) if items else 0.0

        return {
            "cases": float(self.n),
            "hit_at_k": mean(retrieval_cases, lambda r: r.hit),
            "answer_accuracy": mean(answer_cases, lambda r: r.answer_ok and not r.refused),
            "citation_rate": mean(answered_cases, lambda r: r.citations_ok),
            "refusal_recall": mean(refusal_cases, lambda r: r.refused),
            "refusal_precision": mean(
                [r for r in self.results if r.refused], lambda r: r.case.expect_refusal
            ),
        }


def load_cases(path: Path) -> list[Case]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [Case(**c) for c in data]


def evaluate(answerer: Answerer, cases: list[Case]) -> EvalReport:
    results: list[CaseResult] = []
    for case in cases:
        ans = answerer.ask(case.question)
        retrieved_sources = [h.chunk.source for h in ans.retrieved]
        hit = any(any(exp in src for src in retrieved_sources) for exp in case.expected_sources)
        cited_sources = [c.source for c in ans.citations]
        citations_ok = (
            bool(cited_sources)
            and all(any(exp in src for exp in case.expected_sources) for src in cited_sources)
            if case.expected_sources
            else bool(cited_sources)
        )
        answer_ok = all(s.lower() in ans.text.lower() for s in case.expected_answer_contains)
        results.append(
            CaseResult(
                case=case,
                hit=hit,
                answered=not ans.refused,
                refused=ans.refused,
                citations_ok=citations_ok if not ans.refused else False,
                answer_ok=answer_ok,
                answer=ans.text,
            )
        )
    return EvalReport(results)
