from __future__ import annotations

import numpy as np

from ragkit.answer import REFUSAL_TEXT, Answerer, build_prompt, verify_citations
from ragkit.embed import HashingEmbedder
from ragkit.llm import FakeLLM
from ragkit.pipeline import Pipeline
from ragkit.retrieval import rrf


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    e = HashingEmbedder(dim=64)
    a = e.embed(["refund within 14 days", "refund within 14 days", "maintenance window"])
    assert np.allclose(a[0], a[1])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert a[0] @ a[2] < a[0] @ a[1]


def test_rrf_prefers_items_present_in_both_lists() -> None:
    scores = rrf([["a", "b", "c"], ["b", "d"]])
    assert scores["b"] > scores["a"] > scores["c"]
    assert "d" in scores


def test_hybrid_search_finds_the_right_document(pipeline: Pipeline) -> None:
    hits = pipeline.retriever.search("first response time for a Priority 1 incident", k=3)
    assert hits
    assert hits[0].chunk.source == "sla.md"
    assert hits[0].lexical_rank is not None or hits[0].vector_rank is not None


def test_search_with_empty_query_returns_nothing(pipeline: Pipeline) -> None:
    assert pipeline.retriever.search("   ") == []


def test_verify_citations() -> None:
    assert verify_citations("Refunds take 10 days [1]. Returns cost money [2][1].", 2) == (True, [1, 2])
    assert verify_citations("Refunds take 10 days [1]. Returns cost money.", 2) == (False, [])
    assert verify_citations("Refunds take 10 days [7].", 2) == (False, [])
    assert verify_citations("No citation at all.", 2) == (False, [])


def test_answer_is_grounded_and_cited(pipeline: Pipeline) -> None:
    ans = pipeline.answerer.ask("How fast are refunds issued?")
    assert not ans.refused
    assert "10 business days" in ans.text
    assert ans.citations and ans.citations[0].source == "refund-policy.md"
    assert "[1]" in ans.text


def test_prompt_numbers_sources_in_order(pipeline: Pipeline) -> None:
    hits = pipeline.retriever.search("maintenance windows", k=2)
    prompt = build_prompt("q", hits)
    assert prompt.startswith("Sources:\n\n[1] ")
    assert "[2] " in prompt and prompt.endswith("Question: q")


def test_refuses_when_model_says_no_answer(pipeline: Pipeline) -> None:
    pipeline.answerer.llm = FakeLLM(canned="NO_ANSWER")
    ans = pipeline.answerer.ask("What is the CEO's salary?")
    assert ans.refused and ans.text == REFUSAL_TEXT["en"] and ans.citations == []


def test_refuses_when_model_hallucinates_a_citation(pipeline: Pipeline) -> None:
    pipeline.answerer.llm = FakeLLM(canned="The CEO earns 1M [9].")
    ans = pipeline.answerer.ask("What is the CEO's salary?")
    assert ans.refused


def test_refuses_when_a_sentence_is_uncited(pipeline: Pipeline) -> None:
    pipeline.answerer.llm = FakeLLM(canned="Refunds take 10 days [1]. Also, trust me on this.")
    ans = pipeline.answerer.ask("How fast are refunds issued?")
    assert ans.refused


def test_refuses_in_french_when_question_is_french(pipeline: Pipeline) -> None:
    pipeline.answerer.llm = FakeLLM(canned="NO_ANSWER")
    ans = pipeline.answerer.ask("Quel est le salaire du directeur général de la société ?")
    assert ans.refused and ans.text == REFUSAL_TEXT["fr"]


def test_refuses_without_calling_llm_when_index_is_empty() -> None:
    from ragkit.config import Settings
    from ragkit.store import MemoryStore

    llm = FakeLLM()
    p = Pipeline(Settings(store_url="memory://"), MemoryStore(), HashingEmbedder(), llm)
    ans = p.answerer.ask("anything")
    assert ans.refused and llm.calls == []


def test_answerer_min_score_filters_weak_hits(pipeline: Pipeline) -> None:
    a = Answerer(pipeline.retriever, FakeLLM(), k=3, min_score=10.0)
    assert a.ask("How fast are refunds issued?").refused


def test_normalise_citations_moves_trailing_markers_inside_sentence() -> None:
    from ragkit.answer import normalise_citations

    assert normalise_citations("Refunds take 10 days. [1] Returns cost money. [2][3]") == (
        "Refunds take 10 days [1]. Returns cost money [2][3]."
    )
    assert verify_citations("Refunds take 10 days. [1]", 1) == (True, [1])
