from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ragkit import api
from ragkit.evaluate import evaluate, load_cases
from ragkit.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_eval_set_passes_with_fake_llm(pipeline: Pipeline) -> None:
    cases = load_cases(ROOT / "evalsets" / "demo.yaml")
    report = evaluate(pipeline.answerer, cases)
    s = report.summary()
    assert s["cases"] == 6
    assert s["hit_at_k"] == 1.0
    assert s["refusal_recall"] == 1.0
    assert s["answer_accuracy"] >= 0.8
    assert s["citation_rate"] == 1.0


def test_api_ask_ingest_documents_and_token(pipeline: Pipeline, tmp_path: Path) -> None:
    pipeline.settings.api_token = "secret"
    api.get_pipeline.cache_clear()
    api.get_pipeline = lambda: pipeline  # type: ignore[assignment]
    client = TestClient(api.app)

    assert client.get("/healthz").json()["status"] == "ok"
    assert client.post("/ask", json={"question": "x"}).status_code == 401

    headers = {"authorization": "Bearer secret"}
    r = client.post("/ask", json={"question": "How fast are refunds issued?"}, headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert not body["refused"] and body["citations"][0]["source"] == "refund-policy.md"

    assert client.post("/ask", json={"question": ""}, headers=headers).status_code == 422

    note = tmp_path / "note.md"
    note.write_text("# Parking\n\nVisitor parking is on level -2 and costs 5 euros per day.")
    with note.open("rb") as fh:
        r = client.post("/ingest", files=[("files", ("note.md", fh, "text/markdown"))], headers=headers)
    assert r.status_code == 200 and r.json()["documents"] == 1
    assert any(d["source"] == "note.md" for d in client.get("/documents", headers=headers).json())

    with note.open("rb") as fh:
        r = client.post(
            "/ingest", files=[("files", ("evil.exe", fh, "application/octet-stream"))], headers=headers
        )
    assert r.status_code == 415
