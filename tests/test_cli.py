from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ragkit.cli import app

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
ROOT = Path(__file__).resolve().parents[1]


def test_cli_end_to_end(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RAG_STORE_URL", f"memory://{tmp_path / 'idx'}")
    monkeypatch.setenv("RAG_EMBED_PROVIDER", "hash")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "fake")
    runner = CliRunner()

    r = runner.invoke(app, ["ingest", str(CORPUS)])
    assert r.exit_code == 0 and "3 document(s)" in r.output

    r = runner.invoke(app, ["ask", "--json", "How fast are refunds issued?"])
    assert r.exit_code == 0
    payload = json.loads(r.output)
    assert not payload["refused"] and payload["citations"]

    r = runner.invoke(app, ["status"])
    assert r.exit_code == 0 and "refund-policy.md" in r.output

    r = runner.invoke(app, ["eval", str(ROOT / "evalsets" / "demo.yaml"), "--min-accuracy", "0.5"])
    assert r.exit_code == 0 and "hit_at_k" in r.output
