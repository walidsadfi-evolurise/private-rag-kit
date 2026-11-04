from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.config import Settings
from ragkit.embed import HashingEmbedder
from ragkit.llm import FakeLLM
from ragkit.pipeline import Pipeline
from ragkit.store import MemoryStore

CORPUS = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture
def pipeline() -> Pipeline:
    settings = Settings(store_url="memory://", embed_provider="hash", llm_provider="fake", top_k=3)
    p = Pipeline(settings, MemoryStore(), HashingEmbedder(), FakeLLM())
    p.ingest(CORPUS)
    return p
