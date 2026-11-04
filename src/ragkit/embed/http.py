"""Embedders that call an HTTP API: Ollama (local) and OpenAI-compatible."""

from __future__ import annotations

import httpx
import numpy as np

from ragkit.embed.base import Embedder


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(self, model: str, base_url: str, timeout: float = 120.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": texts})
        r.raise_for_status()
        vectors = np.asarray(r.json()["embeddings"], dtype=np.float32)
        self.dim = vectors.shape[1]
        return self.normalise(vectors)


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(
        self, model: str, api_key: str, base_url: str, timeout: float = 120.0, batch: int = 64
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch = batch
        self.client = httpx.Client(timeout=timeout, headers={"authorization": f"Bearer {api_key}"})
        self.dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            r = self.client.post(
                f"{self.base_url}/embeddings", json={"model": self.model, "input": texts[i : i + self.batch]}
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            rows.extend(d["embedding"] for d in data)
        vectors = np.asarray(rows, dtype=np.float32)
        self.dim = vectors.shape[1]
        return self.normalise(vectors)
