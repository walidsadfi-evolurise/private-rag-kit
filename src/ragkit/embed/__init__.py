"""Embedding backends. All return L2-normalised float vectors of a fixed dimension."""

from __future__ import annotations

from ragkit.embed.base import Embedder
from ragkit.embed.hashing import HashingEmbedder
from ragkit.embed.http import OllamaEmbedder, OpenAIEmbedder


def make_embedder(
    provider: str, model: str, base_url: str | None = None, api_key: str | None = None
) -> Embedder:
    if provider == "hash":
        return HashingEmbedder()
    if provider == "ollama":
        return OllamaEmbedder(model=model, base_url=base_url or "http://localhost:11434")
    if provider == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai embedder")
        return OpenAIEmbedder(model=model, api_key=api_key, base_url=base_url or "https://api.openai.com/v1")
    raise ValueError(f"unknown embedder provider: {provider}")


__all__ = ["Embedder", "HashingEmbedder", "OllamaEmbedder", "OpenAIEmbedder", "make_embedder"]
