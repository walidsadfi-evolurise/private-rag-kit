"""Settings from environment variables (prefix RAG_). See .env.example."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    store_url: str = "memory://./data/index"
    embed_provider: str = "hash"  # hash | ollama | openai
    embed_model: str = "nomic-embed-text"
    embed_base_url: str | None = None
    llm_provider: str = "fake"  # fake | ollama | anthropic | openai
    llm_model: str = "llama3.1:8b"
    llm_base_url: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    top_k: int = 5
    candidates: int = 20
    chunk_chars: int = 1200
    chunk_overlap: int = 150
    api_token: str | None = None  # if set, the HTTP API requires `Authorization: Bearer <token>`
