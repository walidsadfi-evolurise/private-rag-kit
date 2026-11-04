"""Chat-completion backends behind one tiny interface."""

from __future__ import annotations

from ragkit.llm.base import LLM, FakeLLM
from ragkit.llm.http import AnthropicLLM, OllamaLLM, OpenAILLM


def make_llm(provider: str, model: str, api_key: str | None = None, base_url: str | None = None) -> LLM:
    if provider == "fake":
        return FakeLLM()
    if provider == "ollama":
        return OllamaLLM(model=model, base_url=base_url or "http://localhost:11434")
    if provider == "anthropic":
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")
        return AnthropicLLM(model=model, api_key=api_key, base_url=base_url or "https://api.anthropic.com")
    if provider == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai provider")
        return OpenAILLM(model=model, api_key=api_key, base_url=base_url or "https://api.openai.com/v1")
    raise ValueError(f"unknown llm provider: {provider}")


__all__ = ["LLM", "FakeLLM", "AnthropicLLM", "OpenAILLM", "OllamaLLM", "make_llm"]
