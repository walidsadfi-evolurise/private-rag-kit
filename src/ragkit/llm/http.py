from __future__ import annotations

import httpx

from ragkit.llm.base import LLM


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, model: str, api_key: str, base_url: str, timeout: float = 120.0) -> None:
        self.model = model
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )

    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
        r = self.client.post(
            "/v1/messages",
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text").strip()


class OpenAILLM(LLM):
    name = "openai"

    def __init__(self, model: str, api_key: str, base_url: str, timeout: float = 120.0) -> None:
        self.model = model
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, headers={"authorization": f"Bearer {api_key}"}
        )

    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
        r = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"]).strip()


class OllamaLLM(LLM):
    name = "ollama"

    def __init__(self, model: str, base_url: str, timeout: float = 300.0) -> None:
        self.model = model
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
        r = self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        return str(r.json()["message"]["content"]).strip()
