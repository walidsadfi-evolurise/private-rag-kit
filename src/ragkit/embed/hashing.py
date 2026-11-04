"""A deterministic, dependency-free embedder for tests, CI and smoke runs.

It is a hashed bag of word unigrams and bigrams: semantically weak, but it
exercises the whole pipeline (vector store, hybrid fusion, citations) without
a model or a network. Never use it for real answers.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from ragkit.embed.base import Embedder

_TOKEN = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)


class HashingEmbedder(Embedder):
    name = "hash"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        toks = [t.lower() for t in _TOKEN.findall(text)]
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text):
                h = hashlib.blake2b(feat.encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                out[i, idx] += sign
        return self.normalise(out)
