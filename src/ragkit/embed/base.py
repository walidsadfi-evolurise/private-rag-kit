from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), rows L2-normalised."""

    @staticmethod
    def normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.asarray(matrix / norms, dtype=np.float32)
