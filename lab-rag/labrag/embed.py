"""Embedding providers. All return unit-length float32 vectors.

The default is `fastembed` (a small ONNX model that runs on any laptop, no
account and no GPU). Ollama and OpenAI-compatible servers are supported for
labs that already run them. `HashEmbedder` needs nothing at all and is used by
the tests and as a last-resort fallback: it is a bag-of-words hash, so it only
captures word overlap, but keyword search still works and the tool stays
usable while the real model downloads.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from itertools import pairwise
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashEmbedder:
    """Deterministic bag-of-words (plus bigrams) hashing. No downloads, no network."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        words = [w.lower() for w in _WORD_RE.findall(text)]
        tokens = words + [f"{a} {b}" for a, b in pairwise(words)]
        for tok in tokens:
            h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            v[idx] += sign
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class FastEmbedEmbedder:
    """Local ONNX embeddings via the `fastembed` package (default BAAI/bge-small-en-v1.5)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None, batch_size: int = 32):
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model}"
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dim = len(probe)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.stack(list(self._model.embed(texts, batch_size=self.batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(np.stack(list(self._model.query_embed([text]))))[0]


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", url: str = "http://localhost:11434", batch_size: int = 32, timeout: float = 120.0):
        import httpx

        self.name = f"ollama:{model}"
        self.model = model
        self.url = url.rstrip("/")
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=timeout)
        self.dim = len(self._raw(["dimension probe"])[0])

    def _raw(self, texts: list[str]) -> list[list[float]]:
        r = self._client.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]

    def _prefixed(self, texts: Iterable[str], kind: str) -> list[str]:
        if self.model.startswith("nomic"):
            return [f"search_{kind}: {t}" for t in texts]
        return list(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[list[float]] = []
        texts = self._prefixed(texts, "document")
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._raw(texts[i : i + self.batch_size]))
        return normalize(np.array(out, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(np.array(self._raw(self._prefixed([text], "query")), dtype=np.float32))[0]


class OpenAIEmbedder:
    """OpenAI or any /v1/embeddings-compatible server."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = "", base_url: str = "https://api.openai.com/v1", batch_size: int = 64, timeout: float = 120.0):
        import httpx

        self.name = f"openai:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)
        self.dim = len(self._raw(["dimension probe"])[0])

    def _raw(self, texts: list[str]) -> list[list[float]]:
        r = self._client.post(f"{self.base_url}/embeddings", json={"model": self.model, "input": texts})
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._raw(texts[i : i + self.batch_size]))
        return normalize(np.array(out, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(np.array(self._raw([text]), dtype=np.float32))[0]
