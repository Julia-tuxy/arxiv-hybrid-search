"""Query embedders used by the search API.

    LocalEmbedder   in-process SentenceTransformer + LRU cache (the default)
    RemoteEmbedder  HTTP client for embedding_service.py (set EMBEDDING_URL)

Both return the same float32, L2-normalised vector for the same text -- the
remote path ships raw float32 bytes (base64), not decimal JSON, so the vectors
are bit-identical and the measured ranking does not move.
"""
from __future__ import annotations

import base64
import threading
import time

import numpy as np

from .cache import EmbeddingCache


class LocalEmbedder:
    def __init__(self, model_name: str, cache_size: int = 10_000):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.cache = EmbeddingCache(cache_size)
        self.name = f"local:{model_name}"

    def embed(self, text: str) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        v = self.cache.get(text)
        hit = v is not None
        if not hit:
            v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False).astype(np.float32)[0]
            self.cache.put(text, v)
        return v, {"embed_ms": (time.perf_counter() - t0) * 1000, "cache_hit": hit}


class RemoteEmbedder:
    def __init__(self, url: str, timeout_s: float = 5.0):
        import httpx

        self.url = url.rstrip("/")
        self._local = threading.local()        # one connection pool per thread
        self._httpx = httpx
        self.timeout = timeout_s
        self.name = f"remote:{self.url}"

    def _client(self):
        c = getattr(self._local, "client", None)
        if c is None:
            c = self._local.client = self._httpx.Client(timeout=self.timeout)
        return c

    def embed(self, text: str) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        r = self._client().post(f"{self.url}/embed", json={"texts": [text], "encoding": "base64"})
        r.raise_for_status()
        body = r.json()
        v = np.frombuffer(base64.b64decode(body["vectors_b64"]), dtype="<f4")
        v = v.reshape(-1, body["dim"])[0].astype(np.float32)
        t = body.get("timings", {})
        return v, {"embed_ms": (time.perf_counter() - t0) * 1000,
                   "cache_hit": body["cached"][0],
                   "embed_server_ms": t.get("total_ms"),
                   "embed_queue_ms": t.get("queue_ms"),
                   "embed_batch_size": t.get("batch_size")}
