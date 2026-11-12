"""Thread-safe LRU cache for query embeddings.

Query traffic is heavy-tailed -- the same handful of queries recur -- and an
embedding is a pure function of (model, text), so a cached vector is exactly the
vector the model would produce. The key collapses whitespace only: that never
changes the tokenisation, whereas lowercasing or stripping punctuation would be a
model-specific assumption.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np


def cache_key(text: str) -> str:
    return " ".join(text.split())


class EmbeddingCache:
    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.capacity > 0

    def get(self, text: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        key = cache_key(text)
        with self._lock:
            v = self._d.get(key)
            if v is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return v

    def put(self, text: str, vec: np.ndarray) -> None:
        if not self.enabled:
            return
        key = cache_key(text)
        vec = np.asarray(vec, dtype=np.float32)
        vec.setflags(write=False)              # shared between requests: never mutate
        with self._lock:
            self._d[key] = vec
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {"capacity": self.capacity, "size": len(self._d), "hits": self.hits,
                    "misses": self.misses,
                    "hit_rate": round(self.hits / total, 4) if total else 0.0}

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.hits = self.misses = 0
