"""Standalone embedding inference service.

    uvicorn embedding_service:app --port 8001

    POST /embed   {"texts": ["..."], "encoding": "float" | "base64"}
    GET  /metrics batching, cache and per-stage latency statistics
    GET  /health

Why a separate process: query encoding is ~90 % of the search API's own CPU time
(eval/README.md section 2.4), it is the only GPU-shaped workload in the system,
and it scales on a different axis from index search. Splitting it out lets the
two be sized, batched and measured independently -- the search API's /metrics
reports `embed_ms` and `ann_ms` as separate stages for exactly that reason.

Each text in a request goes through: cache lookup -> micro-batcher -> model.
Concurrent requests from different clients are batched together.

Configuration (environment):
    MODEL_NAME          all-MiniLM-L6-v2
    EMBED_BATCHING      1      0 = no micro-batching; each text is encoded alone
    EMBED_MAX_BATCH     32
    EMBED_MAX_WAIT_MS   0      0 = greedy batching (measured best); >0 adds a fill window
    EMBED_CACHE_SIZE    10000  0 disables the cache
    TORCH_THREADS       (torch default)
"""
from __future__ import annotations

import asyncio
import base64
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving.batcher import MicroBatcher
from serving.cache import EmbeddingCache
from serving.latency import LatencyRecorder

MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
BATCHING = os.environ.get("EMBED_BATCHING", "1") != "0"
MAX_BATCH = int(os.environ.get("EMBED_MAX_BATCH", "32"))
# 0 = greedy: no timer, a batch is whatever queued while the previous batch ran.
# Measured better than a 5 ms window at every concurrency (eval/README.md 6).
MAX_WAIT_MS = float(os.environ.get("EMBED_MAX_WAIT_MS", "0"))
CACHE_SIZE = int(os.environ.get("EMBED_CACHE_SIZE", "10000"))
MAX_TEXTS_PER_REQUEST = 256


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_TEXTS_PER_REQUEST)
    encoding: Literal["float", "base64"] = "float"


class EmbedResponse(BaseModel):
    model: str
    dim: int
    vectors: list[list[float]] | None = None
    vectors_b64: str | None = None       # little-endian float32, row-major (n, dim)
    cached: list[bool]
    timings: dict[str, float]


class _State:
    model = None
    dim = 0
    batcher: MicroBatcher | None = None
    cache = EmbeddingCache(CACHE_SIZE)
    latency = LatencyRecorder()
    unbatched_sem: asyncio.Semaphore | None = None


S = _State()


def _encode(texts: list[str]) -> np.ndarray:
    return S.model.encode(texts, batch_size=max(len(texts), 1), normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from sentence_transformers import SentenceTransformer
    import torch

    if os.environ.get("TORCH_THREADS"):
        torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
    S.model = SentenceTransformer(MODEL_NAME, device="cpu")
    S.dim = S.model.get_sentence_embedding_dimension()
    _encode(["warm-up"])
    if BATCHING:
        S.batcher = MicroBatcher(_encode, max_batch=MAX_BATCH, max_wait_ms=MAX_WAIT_MS)
        S.batcher.start()
    else:
        # without batching, concurrent requests run concurrent model calls; bound
        # them by core count so the comparison is against a sane baseline
        S.unbatched_sem = asyncio.Semaphore(os.cpu_count() or 4)
    print(f"embedding service: {MODEL_NAME} dim={S.dim} batching={BATCHING} "
          f"max_batch={MAX_BATCH} max_wait_ms={MAX_WAIT_MS} cache={CACHE_SIZE}")
    yield
    if S.batcher:
        await S.batcher.stop()


app = FastAPI(title="Embedding service", version="1.0.0", lifespan=lifespan)


async def _embed_one(text: str) -> tuple[np.ndarray, bool, dict]:
    hit = S.cache.get(text)
    if hit is not None:
        return hit, True, {}
    if S.batcher is not None:
        vec, info = await S.batcher.submit(text)
    else:
        async with S.unbatched_sem:
            t0 = time.perf_counter()
            vec = (await asyncio.to_thread(_encode, [text]))[0]
            info = {"queue_ms": 0.0, "inference_ms": (time.perf_counter() - t0) * 1000,
                    "batch_size": 1}
    S.cache.put(text, vec)
    return vec, False, info


@app.post("/embed", response_model=EmbedResponse, response_model_exclude_none=True)
async def embed(req: EmbedRequest):
    if S.model is None:
        raise HTTPException(503, "model not loaded")
    t0 = time.perf_counter()
    out = await asyncio.gather(*(_embed_one(t) for t in req.texts))
    vecs = np.stack([v for v, _, _ in out]).astype(np.float32)
    cached = [c for _, c, _ in out]
    infos = [i for _, c, i in out if not c]
    timings = {"total_ms": (time.perf_counter() - t0) * 1000}
    if infos:
        timings["queue_ms"] = max(i["queue_ms"] for i in infos)
        timings["inference_ms"] = max(i["inference_ms"] for i in infos)
        timings["batch_size"] = float(max(i["batch_size"] for i in infos))
    S.latency.record("request", timings["total_ms"])
    S.latency.incr("texts", len(req.texts))
    S.latency.incr("cache_hits", sum(cached))
    for i in infos:
        S.latency.record("queue_wait", i["queue_ms"])
        S.latency.record("inference", i["inference_ms"])

    body = {"model": MODEL_NAME, "dim": S.dim, "cached": cached, "timings": timings}
    if req.encoding == "base64":
        body["vectors_b64"] = base64.b64encode(vecs.astype("<f4").tobytes()).decode()
    else:
        body["vectors"] = vecs.tolist()
    return body


@app.get("/metrics")
def metrics():
    return {"model": MODEL_NAME, "batching": BATCHING,
            "batcher": S.batcher.stats() if S.batcher else None,
            "cache": S.cache.stats(), "latency": S.latency.snapshot()}


@app.post("/metrics/reset")
def metrics_reset(clear_cache: bool = False):
    """Load tests reset counters between runs; this is not an operational endpoint."""
    S.latency.reset()
    if S.batcher:
        S.batcher.reset_stats()
    if clear_cache:
        S.cache.clear()
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok" if S.model is not None else "loading", "model": MODEL_NAME}
