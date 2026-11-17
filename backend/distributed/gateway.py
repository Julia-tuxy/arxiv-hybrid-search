"""HTTP front end for the sharded cluster.

    CLUSTER_CONFIG=distributed/cluster.example.json EMBEDDING_URL=http://127.0.0.1:8001 \
        uvicorn distributed.gateway:app --port 8100

    GET  /search?q=&top_k=     same response shape as main.py, plus cluster fields
    GET  /cluster              replica health, failures, pinned version
    POST /admin/version        {"version"} -- switch only if every shard has it
    GET  /metrics              per-stage latency (embed / fanout / merge / fusion)

cluster config (distributed/cluster.example.json):
    {"root": "index/shards",
     "shards": {"0": ["http://127.0.0.1:9100", "http://127.0.0.1:9110"],
                "1": ["http://127.0.0.1:9101", "http://127.0.0.1:9111"]},
     "timeout_ms": 1000, "cooldown_s": 5}
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from distributed import sharding
from distributed.coordinator import Coordinator, HttpTransport, ShardError
from serving.latency import LatencyRecorder
from tokenizer import standard_tokens

BACKEND = Path(__file__).resolve().parents[1]
CONFIG = json.loads(Path(os.environ.get(
    "CLUSTER_CONFIG", BACKEND / "distributed" / "cluster.example.json")).read_text())
ROOT = BACKEND / CONFIG.get("root", "index/shards") if not Path(CONFIG.get("root", "")).is_absolute() \
    else Path(CONFIG["root"])
INDEX_DIR = Path(os.environ.get("INDEX_DIR", BACKEND / "index"))


class _S:
    coord: Coordinator | None = None
    transport: HttpTransport | None = None
    embedder = None
    meta: pd.DataFrame | None = None
    latency = LatencyRecorder()


S = _S()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("EMBEDDING_URL"):
        from serving.embedder import RemoteEmbedder
        S.embedder = RemoteEmbedder(os.environ["EMBEDDING_URL"])
    else:
        from serving.embedder import LocalEmbedder
        S.embedder = LocalEmbedder(os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2"))
    S.meta = pd.read_parquet(INDEX_DIR / "metadata.parquet",
                             columns=["arxiv_id", "title", "abstract", "authors", "year",
                                      "categories"])
    S.transport = HttpTransport()
    S.coord = Coordinator(
        shards={int(k): v for k, v in CONFIG["shards"].items()},
        version=os.environ.get("CLUSTER_VERSION") or sharding.current_version(ROOT),
        transport=S.transport, timeout_s=CONFIG.get("timeout_ms", 1000) / 1000,
        cooldown_s=CONFIG.get("cooldown_s", 5),
        alpha=float(os.environ.get("HYBRID_ALPHA", "0.5")),
        k_each=int(os.environ.get("CANDIDATES_PER_ARM", "100")))
    print(f"gateway: {len(CONFIG['shards'])} shards, version {S.coord.version}")
    yield
    await S.transport.aclose()


app = FastAPI(title="arXiv search gateway (sharded)", lifespan=lifespan)


@app.get("/search")
async def search(q: str = Query(..., min_length=2, max_length=500),
                 top_k: int = Query(10, ge=1, le=50), require_all: bool = False):
    t0 = time.perf_counter()
    vec, einfo = await asyncio.to_thread(S.embedder.embed, q.strip())
    tokens = standard_tokens(q)
    try:
        out = await S.coord.search(vec, tokens, top_k, require_all=require_all)
    except ShardError as e:
        S.latency.incr("errors")
        raise HTTPException(503, str(e))
    timings = {"embed_ms": einfo["embed_ms"], **out["timings"]}
    ann = [i.get("ann_ms") for i in out["shards"].values() if "ann_ms" in i]
    if ann:
        timings["ann_ms_max_shard"] = max(ann)
    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    S.latency.record_many(timings)
    if out["partial"]:
        S.latency.incr("partial_responses")
    if any(i.get("failed_over") for i in out["shards"].values()):
        S.latency.incr("failovers")

    dense = {d for d, _ in out["dense"]}
    sparse = {d for d, _ in out["sparse"]}
    results = []
    for doc, score, d, s in out["hits"]:
        row = S.meta.iloc[doc]
        results.append({
            "doc": int(doc), "arxiv_id": str(row.arxiv_id), "title": str(row.title),
            "abstract_snippet": str(row.abstract)[:300], "authors": str(row.authors),
            "year": str(row.year), "categories": str(row.categories),
            "score": float(score), "vector_score": float(d), "bm25_score": float(s),
            "retrieved_by": "both" if doc in dense and doc in sparse
            else ("dense" if doc in dense else "bm25")})
    return {"query": q, "total_results": len(results), "results": results,
            "version": out["version"], "partial": out["partial"],
            "missing_shards": out["missing_shards"], "shards": out["shards"],
            "timings": timings}


@app.get("/cluster")
def cluster():
    return S.coord.status()


class VersionReq(BaseModel):
    version: str


@app.post("/admin/version")
async def switch_version(req: VersionReq):
    return await S.coord.switch_version(req.version,
                                        lambda url: S.transport.get(f"{url}/health"))


@app.get("/metrics")
def metrics():
    return S.latency.snapshot()


@app.get("/health")
def health():
    return {"status": "ok" if S.coord else "loading"}
