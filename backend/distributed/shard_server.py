"""One shard server: dense + BM25 search over this shard's documents.

    SHARD_ROOT=index/shards SHARD_ID=0 REPLICA=0a uvicorn distributed.shard_server:app --port 9100

    POST /search          {"vector_b64", "tokens", "k", "version"} -> per-arm top-k
    GET  /health          shard id, loaded versions, doc counts
    POST /admin/load      {"version"}  load an additional version (blue/green)
    POST /admin/unload    {"version"}
    POST /admin/fault     {"mode": "ok" | "error" | "slow", "delay_ms"}  -- test hook

A server can hold several index versions at once and answers each request from
the version the request names. That is what makes a version switch safe: every
replica loads v2 next to v1, the coordinator checks they all have it, then flips
its pin; no request is ever answered from a mix of versions. A request naming a
version this replica does not hold gets 409, which the coordinator treats like a
dead replica and fails over.

The shard never sees text: the coordinator embeds and tokenises once and ships
the vector and the tokens, so N shards do not mean N model forward passes.
"""
from __future__ import annotations

import asyncio
import base64
import os
import pickle
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy import sparse

import ann_index
from bm25_index import SparseBM25
from distributed import sharding

ROOT = Path(os.environ.get("SHARD_ROOT", Path(__file__).resolve().parents[1] / "index" / "shards"))
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
REPLICA = os.environ.get("REPLICA", f"{SHARD_ID}")
VERIFY = os.environ.get("SHARD_VERIFY_CHECKSUMS", "1") != "0"
HNSW_EF_SEARCH = int(os.environ.get("HNSW_EF_SEARCH", "256"))
IVF_NPROBE = int(os.environ.get("IVF_NPROBE", "128"))
# one FAISS thread per search: several shard processes share the host, and each
# request is a single query (same reasoning as search.py)
faiss.omp_set_num_threads(int(os.environ.get("FAISS_THREADS", "1")))


class ShardIndex:
    def __init__(self, root: Path, version: str, shard: int):
        if VERIFY:
            sharding.verify(root, version, shard)
        manifest = sharding.read_manifest(root, version)
        sdir = root / version / f"shard-{shard}"
        self.version = version
        self.manifest = manifest
        self.vectors = ann_index.load(sdir / "vectors.faiss")
        ann_index.set_search_params(self.vectors, nprobe=IVF_NPROBE, ef_search=HNSW_EF_SEARCH)
        self.doc_ids = np.load(sdir / "doc_ids.npy")
        with open(root / version / "bm25.vocab.pkl", "rb") as f:
            meta = pickle.load(f)
        W = sparse.load_npz(sdir / "bm25.npz").tocsc()
        self.bm25 = SparseBM25(W, meta["vocab"], k1=meta["k1"], b=meta["b"])

    def search(self, vec: np.ndarray, tokens: list[str], k: int) -> dict:
        t0 = time.perf_counter()
        scores, ids = self.vectors.search(vec.reshape(1, -1), min(k, self.vectors.ntotal))
        t1 = time.perf_counter()
        local, bscores = self.bm25.search(tokens, k)       # local row positions
        t2 = time.perf_counter()
        return {
            "dense": [[int(i), float(s)] for i, s in zip(ids[0], scores[0]) if i >= 0],
            "bm25": [[int(self.doc_ids[i]), float(s)] for i, s in zip(local, bscores)],
            "timings": {"ann_ms": (t1 - t0) * 1000, "bm25_ms": (t2 - t1) * 1000},
        }


class _State:
    versions: dict[str, ShardIndex] = {}
    lock = threading.Lock()
    fault = {"mode": "ok", "delay_ms": 0}


S = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    v = os.environ.get("SHARD_VERSION") or sharding.current_version(ROOT)
    S.versions[v] = ShardIndex(ROOT, v, SHARD_ID)
    print(f"shard {SHARD_ID} replica {REPLICA}: version {v}, "
          f"{S.versions[v].vectors.ntotal:,} docs")
    yield


app = FastAPI(title=f"shard {SHARD_ID}", lifespan=lifespan)


class SearchReq(BaseModel):
    vector_b64: str
    tokens: list[str]
    k: int = 100
    version: str


class VersionReq(BaseModel):
    version: str


class FaultReq(BaseModel):
    mode: str = "ok"
    delay_ms: float = 0


@app.post("/search")
async def search(req: SearchReq):
    if S.fault["mode"] == "error":
        raise HTTPException(503, "injected fault")
    if S.fault["mode"] == "slow":
        await asyncio.sleep(S.fault["delay_ms"] / 1000)
    idx = S.versions.get(req.version)
    if idx is None:
        raise HTTPException(409, f"version {req.version} not loaded on shard {SHARD_ID} "
                                 f"replica {REPLICA}; have {sorted(S.versions)}")
    vec = np.frombuffer(base64.b64decode(req.vector_b64), dtype="<f4").astype(np.float32)
    out = await asyncio.to_thread(idx.search, vec, req.tokens, req.k)
    out.update(shard=SHARD_ID, replica=REPLICA, version=idx.version)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "shard": SHARD_ID, "replica": REPLICA,
            "versions": {v: int(i.vectors.ntotal) for v, i in S.versions.items()},
            "fault": S.fault["mode"]}


@app.post("/admin/load")
async def load(req: VersionReq):
    if req.version in S.versions:
        return {"loaded": sorted(S.versions)}
    idx = await asyncio.to_thread(ShardIndex, ROOT, req.version, SHARD_ID)
    with S.lock:
        S.versions = {**S.versions, req.version: idx}      # copy-on-write swap
    return {"loaded": sorted(S.versions)}


@app.post("/admin/unload")
def unload(req: VersionReq):
    with S.lock:
        S.versions = {v: i for v, i in S.versions.items() if v != req.version}
    return {"loaded": sorted(S.versions)}


@app.post("/admin/fault")
def fault(req: FaultReq):
    S.fault = {"mode": req.mode, "delay_ms": req.delay_ms}
    return S.fault
