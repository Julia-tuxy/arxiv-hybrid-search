from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from models import PaperResult, SearchResponse
from search import embedder_stats, hybrid_search_timed
from serving.latency import LatencyRecorder

app = FastAPI(title="ArXiv Semantic Search", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=["Server-Timing"],
)

# Per-stage latency of recent requests (embed / ann / bm25 / fusion / total),
# so embedding cost and index-search cost are reported separately rather than
# as one end-to-end number.
_latency = LatencyRecorder()
_STAGES = ("embed_ms", "ann_ms", "bm25_ms", "fusion_ms", "rerank_ms", "format_ms", "total_ms")


@app.get("/search", response_model=SearchResponse)
def search(
    response: Response,
    q: str = Query(..., min_length=2, max_length=500, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50),
):
    # A sync endpoint: FastAPI runs it in its thread pool, so concurrent requests
    # overlap on I/O (the embedding-service call) and on FAISS/NumPy, which
    # release the GIL.
    raw, timings = hybrid_search_timed(query=q.strip(), top_k=top_k)
    _latency.record_many({k: v for k, v in timings.items() if k in _STAGES})
    _latency.incr("requests")
    _latency.incr("embed_cache_hits", int(timings.get("embed_cache_hit", 0)))
    response.headers["Server-Timing"] = ", ".join(
        f"{k.removesuffix('_ms')};dur={v:.2f}" for k, v in timings.items() if k in _STAGES)
    return SearchResponse(
        query=q,
        total_results=len(raw),
        results=[PaperResult(**r) for r in raw],
        timings={k: round(v, 3) for k, v in timings.items()},
    )


@app.get("/metrics")
def metrics():
    return {"latency": _latency.snapshot(), **embedder_stats()}


@app.get("/health")
def health():
    return {"status": "ok"}
