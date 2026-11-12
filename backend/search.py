"""Hybrid retrieval: dense and BM25 retrieve independently, then scores are fused.

This is the configuration selected by the ablation in eval/ (see eval/README.md).
Two properties matter and are easy to lose in a refactor:

1. Both arms retrieve over the **whole corpus**. The earlier version pulled 100
   candidates from FAISS and re-fitted BM25 on just those, which estimated IDF
   from a 100-document sample and made a lexically-matching document that the
   embedding missed unreachable at any weight.
2. Each arm's scores are min-max normalised **over its own returned list**, then
   blended. Keep it that way: the measured numbers come from exactly this
   normalisation, so changing it invalidates them.

ALPHA defaults to 0.5, the sweep optimum for short keyword queries (the shape
real users type). Raise it toward 0.6 if traffic skews to long natural-language
queries; lower it toward 0.2 for title-shaped, copy-pasted queries.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import faiss

import ann_index
from bm25_index import SparseBM25
from fusion import arms_that_found, minmax_normalise, weighted_fusion
from serving.latency import StageTimer
from tokenizer import standard_tokens

INDEX_DIR = os.environ.get(
    "INDEX_DIR", os.path.join(os.path.dirname(__file__), "index")
)
MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.5"))
K_EACH = int(os.environ.get("CANDIDATES_PER_ARM", "100"))

# Stage 2 is opt-in and OFF by default. It is worth +5.0 nDCG@10 on keyword
# queries and costs 33 ms -> 601 ms p50 on a 2-vCPU CPU box (eval/README.md
# section 3). Set RERANK_TOP_K=10 to enable; 10 is the measured optimum, deeper
# reranking never helped because the first stage's Recall@10 is already 1.000.
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "0"))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Vector index type (eval/README.md section 5). "flat" is exact search; "hnsw"
# / "ivf" / "ivfpq" load index/faiss_<kind>.bin built by build_ann_index.py.
VECTOR_INDEX = os.environ.get("VECTOR_INDEX", "flat")
HNSW_EF_SEARCH = int(os.environ.get("HNSW_EF_SEARCH", "256"))
IVF_NPROBE = int(os.environ.get("IVF_NPROBE", "128"))

# Query embedding. With EMBEDDING_URL set, encoding happens in the separate
# embedding service (embedding_service.py: micro-batching + cache) and this
# process never loads torch; otherwise an in-process model with an LRU cache.
EMBEDDING_URL = os.environ.get("EMBEDDING_URL")
EMBED_CACHE_SIZE = int(os.environ.get("EMBED_CACHE_SIZE", "10000"))

# One FAISS thread per search. The API serves concurrent requests from a thread
# pool; letting every request fan out to all cores as well oversubscribes the
# CPU, and 1 thread is what eval/README.md section 5 measured.
FAISS_THREADS = int(os.environ.get("FAISS_THREADS", "1"))

print("Loading search artifacts…")
if EMBEDDING_URL:
    from serving.embedder import RemoteEmbedder
    _embedder = RemoteEmbedder(EMBEDDING_URL)
else:
    # torch and faiss in one process: see "macOS: two OpenMP runtimes" in README.md
    from serving.embedder import LocalEmbedder
    _embedder = LocalEmbedder(MODEL_NAME, cache_size=EMBED_CACHE_SIZE)
faiss.omp_set_num_threads(FAISS_THREADS)
_index = faiss.read_index(os.path.join(
    INDEX_DIR, "faiss_index.bin" if VECTOR_INDEX == "flat" else f"faiss_{VECTOR_INDEX}.bin"))
ann_index.set_search_params(_index, nprobe=IVF_NPROBE, ef_search=HNSW_EF_SEARCH)
_meta = pd.read_parquet(os.path.join(INDEX_DIR, "metadata.parquet"))
_bm25 = SparseBM25.load(os.path.join(INDEX_DIR, "bm25_standard"))
_reranker = None
_rerank_texts: list[str] | None = None
if RERANK_TOP_K > 0:
    from sentence_transformers import CrossEncoder

    _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    _rerank_texts = (_meta.title + ". " + _meta.abstract).tolist()

print(
    f"Loaded {_index.ntotal:,} vectors ({VECTOR_INDEX}), {len(_meta):,} metadata rows, "
    f"embedder {_embedder.name}, "
    f"BM25 vocab {len(_bm25.vocab):,} (alpha={ALPHA}, k_each={K_EACH}, "
    f"rerank_top_k={RERANK_TOP_K})"
)


def _dense_arm(vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    scores, ids = _index.search(vec.reshape(1, -1), min(k, _index.ntotal))
    return [(i, s) for i, s in zip(ids[0].tolist(), scores[0].tolist()) if i >= 0]


def _sparse_arm(query: str, k: int) -> list[tuple[int, float]]:
    ids, scores = _bm25.search(standard_tokens(query), k)
    # Zero-scoring documents are kept on purpose: dropping them would shift the
    # min-max floor and change every normalised score, so the served ranking
    # would no longer be the one the ablation measured.
    return [(int(i), float(s)) for i, s in zip(ids, scores)]


def hybrid_search(query: str, top_k: int = 10, alpha: float | None = None,
                  k_each: int | None = None, rerank_top_k: int | None = None) -> list[dict]:
    return hybrid_search_timed(query, top_k, alpha, k_each, rerank_top_k)[0]


def hybrid_search_timed(
    query: str,
    top_k: int = 10,
    alpha: float | None = None,
    k_each: int | None = None,
    rerank_top_k: int | None = None,
) -> tuple[list[dict], dict]:
    """Search and report where the time went, per stage, in milliseconds.

    embed_ms   query encoding (in-process, or a round trip to the embedding service)
    ann_ms     vector index search only
    bm25_ms    sparse arm (tokenise + score)
    fusion_ms  normalisation + blending
    rerank_ms  cross-encoder, when enabled
    format_ms  metadata lookup and response rows
    total_ms   all of the above
    """
    alpha = ALPHA if alpha is None else alpha
    k_each = K_EACH if k_each is None else k_each
    t = StageTimer()
    t0 = time.perf_counter()

    with t.stage("embed_ms"):
        vec, einfo = _embedder.embed(query)
    with t.stage("ann_ms"):
        dense_hits = _dense_arm(vec, k_each)
    with t.stage("bm25_ms"):
        sparse_hits = _sparse_arm(query, k_each)
    with t.stage("fusion_ms"):
        dense = minmax_normalise(dense_hits)
        sparse = minmax_normalise(sparse_hits)
        fused = weighted_fusion(dense_hits, sparse_hits, alpha)

    rerank_k = RERANK_TOP_K if rerank_top_k is None else rerank_top_k
    if rerank_k > 0 and _reranker is not None:
        with t.stage("rerank_ms"):
            head = fused[:rerank_k]
            ce = _reranker.predict([(query, _rerank_texts[doc]) for doc, *_ in head],
                                   show_progress_bar=False)
            # keep the fused score for display; reorder by the cross-encoder, and
            # leave the tail in first-stage order
            head = [head[i] for i in np.argsort(-np.asarray(ce))]
            fused = head + fused[rerank_k:]

    with t.stage("format_ms"):
        results = []
        for doc, score, d, s in fused[:top_k]:
            row = _meta.iloc[doc]
            results.append(
                {
                    "arxiv_id": str(row["arxiv_id"]),
                    "title": str(row["title"]),
                    "abstract_snippet": str(row["abstract"])[:300],
                    "authors": str(row["authors"]),
                    "year": str(row["year"]),
                    "categories": str(row["categories"]),
                    "score": float(score),
                    "vector_score": float(d),
                    "bm25_score": float(s),
                    "retrieved_by": arms_that_found(doc, dense, sparse),
                }
            )
    t.timings["total_ms"] = (time.perf_counter() - t0) * 1000
    t.timings["embed_cache_hit"] = float(einfo.get("cache_hit", False))
    for k in ("embed_server_ms", "embed_queue_ms", "embed_batch_size"):
        if einfo.get(k) is not None:
            t.timings[k] = float(einfo[k])
    return results, t.timings


def embedder_stats() -> dict:
    cache = getattr(_embedder, "cache", None)
    return {"embedder": _embedder.name, "cache": cache.stats() if cache else None}
