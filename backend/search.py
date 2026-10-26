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

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from bm25_index import SparseBM25
from fusion import arms_that_found, minmax_normalise, weighted_fusion
from tokenizer import standard_tokens

INDEX_DIR = os.environ.get(
    "INDEX_DIR", os.path.join(os.path.dirname(__file__), "index")
)
MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.5"))
K_EACH = int(os.environ.get("CANDIDATES_PER_ARM", "100"))

print("Loading search artifacts…")
_model = SentenceTransformer(MODEL_NAME)
_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))
_meta = pd.read_parquet(os.path.join(INDEX_DIR, "metadata.parquet"))
_bm25 = SparseBM25.load(os.path.join(INDEX_DIR, "bm25_standard"))
print(
    f"Loaded {_index.ntotal:,} vectors, {len(_meta):,} metadata rows, "
    f"BM25 vocab {len(_bm25.vocab):,} (alpha={ALPHA}, k_each={K_EACH})"
)


def _dense_arm(query: str, k: int) -> list[tuple[int, float]]:
    vec = _model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, ids = _index.search(vec, min(k, _index.ntotal))
    return list(zip(ids[0].tolist(), scores[0].tolist()))


def _sparse_arm(query: str, k: int) -> list[tuple[int, float]]:
    ids, scores = _bm25.search(standard_tokens(query), k)
    # Zero-scoring documents are kept on purpose: dropping them would shift the
    # min-max floor and change every normalised score, so the served ranking
    # would no longer be the one the ablation measured.
    return [(int(i), float(s)) for i, s in zip(ids, scores)]


def hybrid_search(
    query: str,
    top_k: int = 10,
    alpha: float | None = None,
    k_each: int | None = None,
) -> list[dict]:
    alpha = ALPHA if alpha is None else alpha
    k_each = K_EACH if k_each is None else k_each

    dense_hits = _dense_arm(query, k_each)
    sparse_hits = _sparse_arm(query, k_each)
    dense = minmax_normalise(dense_hits)
    sparse = minmax_normalise(sparse_hits)
    fused = weighted_fusion(dense_hits, sparse_hits, alpha)

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
    return results
