"""
Loads all index artifacts once at module import, exposes hybrid_search().
"""

import os
import pickle

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
MODEL_NAME = "all-MiniLM-L6-v2"
ALPHA = 0.7  # weight for vector similarity vs BM25

print("Loading search artifacts…")
_model = SentenceTransformer(MODEL_NAME)
_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))
_meta = pd.read_parquet(os.path.join(INDEX_DIR, "metadata.parquet"))
with open(os.path.join(INDEX_DIR, "bm25_corpus.pkl"), "rb") as _f:
    _corpus_tokens: list[list[str]] = pickle.load(_f)
print(f"Loaded index with {_index.ntotal} vectors, {len(_meta)} metadata rows.")


def hybrid_search(query: str, top_k: int = 10, faiss_k: int = 100) -> list[dict]:
    # 1. Encode query
    q_vec = _model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    # 2. FAISS retrieval
    actual_k = min(faiss_k, _index.ntotal)
    scores, indices = _index.search(q_vec, actual_k)
    vector_scores = scores[0]   # cosine similarities in [-1, 1]
    candidate_ids = indices[0]  # indices into metadata / corpus

    # 3. BM25 re-ranking over candidates only
    query_tokens = query.lower().split()
    candidate_corpus = [_corpus_tokens[i] for i in candidate_ids]
    bm25_local = BM25Okapi(candidate_corpus)
    bm25_raw = np.array(bm25_local.get_scores(query_tokens), dtype=np.float32)

    # 4. Normalize both scores to [0, 1]
    vector_norm = (vector_scores + 1.0) / 2.0
    bm25_max = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
    bm25_norm = bm25_raw / bm25_max

    # 5. Combine
    hybrid_scores = ALPHA * vector_norm + (1.0 - ALPHA) * bm25_norm

    # 6. Sort and return top_k
    ranked_order = np.argsort(-hybrid_scores)[:top_k]

    results = []
    for pos in ranked_order:
        idx = int(candidate_ids[pos])
        row = _meta.iloc[idx]
        results.append({
            "arxiv_id":         str(row["arxiv_id"]),
            "title":            str(row["title"]),
            "abstract_snippet": str(row["abstract"])[:300],
            "authors":          str(row["authors"]),
            "year":             str(row["year"]),
            "categories":       str(row["categories"]),
            "score":            float(hybrid_scores[pos]),
            "vector_score":     float(vector_norm[pos]),
            "bm25_score":       float(bm25_norm[pos]),
        })
    return results
