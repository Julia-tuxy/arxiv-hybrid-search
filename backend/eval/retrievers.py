"""Retrieval strategies compared in the ablation.

Every retriever exposes `search(query: str, top_k: int) -> list[(doc_pos, score)]`
where doc_pos is the row index into metadata.parquet.
"""
from __future__ import annotations

import numpy as np

import _path  # noqa: F401  -- puts backend/ on sys.path
from bm25_index import SparseBM25
from fusion import weighted_fusion
from tokenizer import simple_tokens, standard_tokens


# --------------------------------------------------------------------------
# single-signal retrievers
# --------------------------------------------------------------------------
class _Cached:
    """Memoises (query, k) -> run so an alpha sweep does not re-search per setting.
    Disabled for latency measurement."""

    def __init__(self):
        self.cache_enabled = True
        self._runs: dict[tuple[str, int], list] = {}

    def _memo(self, query, top_k, fn):
        if not self.cache_enabled:
            return fn()
        key = (query, top_k)
        hit = self._runs.get(key)
        if hit is None:
            hit = fn()
            self._runs[key] = hit
        return hit


class DenseRetriever(_Cached):
    name = "dense"

    def __init__(self, faiss_index, encoder):
        super().__init__()
        self.index = faiss_index
        self.encoder = encoder
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, query: str) -> np.ndarray:
        if not self.cache_enabled:
            return self.encoder.encode([query])
        v = self._cache.get(query)
        if v is None:
            v = self.encoder.encode([query])
            self._cache[query] = v
        return v

    def _search(self, query: str, top_k: int):
        scores, idx = self.index.search(self.embed(query), min(top_k, self.index.ntotal))
        return list(zip(idx[0].tolist(), scores[0].tolist()))

    def search(self, query: str, top_k: int):
        return self._memo(query, top_k, lambda: self._search(query, top_k))


class BM25Retriever(_Cached):
    def __init__(self, index: SparseBM25, tokenize=standard_tokens, name="bm25"):
        super().__init__()
        self.index = index
        self.tokenize = tokenize
        self.name = name

    def _search(self, query: str, top_k: int):
        ids, scores = self.index.search(self.tokenize(query), top_k)
        return list(zip(ids.tolist(), scores.tolist()))

    def search(self, query: str, top_k: int):
        return self._memo(query, top_k, lambda: self._search(query, top_k))


# --------------------------------------------------------------------------
# the shipped baseline: dense retrieval, BM25 only re-scores the candidates
# --------------------------------------------------------------------------
class LegacyHybrid:
    """Faithful reproduction of backend/search.py as originally written.

    BM25Okapi is re-fitted on the FAISS candidate set for every query, so IDF is
    estimated from ~100 documents and BM25 can never surface a document the
    dense stage missed.
    """
    name = "hybrid_legacy"

    def __init__(self, dense: DenseRetriever, corpus_tokens: list[list[str]],
                 alpha: float = 0.7, faiss_k: int = 100):
        from rank_bm25 import BM25Okapi
        self._BM25Okapi = BM25Okapi
        self.dense = dense
        self.corpus_tokens = corpus_tokens
        self.alpha = alpha
        self.faiss_k = faiss_k

    def search(self, query: str, top_k: int):
        scores, idx = self.dense.index.search(self.dense.embed(query), self.faiss_k)
        vec, cand = scores[0], idx[0]
        bm25 = self._BM25Okapi([self.corpus_tokens[i] for i in cand])
        raw = np.asarray(bm25.get_scores(simple_tokens(query)), dtype=np.float32)
        vec_norm = (vec + 1.0) / 2.0
        raw_norm = raw / (raw.max() if raw.max() > 0 else 1.0)
        fused = self.alpha * vec_norm + (1.0 - self.alpha) * raw_norm
        order = np.argsort(-fused)[:top_k]
        return [(int(cand[p]), float(fused[p])) for p in order]


# --------------------------------------------------------------------------
# proper fusion: both arms retrieve independently over the full corpus
# --------------------------------------------------------------------------
class WeightedFusion:
    """alpha * norm(dense) + (1 - alpha) * norm(bm25) over the union of both lists.

    A document retrieved by only one arm scores 0 on the other, which is what
    makes a lexical-only match reachable at all.
    """

    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever,
                 alpha: float = 0.5, k_each: int = 100):
        self.dense, self.sparse, self.alpha, self.k_each = dense, sparse, alpha, k_each
        self.name = f"fusion_weighted_a{alpha:g}"

    def search(self, query: str, top_k: int):
        fused = weighted_fusion(self.dense.search(query, self.k_each),
                                self.sparse.search(query, self.k_each),
                                self.alpha)
        return [(doc, score) for doc, score, _, _ in fused[:top_k]]


class RRFusion:
    """Reciprocal Rank Fusion: score = sum_arms 1 / (k + rank).

    Rank-based, so it needs no score normalisation and is stable across queries
    whose score distributions differ wildly.
    """

    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever,
                 k_rrf: int = 60, k_each: int = 100):
        self.dense, self.sparse, self.k_rrf, self.k_each = dense, sparse, k_rrf, k_each
        self.name = f"fusion_rrf_k{k_rrf}"

    def search(self, query: str, top_k: int):
        fused: dict[int, float] = {}
        for arm in (self.dense, self.sparse):
            for rank, (doc, _) in enumerate(arm.search(query, self.k_each), start=1):
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (self.k_rrf + rank)
        return sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]


# --------------------------------------------------------------------------
# stage 2: cross-encoder reranking
# --------------------------------------------------------------------------
class CrossEncoderReranker:
    """Rerank the first-stage candidates with a query-document cross-encoder."""

    def __init__(self, base, texts: list[str], model_name: str, rerank_k: int = 50,
                 batch_size: int = 32, device: str = "cpu"):
        from sentence_transformers import CrossEncoder
        self.base, self.texts, self.rerank_k, self.batch_size = base, texts, rerank_k, batch_size
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.name = f"{base.name}+ce{rerank_k}"

    def search(self, query: str, top_k: int):
        cands = [doc for doc, _ in self.base.search(query, self.rerank_k)]
        if not cands:
            return []
        scores = self.model.predict([(query, self.texts[i]) for i in cands],
                                    batch_size=self.batch_size, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))[:top_k]
        return [(cands[p], float(scores[p])) for p in order]
