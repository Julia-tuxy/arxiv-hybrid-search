"""Corpus-wide BM25 over a sparse CSC matrix.

Why not rank_bm25: the original pipeline re-instantiated BM25Okapi on the ~100
FAISS candidates of every request. That makes IDF a function of the candidate
set rather than the corpus, and makes BM25 incapable of *retrieving* anything --
it can only re-score what the dense stage already found.

This index scores the whole corpus. The BM25 weight of a (doc, term) pair does
not depend on the query, so the entire saturated weight matrix

    W[d, t] = idf[t] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len[d] / avgdl))

is precomputed once at build time. A query is then a sum of |q| sparse columns.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import sparse


class SparseBM25:
    def __init__(self, matrix_csc: sparse.csc_matrix, vocab: dict[str, int],
                 k1: float = 1.5, b: float = 0.75):
        self.W = matrix_csc
        self.vocab = vocab
        self.k1 = k1
        self.b = b
        self.n_docs = matrix_csc.shape[0]

    # ---------- build ----------
    @classmethod
    def build(cls, corpus_tokens: list[list[str]], k1: float = 1.5,
              b: float = 0.75) -> "SparseBM25":
        vocab: dict[str, int] = {}
        indptr = [0]
        indices: list[int] = []
        tf_data: list[int] = []
        doc_len = np.empty(len(corpus_tokens), dtype=np.float32)

        for d, tokens in enumerate(corpus_tokens):
            doc_len[d] = len(tokens)
            counts: dict[int, int] = {}
            for tok in tokens:
                tid = vocab.get(tok)
                if tid is None:
                    tid = len(vocab)
                    vocab[tok] = tid
                counts[tid] = counts.get(tid, 0) + 1
            indices.extend(counts.keys())
            tf_data.extend(counts.values())
            indptr.append(len(indices))

        n_docs, n_terms = len(corpus_tokens), len(vocab)
        tf = sparse.csr_matrix(
            (np.asarray(tf_data, dtype=np.float32),
             np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int64)),
            shape=(n_docs, n_terms),
        )

        # document frequency -> idf (BM25Okapi / Lucene form, always positive)
        df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.float32)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # length normalisation, applied per row
        avgdl = float(doc_len.mean())
        norm = (k1 * (1.0 - b + b * doc_len / avgdl)).astype(np.float32)

        # W[d, t] = idf[t] * tf * (k1 + 1) / (tf + norm[d])
        W = tf.tocoo()
        denom = W.data + norm[W.row]
        W.data = idf[W.col] * W.data * (k1 + 1.0) / denom
        return cls(W.tocsc(), vocab, k1=k1, b=b)

    # ---------- search ----------
    def scores(self, query_tokens: list[str]) -> np.ndarray:
        cols = [self.vocab[t] for t in query_tokens if t in self.vocab]
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not cols:
            return out
        indptr, indices, data = self.W.indptr, self.W.indices, self.W.data
        for c in cols:                       # summing per occurrence matches
            s, e = indptr[c], indptr[c + 1]  # rank_bm25's handling of repeats
            np.add.at(out, indices[s:e], data[s:e])
        return out

    def search(self, query_tokens: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(query_tokens)
        k = min(k, self.n_docs)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return top, s[top]

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        sparse.save_npz(path.with_suffix(".npz"), self.W)
        with open(path.with_suffix(".vocab.pkl"), "wb") as f:
            pickle.dump({"vocab": self.vocab, "k1": self.k1, "b": self.b}, f)

    @classmethod
    def load(cls, path: str | Path) -> "SparseBM25":
        path = Path(path)
        W = sparse.load_npz(path.with_suffix(".npz")).tocsc()
        with open(path.with_suffix(".vocab.pkl"), "rb") as f:
            meta = pickle.load(f)
        return cls(W, meta["vocab"], k1=meta["k1"], b=meta["b"])
