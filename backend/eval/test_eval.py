"""Correctness tests for the evaluation harness.

Run with:  python -m pytest test_eval.py -q     (or: python test_eval.py)

These guard the two things an ablation cannot survive being wrong about: the
metrics, and the BM25 scores the ablation is comparing.
"""
from __future__ import annotations

import math

import numpy as np

import _path  # noqa: F401  -- puts backend/ on sys.path
from bm25_index import SparseBM25
from fusion import arms_that_found, minmax_normalise, weighted_fusion
from metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from run_eval import paired_bootstrap


# ---------------------------------------------------------------- metrics
def test_recall():
    assert recall_at_k(["a", "b", "c"], {"a"}, 10) == 1.0
    assert recall_at_k(["x", "b", "c"], {"a"}, 10) == 0.0
    assert recall_at_k(["x", "a"], {"a"}, 1) == 0.0
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 10) == 1.0
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(["a"], {"a"}, 10) == 1.0
    assert reciprocal_rank(["x", "a"], {"a"}, 10) == 0.5
    assert reciprocal_rank(["x", "a"], {"a"}, 1) == 0.0
    assert reciprocal_rank(["x"], {"a"}, 10) == 0.0


def test_ndcg_against_hand_computation():
    assert abs(ndcg_at_k(["a"], {"a"}, 10) - 1.0) < 1e-9
    assert abs(ndcg_at_k(["x", "a"], {"a"}, 10) - 1 / math.log2(3)) < 1e-9
    assert abs(ndcg_at_k(["a", "b"], {"a", "b"}, 10) - 1.0) < 1e-9
    dcg = 1 + 1 / math.log2(4)          # relevant at ranks 1 and 3
    idcg = 1 + 1 / math.log2(3)         # ideal: ranks 1 and 2
    assert abs(ndcg_at_k(["a", "x", "b"], {"a", "b"}, 10) - dcg / idcg) < 1e-9
    assert ndcg_at_k(["x", "y"], {"a"}, 10) == 0.0


# ---------------------------------------------------------------- BM25
def _reference_bm25(query, corpus, k1=1.5, b=0.75):
    """Textbook BM25, written independently of the sparse implementation."""
    n = len(corpus)
    dl = np.array([len(d) for d in corpus], dtype=float)
    avgdl = dl.mean()
    df: dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    out = np.zeros(n)
    for term in query:
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, doc in enumerate(corpus):
            tf = doc.count(term)
            if tf:
                out[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[i] / avgdl))
    return out


def test_sparse_bm25_matches_reference():
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(40)]
    corpus = [[vocab[i] for i in rng.integers(0, 40, rng.integers(5, 30))]
              for _ in range(200)]
    index = SparseBM25.build(corpus)
    for _ in range(20):
        q = [vocab[i] for i in rng.integers(0, 40, rng.integers(1, 5))]
        assert np.allclose(index.scores(q), _reference_bm25(q, corpus), atol=1e-4)


def test_sparse_bm25_search_is_sorted_and_consistent():
    rng = np.random.default_rng(1)
    vocab = [f"w{i}" for i in range(20)]
    corpus = [[vocab[i] for i in rng.integers(0, 20, 12)] for _ in range(100)]
    index = SparseBM25.build(corpus)
    ids, scores = index.search(["w1", "w2"], 10)
    assert list(scores) == sorted(scores, reverse=True)
    full = index.scores(["w1", "w2"])
    assert np.allclose(full[ids], scores)


def test_out_of_vocabulary_query_scores_zero():
    index = SparseBM25.build([["a", "b"], ["b", "c"]])
    assert index.scores(["zzz"]).sum() == 0.0


# ---------------------------------------------------------------- fusion
def test_minmax_normalise():
    assert minmax_normalise([]) == {}
    assert minmax_normalise([(1, 5.0)]) == {1: 1.0}          # degenerate span
    assert minmax_normalise([(1, 2.0), (2, 2.0)]) == {1: 1.0, 2: 1.0}
    out = minmax_normalise([(1, 0.0), (2, 5.0), (3, 10.0)])
    assert out == {1: 0.0, 2: 0.5, 3: 1.0}


def test_weighted_fusion_takes_the_union_of_both_arms():
    dense = [(1, 1.0), (2, 0.0)]      # -> 1.0, 0.0
    sparse = [(3, 4.0), (2, 0.0)]     # -> 1.0, 0.0
    fused = weighted_fusion(dense, sparse, alpha=0.5)
    assert {doc for doc, *_ in fused} == {1, 2, 3}           # doc 3 is reachable
    scores = {doc: s for doc, s, *_ in fused}
    assert scores[1] == 0.5 and scores[3] == 0.5 and scores[2] == 0.0
    assert [s for _, s, *_ in fused] == sorted(scores.values(), reverse=True)


def test_weighted_fusion_alpha_endpoints_reduce_to_single_arms():
    dense = [(1, 1.0), (2, 0.0)]
    sparse = [(2, 1.0), (1, 0.0)]
    assert weighted_fusion(dense, sparse, alpha=1.0)[0][0] == 1   # dense only
    assert weighted_fusion(dense, sparse, alpha=0.0)[0][0] == 2   # BM25 only


def test_arms_that_found():
    d, s = {1: 1.0, 2: 0.5}, {2: 1.0, 3: 0.5}
    assert arms_that_found(1, d, s) == "dense"
    assert arms_that_found(2, d, s) == "both"
    assert arms_that_found(3, d, s) == "bm25"


# ---------------------------------------------------------------- statistics
def test_paired_bootstrap_calibration():
    rng = np.random.default_rng(2)
    a = rng.random(120)
    assert paired_bootstrap(a, a) == 1.0                 # identical -> no evidence
    assert paired_bootstrap(a, a + 0.05) < 0.01          # uniform win -> significant
    assert paired_bootstrap(a, a - 0.05) == 1.0          # a loss is never significant


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
    print("all tests passed")
