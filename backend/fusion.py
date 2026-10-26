"""Score fusion shared by the API and the evaluation harness.

Kept dependency-free and side-effect-free on purpose: `search.py` loads ~300 MB
of artifacts at import time, so the ranking logic lives here where it can be
imported and unit-tested on its own -- and where the served code and the
measured code are literally the same function rather than two copies that drift.
"""

from __future__ import annotations

Pairs = list[tuple[int, float]]


def minmax_normalise(pairs: Pairs) -> dict[int, float]:
    """Scale one arm's scores into [0, 1] across the documents it returned.

    Normalising per returned list (rather than per corpus) is what makes two
    arms with incomparable score scales -- cosine similarity and BM25 -- add up
    to something meaningful. It also means the arm's own top hit is always 1.0.
    """
    if not pairs:
        return {}
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if hi - lo <= 0:
        return {doc: 1.0 for doc, _ in pairs}
    span = hi - lo
    return {doc: (s - lo) / span for doc, s in pairs}


def weighted_fusion(dense: Pairs, sparse: Pairs, alpha: float
                    ) -> list[tuple[int, float, float, float]]:
    """Fuse two ranked lists into (doc, fused, dense_norm, sparse_norm), best first.

    The union is taken over both lists, so a document either arm found is
    reachable; the arm that missed it contributes 0 rather than excluding it.
    """
    d = minmax_normalise(dense)
    s = minmax_normalise(sparse)
    fused = [(doc, alpha * d.get(doc, 0.0) + (1.0 - alpha) * s.get(doc, 0.0),
              d.get(doc, 0.0), s.get(doc, 0.0))
             for doc in d.keys() | s.keys()]
    fused.sort(key=lambda t: -t[1])
    return fused


def arms_that_found(doc: int, dense: dict[int, float], sparse: dict[int, float]) -> str:
    if doc in dense and doc in sparse:
        return "both"
    return "dense" if doc in dense else "bm25"
