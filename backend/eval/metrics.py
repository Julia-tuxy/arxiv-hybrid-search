"""Ranking metrics for binary-relevance retrieval evaluation.

All functions take:
    ranked: list of doc ids returned by the system, best first
    relevant: set of doc ids judged relevant for the query
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG with the standard log2(rank+1) discount."""
    dcg = 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(per_query: Iterable[dict]) -> dict:
    """Mean of each metric across queries."""
    rows = list(per_query)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def score_run(ranked: Sequence[str], relevant: set[str]) -> dict:
    return {
        "recall@10": recall_at_k(ranked, relevant, 10),
        "recall@100": recall_at_k(ranked, relevant, 100),
        "mrr@10": reciprocal_rank(ranked, relevant, 10),
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
    }
