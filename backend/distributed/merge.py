"""Merging per-shard top-k lists into a global top-k.

Why this is exact, not approximate: shards partition the corpus, so every
document's score is computed on exactly one shard, and that score is the same
number a single node would compute --

  * dense: inner product between the query vector and the document vector,
    which does not depend on any other document;
  * BM25: shards hold rows of the globally-built weight matrix (global IDF and
    average length), so a document's BM25 score is also shard-independent.

If document d is in the global top-k, at most k-1 documents outscore it in the
whole corpus, hence at most k-1 on its own shard, hence it is in its shard's
top-k. So the union of per-shard top-k lists contains the global top-k, and a
k-way merge by score recovers it. Each shard must return k (not k/N) results;
asking each for k/N would silently drop results whenever the true top-k is
unevenly spread, which for k=100 over 2 shards it nearly always is.

Score fusion runs *after* the merge, on the global lists, because min-max
normalisation depends on the whole list: normalising per shard and merging
would compare scores on different scales.
"""
from __future__ import annotations

import heapq
from typing import Iterable

Hits = list[tuple[int, float]]


def merge_topk(per_shard: Iterable[Hits], k: int) -> Hits:
    """k-way merge of score-descending lists; ties broken by doc id for determinism."""
    streams = [sorted(h, key=lambda t: (-t[1], t[0])) for h in per_shard if h]
    merged = heapq.merge(*streams, key=lambda t: (-t[1], t[0]))
    out, seen = [], set()
    for doc, score in merged:
        if doc in seen:                 # a doc is on one shard; guard anyway
            continue
        seen.add(doc)
        out.append((doc, score))
        if len(out) == k:
            break
    return out
