"""Check that the served search still reproduces the measured ranking.

    python loadtest/verify_parity.py                                   # in-process model
    EMBEDDING_URL=http://127.0.0.1:8001 python loadtest/verify_parity.py   # via the service

Runs all 740 evaluation queries through the production `hybrid_search()` and
prints nDCG@10 per query set. The ablation (eval/README.md section 2) measured
title 0.9906 / paraphrased 0.9461 / keyword 0.9285; any other number means the
serving path changed the ranking.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND), str(BACKEND / "eval")]

from metrics import ndcg_at_k  # noqa: E402
from search import _meta, embedder_stats, hybrid_search  # noqa: E402

EXPECTED = {"title": 0.9906, "nl": 0.9461, "short": 0.9285}


def main() -> None:
    pos = {a: i for i, a in enumerate(_meta.arxiv_id.astype(str))}
    by: dict[str, list[float]] = {}
    for p in sorted((BACKEND / "eval" / "queries").glob("*_queries.jsonl")):
        for line in p.read_text().splitlines():
            q = json.loads(line)
            ranked = [pos[r["arxiv_id"]] for r in hybrid_search(q["query"], top_k=10)]
            by.setdefault(q["type"], []).append(ndcg_at_k(ranked, set(q["relevant"]), 10))
    got = {t: round(sum(v) / len(v), 4) for t, v in by.items()}
    ok = all(abs(got[t] - EXPECTED[t]) < 1e-4 for t in EXPECTED)
    print(json.dumps({"ndcg@10": got, "expected": EXPECTED, "match": ok, **embedder_stats()}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
