"""Stage-2 experiment: does cross-encoder reranking earn its latency?

First stage is the served configuration (weighted fusion, alpha=0.5, 100
candidates per arm). A cross-encoder then re-scores the top `--depth`
candidates; everything below the reranked prefix keeps its first-stage order,
which is how a reranker behaves in production and is what keeps Recall@100
comparable across depths.

Cross-encoder scores are computed once per (model, query) over the full depth
and cached, so the depth sweep is free: reranking the top 10 is the top-100
scores restricted to the first 10 candidates.

Usage:
    python run_rerank_eval.py --ce-models cross-encoder/ms-marco-MiniLM-L-6-v2 \
                                          cross-encoder/ms-marco-MiniLM-L-12-v2
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401  -- puts backend/ on sys.path
from bm25_index import SparseBM25
from metrics import aggregate, score_run
from retrievers import BM25Retriever, DenseRetriever, WeightedFusion
from run_eval import load_queries, paired_bootstrap
from tokenizer import standard_tokens

HERE = Path(__file__).resolve().parent
DEPTHS = (10, 20, 50, 100)


def doc_text(row) -> str:
    return f"{row.title}. {row.abstract}"


def first_stage_runs(system, queries: list[dict], depth: int) -> dict[str, list[int]]:
    return {q["qid"]: [doc for doc, _ in system.search(q["query"], depth)] for q in queries}


def predict_in_length_order(predict_fn, pairs: list[tuple[str, str]]) -> np.ndarray:
    """Score pairs shortest-first, then restore the caller's order.

    Batches are padded to their longest member, so scoring a random mix of
    5-token and 400-token documents wastes most of the compute on padding.
    Sorting by length first removes that; the permutation is inverted so the
    caller never sees the reordering.
    """
    order = np.argsort([len(a) + len(b) for a, b in pairs], kind="stable")
    scored = np.asarray(predict_fn([pairs[i] for i in order]), dtype=np.float32)
    out = np.empty(len(pairs), dtype=np.float32)
    out[order] = scored
    return out


def cross_encode(model_name: str, queries: list[dict], runs: dict[str, list[int]],
                 texts: list[str], batch_size: int, max_length: int, device: str):
    """Score every (query, candidate) pair. Returns scores per qid and pairs/second."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device, max_length=max_length)

    flat, owner = [], []
    for q in queries:
        for doc in runs[q["qid"]]:
            flat.append((q["query"], texts[doc]))
            owner.append(q["qid"])

    t0 = time.perf_counter()
    out_flat = predict_in_length_order(
        lambda ps: model.predict(ps, batch_size=batch_size, show_progress_bar=False),
        flat,
    )
    elapsed = time.perf_counter() - t0

    per_query: dict[str, list[float]] = {}
    for qid, s in zip(owner, out_flat):
        per_query.setdefault(qid, []).append(float(s))
    return per_query, len(flat) / elapsed


def rerank(candidates: list[int], scores: list[float], depth: int) -> list[int]:
    head = sorted(range(min(depth, len(candidates))), key=lambda i: -scores[i])
    return [candidates[i] for i in head] + candidates[depth:]


def evaluate(queries: list[dict], runs, scores, depth) -> tuple[dict, list[float]]:
    rows = []
    for q in queries:
        ranked = (runs[q["qid"]] if depth == 0
                  else rerank(runs[q["qid"]], scores[q["qid"]], depth))
        rows.append(score_run(ranked, set(q["relevant"])))
    return aggregate(rows), [r["ndcg@10"] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(HERE.parent / "index"))
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--ce-models", nargs="+",
                    default=["cross-encoder/ms-marco-MiniLM-L-6-v2",
                             "cross-encoder/ms-marco-MiniLM-L-12-v2"])
    ap.add_argument("--queries", nargs="+", default=None)
    ap.add_argument("--title-sample", type=int, default=150,
                    help="the title set is saturated; subsample it to save compute")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache", default=str(HERE / "results" / "ce_scores"))
    ap.add_argument("--out", default=str(HERE / "results" / "rerank.json"))
    args = ap.parse_args()

    import faiss
    from encoder import QueryEncoder

    index_dir = Path(args.index_dir)
    df = pd.read_parquet(args.metadata or index_dir / "metadata.parquet")
    texts = [doc_text(r) for r in df.itertuples()]

    dense = DenseRetriever(faiss.read_index(str(index_dir / "faiss_index.bin")),
                           QueryEncoder(args.model))
    sparse = BM25Retriever(SparseBM25.load(index_dir / "bm25_standard"), standard_tokens)
    first = WeightedFusion(dense, sparse, alpha=args.alpha, k_each=100)

    files = ([Path(p) for p in args.queries] if args.queries
             else sorted((HERE / "queries").glob("*_queries.jsonl")))
    by_type: dict[str, list[dict]] = {}
    for q in load_queries(files):
        by_type.setdefault(q["type"], []).append(q)
    if "title" in by_type and args.title_sample:
        by_type["title"] = by_type["title"][:args.title_sample]
    print({k: len(v) for k, v in by_type.items()})

    print("running first stage…")
    runs = {t: first_stage_runs(first, qs, args.depth) for t, qs in by_type.items()}

    results = {"config": {"first_stage": f"weighted_fusion_alpha{args.alpha}",
                          "depth": args.depth, "max_length": args.max_length,
                          "batch_size": args.batch_size, "device": args.device,
                          "n_queries": {k: len(v) for k, v in by_type.items()}},
               "first_stage": {}, "rerank": {}, "throughput": {}}
    baseline_vecs = {}
    for t, qs in by_type.items():
        agg, vec = evaluate(qs, runs[t], None, 0)
        results["first_stage"][t] = {k: round(v, 4) for k, v in agg.items()}
        baseline_vecs[t] = vec
        print(f"  first stage {t:6s} ndcg@10={agg['ndcg@10']:.4f}")

    cache_dir = Path(args.cache); cache_dir.mkdir(parents=True, exist_ok=True)
    for ce in args.ce_models:
        tag = ce.split("/")[-1]
        results["rerank"][tag] = {}
        for t, qs in by_type.items():
            cache = cache_dir / f"{tag}__{t}__d{args.depth}.pkl"
            if cache.exists():
                scores, pps = pickle.loads(cache.read_bytes())
                print(f"  [{tag}] {t}: cached ({pps:.0f} pairs/s at build time)")
            else:
                print(f"  [{tag}] {t}: scoring {len(qs) * args.depth:,} pairs…", flush=True)
                scores, pps = cross_encode(ce, qs, runs[t], texts,
                                           args.batch_size, args.max_length, args.device)
                cache.write_bytes(pickle.dumps((scores, pps)))
                print(f"    {pps:.1f} pairs/s")
            results["throughput"].setdefault(tag, {})[t] = round(pps, 1)

            for d in DEPTHS:
                if d > args.depth:
                    continue
                agg, vec = evaluate(qs, runs[t], scores, d)
                p = paired_bootstrap(baseline_vecs[t], vec)
                entry = {k: round(v, 4) for k, v in agg.items()}
                entry["p_vs_first_stage"] = round(p, 5)
                entry["added_ms_p50"] = round(d / pps * 1000, 1)
                results["rerank"][tag].setdefault(t, {})[f"top{d}"] = entry
                print(f"    {t:6s} top{d:<3d} ndcg@10={agg['ndcg@10']:.4f} "
                      f"(first stage {results['first_stage'][t]['ndcg@10']:.4f}, "
                      f"p={p:.4f}, +{entry['added_ms_p50']:.0f}ms)")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
