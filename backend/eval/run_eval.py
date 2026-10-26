"""Offline retrieval ablation over the arXiv index.

Usage:
    python run_eval.py --index-dir ../index --metadata ../index/metadata.parquet \
                       --model all-MiniLM-L6-v2 --out results/
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401  -- puts backend/ on sys.path
from bm25_index import SparseBM25
from metrics import aggregate, score_run
from retrievers import (BM25Retriever, DenseRetriever, LegacyHybrid, RRFusion,
                        WeightedFusion)
from tokenizer import simple_tokens, standard_tokens

HERE = Path(__file__).resolve().parent
TOP_K = 100          # depth scored
K_EACH = 100         # candidates each arm contributes to fusion


# --------------------------------------------------------------------------
def load_queries(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows


def evaluate(system, queries: list[dict]) -> tuple[dict, list[float]]:
    per_query = []
    for q in queries:
        ranked = [doc for doc, _ in system.search(q["query"], TOP_K)]
        per_query.append(score_run(ranked, set(q["relevant"])))
    return aggregate(per_query), [r["ndcg@10"] for r in per_query]


def measure_latency(system, queries: list[dict], repeats: int = 1) -> dict:
    for r in (getattr(system, "dense", None), getattr(system, "sparse", None), system):
        if hasattr(r, "cache_enabled"):
            r.cache_enabled = False
    times = []
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            system.search(q["query"], 10)
            times.append((time.perf_counter() - t0) * 1000)
    for r in (getattr(system, "dense", None), getattr(system, "sparse", None), system):
        if hasattr(r, "cache_enabled"):
            r.cache_enabled = True
    times.sort()
    return {"p50_ms": times[len(times) // 2],
            "p95_ms": times[int(len(times) * 0.95)],
            "mean_ms": statistics.fmean(times)}


def paired_bootstrap(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> float:
    """One-sided p-value that b > a, by the paired bootstrap of Koehn (2004).

    Resample query indices with replacement; under the null the mean difference
    is 0, so the resampled deltas are re-centred by subtracting the observed
    delta and the p-value is how often the centred delta still reaches it.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    obs = d.mean()
    if obs <= 0:
        return 1.0
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(axis=1)
    return float((boot - obs >= obs).mean())


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(HERE.parent / "index"))
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--queries", nargs="+", default=None)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--latency-n", type=int, default=50)
    args = ap.parse_args()

    import faiss
    from encoder import QueryEncoder

    index_dir = Path(args.index_dir)
    meta_path = Path(args.metadata or index_dir / "metadata.parquet")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("loading artifacts…")
    df = pd.read_parquet(meta_path)
    faiss_index = faiss.read_index(str(index_dir / "faiss_index.bin"))
    bm25_simple = SparseBM25.load(index_dir / "bm25_simple")
    bm25_standard = SparseBM25.load(index_dir / "bm25_standard")
    encoder = QueryEncoder(args.model)
    legacy_corpus = [simple_tokens(t) for t in
                     ("[TITLE] " + df.title + " [ABSTRACT] " + df.abstract)]

    dense = DenseRetriever(faiss_index, encoder)
    sparse_std = BM25Retriever(bm25_standard, standard_tokens, name="bm25_standard")
    sparse_sim = BM25Retriever(bm25_simple, simple_tokens, name="bm25_simple")

    systems = [
        sparse_sim,
        sparse_std,
        dense,
        LegacyHybrid(dense, legacy_corpus, alpha=0.7, faiss_k=K_EACH),
    ]
    systems += [WeightedFusion(dense, sparse_std, alpha=a, k_each=K_EACH)
                for a in np.round(np.arange(0.0, 1.01, 0.1), 2)]
    systems += [RRFusion(dense, sparse_std, k_rrf=k, k_each=K_EACH) for k in (10, 60)]

    query_files = ([Path(p) for p in args.queries] if args.queries
                   else sorted((HERE / "queries").glob("*_queries.jsonl")))
    queries = load_queries(query_files)
    by_type: dict[str, list[dict]] = {}
    for q in queries:
        by_type.setdefault(q["type"], []).append(q)
    print({k: len(v) for k, v in by_type.items()})

    results: dict = {"config": {"top_k": TOP_K, "k_each": K_EACH,
                                "model": args.model, "n_docs": int(faiss_index.ntotal)},
                     "runs": {}}
    ndcg_vectors: dict[str, dict[str, list[float]]] = {}

    for sysm in systems:
        entry = {}
        for qtype, qs in by_type.items():
            t0 = time.time()
            agg, vec = evaluate(sysm, qs)
            entry[qtype] = {k: round(v, 4) for k, v in agg.items()}
            ndcg_vectors.setdefault(sysm.name, {})[qtype] = vec
            print(f"{sysm.name:24s} {qtype:6s} "
                  f"ndcg@10={agg['ndcg@10']:.4f} r@10={agg['recall@10']:.4f} "
                  f"r@100={agg['recall@100']:.4f} mrr@10={agg['mrr@10']:.4f} "
                  f"({time.time()-t0:.1f}s)")
        results["runs"][sysm.name] = entry

    # latency on a fixed subset, caches off
    lat_qs = (by_type.get("short") or by_type["nl"])[:args.latency_n]
    for sysm in systems:
        if sysm.name in ("bm25_simple", "bm25_standard", "dense", "hybrid_legacy",
                         "fusion_weighted_a0.5", "fusion_rrf_k60"):
            lat = measure_latency(sysm, lat_qs)
            results["runs"][sysm.name]["latency"] = {k: round(v, 2) for k, v in lat.items()}
            print(f"{sysm.name:24s} latency p50={lat['p50_ms']:.1f}ms "
                  f"p95={lat['p95_ms']:.1f}ms")

    # significance: best fusion vs the two single-signal baselines and the legacy hybrid
    results["significance"] = {}
    for qtype in by_type:
        best = max((s.name for s in systems if s.name.startswith("fusion_")),
                   key=lambda n: results["runs"][n][qtype]["ndcg@10"])
        results["significance"][qtype] = {"best_fusion": best, "vs": {}}
        for base in ("dense", "bm25_standard", "hybrid_legacy"):
            p = paired_bootstrap(ndcg_vectors[base][qtype], ndcg_vectors[best][qtype])
            results["significance"][qtype]["vs"][base] = round(p, 5)

    results["per_query_ndcg10"] = ndcg_vectors
    (out_dir / "ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir / 'ablation.json'}")


if __name__ == "__main__":
    main()
