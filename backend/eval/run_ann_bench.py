"""Vector index benchmark: Flat vs IVF vs IVF-PQ vs HNSW on the 100K arXiv vectors.

Measures, per index and per query-time setting:

  * index recall@1/10/100   -- agreement with exact (Flat) search
  * task nDCG@10            -- known-item quality of the dense arm alone AND of the
                               served hybrid (dense + BM25, alpha=0.5), per query set,
                               so it is visible whether ANN error survives fusion
  * P50 / P95 / P99 latency -- single-query search, encoding excluded
  * build time (train + add), serialized size, resident memory of a fresh
    process holding only that index
  * updates: incremental add throughput, recall after incremental adds vs a
    full rebuild, deletes (physical for Flat/IVF, tombstone filter for HNSW)

Query vectors are the 740 evaluation queries encoded with the served model; ground
truth is exact inner-product search over the same vectors.

    python run_ann_bench.py                  # -> results/ann_bench.json
    python plot_ann.py                       # -> results/ann_frontier.png
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
# torch and faiss in one process: see "macOS: two OpenMP runtimes" in README.md
from sentence_transformers import SentenceTransformer

import _path  # noqa: F401  -- puts backend/ on sys.path
import ann_index
from ann_index import IndexSpec
from bm25_index import SparseBM25
from fusion import weighted_fusion
from metrics import ndcg_at_k
from run_eval import load_queries
from tokenizer import standard_tokens

HERE = Path(__file__).resolve().parent
ALPHA, K_EACH = 0.5, 100

SWEEPS = {
    "flat":  [None],
    "ivf":   [1, 2, 4, 8, 16, 32, 64, 128],
    "ivfpq": [4, 8, 16, 32, 64, 128],
    "hnsw":  [16, 32, 64, 128, 256, 512],
}


# ------------------------------------------------------------------ helpers
def pct(xs: list[float]) -> dict:
    a = np.asarray(xs)
    return {"p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "mean_ms": round(float(a.mean()), 3)}


def time_single_queries(index, q: np.ndarray, k: int, repeats: int) -> list[float]:
    for i in range(min(20, len(q))):                       # warm-up
        index.search(q[i:i + 1], k)
    out = []
    for _ in range(repeats):
        for i in range(len(q)):
            t0 = time.perf_counter()
            index.search(q[i:i + 1], k)
            out.append((time.perf_counter() - t0) * 1000)
    return out


def apply_param(index, kind: str, p) -> None:
    if kind in ("ivf", "ivfpq"):
        ann_index.set_search_params(index, nprobe=p)
    elif kind == "hnsw":
        ann_index.set_search_params(index, ef_search=p)


def rss_of_index_in_fresh_process(path: Path) -> int:
    """Resident bytes added by loading the index, in a process that holds nothing else."""
    code = (
        "import faiss, resource, sys, gc\n"
        "def rss():\n"
        "    try:\n"
        "        import psutil; return psutil.Process().memory_info().rss\n"
        "    except ImportError:\n"
        "        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "        return r if sys.platform == 'darwin' else r * 1024\n"
        "gc.collect(); a = rss()\n"
        f"ix = faiss.read_index({str(path)!r})\n"
        "gc.collect(); print(rss() - a)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return int(out.stdout.strip().splitlines()[-1])


def task_ndcg(ann_ids: np.ndarray, ann_scores: np.ndarray, queries: list[dict],
              bm25_runs: list[list[tuple[int, float]]]) -> dict:
    """nDCG@10 of the dense arm and of the hybrid, per query type."""
    by: dict[str, dict[str, list[float]]] = {}
    for i, q in enumerate(queries):
        rel = set(q["relevant"])
        dense = [(int(d), float(s)) for d, s in zip(ann_ids[i], ann_scores[i]) if d >= 0]
        fused = weighted_fusion(dense, bm25_runs[i], ALPHA)
        slot = by.setdefault(q["type"], {"dense": [], "hybrid": []})
        slot["dense"].append(ndcg_at_k([d for d, _ in dense], rel, 10))
        slot["hybrid"].append(ndcg_at_k([d for d, *_ in fused], rel, 10))
    return {t: {k: round(float(np.mean(v)), 4) for k, v in m.items()} for t, m in by.items()}


# ------------------------------------------------------------------ update tests
def update_test(spec: IndexSpec, xb: np.ndarray, xq: np.ndarray, gt_full: np.ndarray,
                param, base_frac: float = 0.9, batch: int = 1000) -> dict:
    """Build on the first 90% of vectors, stream the rest in, then delete some."""
    n = len(xb)
    n0 = int(n * base_frac)
    index, _ = ann_index.build(spec, xb[:n0])
    apply_param(index, spec.kind, param)

    add_ms = []
    for s in range(n0, n, batch):
        t0 = time.perf_counter()
        index.add(xb[s:s + batch])
        add_ms.append((time.perf_counter() - t0) * 1000)
    _, inc_ids = index.search(xq, 10)
    out = {
        "base_vectors": n0, "added_vectors": n - n0, "batch": batch,
        "add_ms_per_batch": pct(add_ms),
        "add_vectors_per_s": round(batch / (np.median(add_ms) / 1000), 1),
        "recall@10_after_incremental_add": round(ann_index.recall_at_k(inc_ids, gt_full, 10), 4),
    }

    # deletes: 1,000 ids that are currently in some query's exact top-10, so a
    # delete that does not take effect is guaranteed to show up
    rng = np.random.default_rng(0)
    pool = np.unique(gt_full[:, :10])
    dels = np.sort(rng.choice(pool, min(1000, len(pool)), replace=False)).astype(np.int64)

    if spec.kind == "hnsw":
        t0 = time.perf_counter()
        _, ids = ann_index.search(index, xq, 10, exclude_ids=dels)
        filt_ms = (time.perf_counter() - t0) * 1000 / len(xq)
        t0 = time.perf_counter()
        index.search(xq, 10)
        plain_ms = (time.perf_counter() - t0) * 1000 / len(xq)
        out["delete"] = {
            "method": "tombstone filter (IDSelector) at query time",
            "deleted": int(len(dels)),
            "leaked_deleted_ids": int(np.isin(ids, dels).sum()),
            "search_ms_per_query_filtered": round(filt_ms, 3),
            "search_ms_per_query_unfiltered": round(plain_ms, 3),
        }
    else:
        t0 = time.perf_counter()
        removed = ann_index.remove(index, dels)
        rm_ms = (time.perf_counter() - t0) * 1000
        out["delete"] = {"method": "remove_ids (physical)", "deleted": removed,
                         "remove_ms_total": round(rm_ms, 2)}
        if spec.kind != "flat":           # IVF keeps explicit ids; Flat renumbers
            _, ids = index.search(xq, 10)
            out["delete"]["leaked_deleted_ids"] = int(np.isin(ids, dels).sum())
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=os.environ.get("INDEX_DIR", str(HERE.parent / "index")))
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--out", default=str(HERE / "results" / "ann_bench.json"))
    ap.add_argument("--threads", type=int, default=1,
                    help="FAISS OpenMP threads for latency (1 = per-core cost)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--kinds", nargs="+", default=list(SWEEPS))
    args = ap.parse_args()

    import faiss

    index_dir = Path(args.index_dir)
    flat = faiss.read_index(str(index_dir / "faiss_index.bin"))
    xb = flat.reconstruct_n(0, flat.ntotal).astype(np.float32)
    print(f"{xb.shape[0]:,} vectors x {xb.shape[1]} dims")

    queries = load_queries(sorted((HERE / "queries").glob("*_queries.jsonl")))
    model = SentenceTransformer(args.model, device="cpu")
    xq = model.encode([q["query"] for q in queries], batch_size=64,
                      normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    print(f"{len(xq)} query vectors")

    bm25 = SparseBM25.load(index_dir / "bm25_standard")
    bm25_runs = []
    for q in queries:
        ids, sc = bm25.search(standard_tokens(q["query"]), K_EACH)
        bm25_runs.append([(int(i), float(s)) for i, s in zip(ids, sc)])

    faiss.omp_set_num_threads(os.cpu_count() or 1)
    gt_scores, gt = flat.search(xq, K_EACH)              # exact ground truth
    faiss.omp_set_num_threads(args.threads)

    specs = {"flat": IndexSpec("flat"), "ivf": IndexSpec("ivf", nlist=1024),
             "ivfpq": IndexSpec("ivfpq", nlist=1024, pq_m=48),
             "hnsw": IndexSpec("hnsw", hnsw_m=32, ef_construction=200)}
    results = {"config": {"n_vectors": int(xb.shape[0]), "dim": int(xb.shape[1]),
                          "n_queries": len(xq), "faiss_threads": args.threads,
                          "faiss_version": faiss.__version__, "alpha": ALPHA,
                          "k_each": K_EACH, "repeats": args.repeats,
                          "raw_vectors_mb": round(xb.nbytes / 1e6, 1)},
               "indexes": {}}

    tmp = Path(tempfile.mkdtemp())
    for kind in args.kinds:
        spec = specs[kind]
        faiss.omp_set_num_threads(os.cpu_count() or 1)    # builds use every core
        index, bt = ann_index.build(spec, xb)
        faiss.omp_set_num_threads(args.threads)
        path = tmp / f"{kind}.bin"
        ann_index.save(index, path)
        entry = {"spec": spec.label,
                 "build": {k: round(v, 2) for k, v in bt.items()},
                 "size_mb": round(ann_index.serialized_bytes(index) / 1e6, 1),
                 "rss_mb": round(rss_of_index_in_fresh_process(path) / 1e6, 1),
                 "sweep": []}
        print(f"\n[{spec.label}] build train={bt['train_s']:.1f}s add={bt['add_s']:.1f}s "
              f"size={entry['size_mb']}MB rss={entry['rss_mb']}MB")
        for p in SWEEPS[kind]:
            apply_param(index, kind, p)
            ids_scores = index.search(xq, K_EACH)
            sc, ids = ids_scores
            lat = time_single_queries(index, xq, K_EACH, args.repeats)
            row = {"param": p,
                   "recall@1": round(ann_index.recall_at_k(ids, gt, 1), 4),
                   "recall@10": round(ann_index.recall_at_k(ids, gt, 10), 4),
                   "recall@100": round(ann_index.recall_at_k(ids, gt, 100), 4),
                   "latency": pct(lat),
                   "ndcg@10": task_ndcg(ids, sc, queries, bm25_runs)}
            entry["sweep"].append(row)
            print(f"  param={p!s:>4} r@1={row['recall@1']:.3f} r@10={row['recall@10']:.3f} "
                  f"r@100={row['recall@100']:.3f} p50={row['latency']['p50_ms']:.2f}ms "
                  f"p99={row['latency']['p99_ms']:.2f}ms "
                  f"hybrid-short={row['ndcg@10']['short']['hybrid']:.4f}")

        # updates at a representative operating point
        upd_param = {"flat": None, "ivf": 16, "ivfpq": 32, "hnsw": 64}[kind]
        faiss.omp_set_num_threads(os.cpu_count() or 1)
        entry["updates"] = update_test(spec, xb, xq, gt, upd_param)
        entry["updates"]["param"] = upd_param
        apply_param(index, kind, upd_param)
        _, full_ids = index.search(xq, 10)
        entry["updates"]["recall@10_full_rebuild"] = round(
            ann_index.recall_at_k(full_ids, gt, 10), 4)
        faiss.omp_set_num_threads(args.threads)
        print(f"  updates: {json.dumps(entry['updates'])}")
        results["indexes"][kind] = entry
        path.unlink()
        del index

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
