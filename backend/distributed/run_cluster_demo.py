"""Launch a 2-shard x 2-replica cluster and measure it end to end.

    cd backend && python -m distributed.run_cluster_demo     # -> eval/results/cluster.json

Processes (all real HTTP on localhost):
    embedding service  :8201
    shard 0            :9100 (replica 0a)   :9110 (replica 0b)
    shard 1            :9101 (replica 1a)   :9111 (replica 1b)
    gateway            :8200

Phases
  1. correctness   all 740 eval queries through the gateway vs an in-process
                   single-node search over the unsharded index: top-10 identity
                   and nDCG@10 per query set
  2. latency       gateway at concurrency 1 and 8, per-stage breakdown
  3. failover      SIGKILL one replica of shard 0 mid-load; then make a shard 1
                   replica hang past the coordinator timeout
  4. shard outage  kill the second shard-0 replica: partial results, quality cost
  5. versioning    build v2 (a different index type), load it next to v1 on every
                   replica, switch the gateway mid-load, check no response mixed versions
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import signal
import sys
import time
from pathlib import Path

import httpx
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND), str(BACKEND / "loadtest"), str(BACKEND / "eval")]

from run_load import Service, closed_loop, pct  # noqa: E402

from distributed import sharding  # noqa: E402
from fusion import weighted_fusion  # noqa: E402
from metrics import ndcg_at_k  # noqa: E402
from tokenizer import standard_tokens  # noqa: E402

INDEX = Path(os.environ.get("INDEX_DIR", BACKEND / "index"))
ROOT = Path(os.environ.get("SHARD_ROOT", INDEX / "shards"))
RESULTS = BACKEND / "eval" / "results"
EMBED_PORT, GATEWAY_PORT = 8201, 8200
REPLICAS = {(0, "a"): 9100, (0, "b"): 9110, (1, "a"): 9101, (1, "b"): 9111}


def load_eval_queries() -> list[dict]:
    rows = []
    for p in sorted((BACKEND / "eval" / "queries").glob("*_queries.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def ensure_version(version: str, kind: str) -> dict:
    if (ROOT / version / sharding.MANIFEST).exists():
        return sharding.read_manifest(ROOT, version)
    from distributed.build_shards import build
    return build(version, 2, kind, ROOT, INDEX, set_current=not (ROOT / "CURRENT").exists())


def shard_service(shard: int, rep: str, port: int) -> Service:
    return Service("distributed.shard_server", port,
                   {"SHARD_ROOT": ROOT, "SHARD_ID": shard, "REPLICA": f"{shard}{rep}"},
                   f"shard{shard}{rep}")


async def gateway_req(client, q, url):
    r = await client.get(f"{url}/search", params={"q": q, "top_k": 10})
    r.raise_for_status()
    return r.json()


def run_queries(url: str, texts: list[str], concurrency: int, warmup: bool = True) -> dict:
    # failure phases pass warmup=False: a fault hit during the untimed warm-up
    # opens the circuit breaker and hides the failover from the measurement
    return asyncio.run(closed_loop(lambda c, p: gateway_req(c, p, url), texts, concurrency,
                                   warmup=warmup))


def summarize(res: dict) -> dict:
    stages: dict[str, list[float]] = {}
    for b in res["bodies"]:
        for k, v in b["timings"].items():
            stages.setdefault(k, []).append(v)
    return {"requests": len(res["lat"]), "errors": res["errors"],
            "throughput_rps": round(len(res["lat"]) / res["wall_s"], 1),
            "client": pct(res["lat"]), "client_max_ms": round(max(res["lat"]), 1),
            "requests_over_500ms": sum(l > 500 for l in res["lat"]),
            "stages": {k: pct(v) for k, v in stages.items()},
            "partial": sum(b["partial"] for b in res["bodies"]),
            "failed_over": sum(any(s.get("failed_over") for s in b["shards"].values())
                               for b in res["bodies"])}


def ndcg_by_type(queries, bodies) -> dict:
    by: dict[str, list[float]] = {}
    for q, b in zip(queries, bodies):
        ranked = [r["doc"] for r in b["results"]]
        by.setdefault(q["type"], []).append(ndcg_at_k(ranked, set(q["relevant"]), 10))
    return {t: round(float(np.mean(v)), 4) for t, v in by.items()}


def ordered_bodies(url, queries, concurrency=8):
    """Results in query order (closed_loop returns completion order)."""
    async def go():
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=60) as client:
            async def one(q):
                async with sem:
                    return await gateway_req(client, q["query"], url)
            return await asyncio.gather(*(one(q) for q in queries))
    return asyncio.run(go())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-kind", default="hnsw")
    ap.add_argument("--out", default=str(RESULTS / "cluster.json"))
    args = ap.parse_args()

    import faiss
    from bm25_index import SparseBM25

    m1 = ensure_version("v1", "flat")
    queries = load_eval_queries()
    out: dict = {"manifest_v1": {k: m1[k] for k in ("version", "n_shards", "assignment", "n_docs", "index")}
                 | {"docs_per_shard": [s["n_docs"] for s in m1["shards"]]}}

    cfg = {"root": str(ROOT), "timeout_ms": 1000, "cooldown_s": 5,
           "shards": {"0": [f"http://127.0.0.1:{REPLICAS[(0, 'a')]}", f"http://127.0.0.1:{REPLICAS[(0, 'b')]}"],
                      "1": [f"http://127.0.0.1:{REPLICAS[(1, 'a')]}", f"http://127.0.0.1:{REPLICAS[(1, 'b')]}"]}}
    cfg_path = RESULTS / ".cluster.json"
    cfg_path.write_text(json.dumps(cfg))

    emb = Service("embedding_service", EMBED_PORT, {"EMBED_CACHE_SIZE": 0}, "cluster-embed")
    shards = {k: shard_service(k[0], k[1], p) for k, p in REPLICAS.items()}
    gw = Service("distributed.gateway", GATEWAY_PORT,
                 {"CLUSTER_CONFIG": cfg_path, "EMBEDDING_URL": f"http://127.0.0.1:{EMBED_PORT}",
                  "CLUSTER_VERSION": "v1"}, "gateway")
    services = [emb, *shards.values(), gw]
    try:
        emb.start()
        for s in shards.values():
            s.start()
        gw.start()

        # ---------------------------------------------------------- 1. correctness
        print("1. correctness vs single node")
        flat = faiss.read_index(str(INDEX / "faiss_index.bin"))
        faiss.omp_set_num_threads(1)
        bm25 = SparseBM25.load(INDEX / "bm25_standard")
        qvecs = []
        for i in range(0, len(queries), 64):
            r = httpx.post(f"{emb.url}/embed", timeout=120, json={
                "texts": [q["query"] for q in queries[i:i + 64]], "encoding": "base64"}).json()
            qvecs.append(np.frombuffer(base64.b64decode(r["vectors_b64"]), "<f4").reshape(-1, r["dim"]))
        qvecs = np.vstack(qvecs).astype(np.float32)
        sc, ids = flat.search(qvecs, 100)
        bodies = ordered_bodies(gw.url, queries)
        same_top10, same_set, ref_bodies = 0, 0, []
        for i, (q, b) in enumerate(zip(queries, bodies)):
            dense = list(zip(ids[i].tolist(), sc[i].tolist()))
            bi, bs = bm25.search(standard_tokens(q["query"]), 100)
            ref = [d for d, *_ in weighted_fusion(dense, list(zip(bi.tolist(), bs.tolist())), 0.5)[:10]]
            got = [r["doc"] for r in b["results"]]
            same_top10 += got == ref
            same_set += set(got) == set(ref)
            ref_bodies.append({"results": [{"doc": d} for d in ref]})
        out["correctness"] = {
            "queries": len(queries), "identical_top10_order": same_top10,
            "identical_top10_set": same_set,
            "ndcg@10_cluster": ndcg_by_type(queries, bodies),
            "ndcg@10_single_node": ndcg_by_type(queries, ref_bodies)}
        print("  ", out["correctness"])

        # ---------------------------------------------------------- 2. latency
        print("2. latency")
        texts = [q["query"] for q in queries]
        out["latency"] = {}
        for c in (1, 8):
            out["latency"][c] = summarize(run_queries(gw.url, texts[:400], c))
            s = out["latency"][c]
            print(f"   c={c}: {s['throughput_rps']} rps p50={s['client']['p50_ms']} "
                  f"p99={s['client']['p99_ms']} fanout p50={s['stages']['fanout_ms']['p50_ms']}")

        # ---------------------------------------------------------- 3. failover
        print("3a. kill replica 0a mid-load")
        reference = {q["query"]: [r["doc"] for r in b["results"]] for q, b in zip(queries, bodies)}

        async def kill_after(delay):
            await asyncio.sleep(delay)
            os.killpg(shards[(0, "a")].proc.pid, signal.SIGKILL)

        async def load_with_kill():
            task = asyncio.create_task(kill_after(1.0))
            res = await closed_loop(lambda c, p: gateway_req(c, p, gw.url), texts[:400], 4,
                                    warmup=False)
            await task
            return res
        res = asyncio.run(load_with_kill())
        summ = summarize(res)
        summ["results_identical_to_healthy_cluster"] = sum(
            [r["doc"] for r in b["results"]] == reference[b["query"]] for b in res["bodies"])
        fo = [l for l, b in zip(res["lat"], res["bodies"])
              if any(s.get("failed_over") for s in b["shards"].values())]
        summ["failed_over_latency"] = pct(fo)
        out["failover_kill"] = summ
        print(f"   max={summ['client_max_ms']} failover-lat={summ['failed_over_latency']} "
              f"errors={summ['errors']} partial={summ['partial']} failed_over={summ['failed_over']} "
              f"identical={summ['results_identical_to_healthy_cluster']}/{summ['requests']}")

        print("3b. replica 1a hangs (3 s > 1 s timeout)")
        time.sleep(5.5)          # let breakers from 3a close: start from a clean state
        httpx.post(f"{shards[(1, 'a')].url}/admin/fault", json={"mode": "slow", "delay_ms": 3000})
        # ~12 s of traffic: longer than the 5 s breaker cooldown, so the periodic
        # re-probe of the hung replica (and what it costs) is in the window
        res = run_queries(gw.url, (texts * 2)[:900], 4, warmup=False)
        summ = summarize(res)
        summ["results_identical_to_healthy_cluster"] = sum(
            [r["doc"] for r in b["results"]] == reference[b["query"]] for b in res["bodies"])
        out["failover_slow"] = summ
        httpx.post(f"{shards[(1, 'a')].url}/admin/fault", json={"mode": "ok"})
        print(f"   errors={summ['errors']} failed_over={summ['failed_over']} "
              f"p50={summ['client']['p50_ms']} p99={summ['client']['p99_ms']} "
              f"max={summ['client_max_ms']} over500={summ['requests_over_500ms']}")
        out["cluster_status_after_failover"] = httpx.get(f"{gw.url}/cluster").json()

        # ---------------------------------------------------------- 4. shard outage
        print("4. whole shard 0 down")
        os.killpg(shards[(0, "b")].proc.pid, signal.SIGKILL)
        time.sleep(0.5)
        bodies_partial = ordered_bodies(gw.url, queries)
        on_shard1 = set(np.load(ROOT / "v1" / "shard-1" / "doc_ids.npy").tolist())
        reachable = sum(q["relevant"][0] in on_shard1 for q in queries)
        out["shard_outage"] = {
            "partial_responses": sum(b["partial"] for b in bodies_partial),
            "queries": len(queries),
            "queries_whose_target_is_on_surviving_shard": reachable,
            "ndcg@10_degraded": ndcg_by_type(queries, bodies_partial),
            "ndcg@10_healthy": out["correctness"]["ndcg@10_cluster"],
            "require_all_status": httpx.get(f"{gw.url}/search", params={
                "q": "graph neural networks", "require_all": True}).status_code}
        print("  ", out["shard_outage"])

        # restart shard 0
        for k in ((0, "a"), (0, "b")):
            shards[k].start()
        time.sleep(5.5)          # let the circuit breakers close

        # ---------------------------------------------------------- 5. versioning
        print(f"5. version switch v1 -> v2 ({args.v2_kind}) under load")
        ensure_version("v2", args.v2_kind)
        early = httpx.post(f"{gw.url}/admin/version", json={"version": "v2"}).json()

        async def rollout():
            async with httpx.AsyncClient(timeout=300) as c:
                await asyncio.sleep(0.5)
                await asyncio.gather(*(c.post(f"{s.url}/admin/load", json={"version": "v2"})
                                       for s in shards.values()))
                return (await c.post(f"{gw.url}/admin/version", json={"version": "v2"})).json()

        async def load_during_rollout():
            rt = asyncio.create_task(rollout())
            stream = (texts * 4)[:1500]
            res = await closed_loop(lambda c, p: gateway_req(c, p, gw.url), stream, 4)
            return res, await rt
        res, switched = asyncio.run(load_during_rollout())
        mixed = sum(len({s.get("version", b["version"]) for s in b["shards"].values()} | {b["version"]}) > 1
                    for b in res["bodies"])
        versions = [b["version"] for b in res["bodies"]]
        bodies_v2 = ordered_bodies(gw.url, queries)
        out["versioning"] = {
            "switch_before_rollout": early, "switch_after_rollout": switched,
            "requests_during_rollout": len(res["bodies"]), "errors": res["errors"],
            "responses_v1": versions.count("v1"), "responses_v2": versions.count("v2"),
            "responses_with_mixed_versions": mixed,
            "ndcg@10_v2": ndcg_by_type(queries, bodies_v2),
            "manifest_v2_index": sharding.read_manifest(ROOT, "v2")["index"]}
        print("  ", {k: v for k, v in out["versioning"].items() if k != "switch_after_rollout"})
        out["gateway_metrics"] = httpx.get(f"{gw.url}/metrics").json()
    finally:
        for s in services:
            s.stop()

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
