"""Concurrent load test for the embedding service and the search API.

Starts the services itself (separate processes, real HTTP), drives them with a
closed-loop async client at several concurrency levels, and records request-level
latency on the client side plus the per-stage breakdown each server reports.

Experiments
  embed_batching   embedding service alone, cache off: micro-batching on vs off
  embed_cache      embedding service, cache on, Zipf-distributed repeat traffic
  search_e2e       search API -> embedding service: total latency split into
                   embed / ann / bm25 / fusion per request

    cd backend && python loadtest/run_load.py            # -> eval/results/load_test.json
    python loadtest/plot_load.py                         # -> eval/results/load_test.png
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
EVAL = BACKEND / "eval"
PY = sys.executable


# ------------------------------------------------------------------ processes
class Service:
    def __init__(self, module: str, port: int, env: dict, name: str):
        self.module, self.port, self.env, self.name = module, port, env, name
        self.proc: subprocess.Popen | None = None
        self.log = EVAL / "results" / f".{name}.log"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout_s: float = 900) -> "Service":
        env = {**os.environ, **{k: str(v) for k, v in self.env.items()}}
        self.proc = subprocess.Popen(
            [PY, "-m", "uvicorn", f"{self.module}:app", "--port", str(self.port),
             "--log-level", "warning"],
            cwd=BACKEND, env=env, stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
            start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited; see {self.log}")
            try:
                if httpx.get(f"{self.url}/health", timeout=1).json().get("status") == "ok":
                    print(f"  {self.name} up on :{self.port} ({time.time()-t0:.0f}s)")
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"{self.name} did not become healthy")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# ------------------------------------------------------------------ load generator
async def closed_loop(make_request, payloads: list, concurrency: int,
                      warmup: bool = True, after_warmup=None) -> dict:
    """`concurrency` workers each send the next payload as soon as their previous
    request finishes. Returns client-side latencies and per-request server bodies.

    Connections are opened in an untimed warm-up first. Without it, a c=64 run
    starts with 64 simultaneous TCP connects; the ones that lose the race wait for
    a SYN retransmit (1 s, then 2 s), and the timed results show those seconds as
    if they were server latency.
    """
    lat, bodies, errors = [], [], 0
    it = iter(enumerate(payloads))
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=60) as client:
        if warmup:
            sem = asyncio.Semaphore(8)               # open connections a few at a time

            async def warm(p):
                async with sem:
                    try:
                        await make_request(client, p)
                    except httpx.HTTPError:
                        pass
            await asyncio.gather(*(warm(payloads[i % len(payloads)]) for i in range(concurrency)))
            # then hold them all open at once so the pool keeps `concurrency` sockets
            await asyncio.gather(*(make_request(client, payloads[i % len(payloads)])
                                   for i in range(concurrency)), return_exceptions=True)
        if after_warmup is not None:
            await after_warmup(client)

        async def worker():
            nonlocal errors
            for _, p in it:
                t0 = time.perf_counter()
                try:
                    body = await make_request(client, p)
                    lat.append((time.perf_counter() - t0) * 1000)
                    bodies.append(body)
                except httpx.HTTPError:
                    errors += 1
        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    return {"lat": lat, "bodies": bodies, "errors": errors, "wall_s": wall}


# One asyncio + httpx client process saturates at ~100 req/s around 64 open
# connections: measured against the same server, 1 process x 64 connections got
# 101 rps at p50 430 ms while the server reported p50 12 ms per request, and
# 4 processes x 16 got 489 rps. Above PER_PROC connections the load is therefore
# spread over several client processes that start their timed phase together.
PER_PROC = 16


def _proc_main(kind: str, url: str, payloads: list, concurrency: int, start_at: float) -> dict:
    req = {"embed": embed_req, "search": search_req}[kind]

    async def go():
        async def hold(_client):
            await asyncio.sleep(max(0.0, start_at - time.time()))
        return await closed_loop(lambda c, p: req(c, p, url), payloads, concurrency,
                                 after_warmup=hold)
    return asyncio.run(go())


def drive(kind: str, url: str, payloads: list, concurrency: int, before_start=None) -> dict:
    """Closed-loop load at `concurrency`, split over ceil(c / PER_PROC) processes.

    `before_start(url)` runs after every process has warmed up and just before the
    timed phase (used to reset server-side counters)."""
    import concurrent.futures as cf
    import multiprocessing as mp

    n_proc = max(1, -(-concurrency // PER_PROC))
    if n_proc == 1:
        req = {"embed": embed_req, "search": search_req}[kind]

        async def hook(_client):
            if before_start:
                before_start(url)
        return asyncio.run(closed_loop(lambda c, p: req(c, p, url), payloads, concurrency,
                                       after_warmup=hook))
    start_at = time.time() + 8.0          # process spawn + imports + warm-up budget
    per = [concurrency // n_proc + (i < concurrency % n_proc) for i in range(n_proc)]
    with cf.ProcessPoolExecutor(n_proc, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_proc_main, kind, url, payloads[i::n_proc], per[i], start_at)
                for i in range(n_proc)]
        time.sleep(max(0.0, start_at - time.time() - 0.3))
        if before_start:
            before_start(url)
        parts = [f.result() for f in futs]
    return {"lat": [x for p in parts for x in p["lat"]],
            "bodies": [b for p in parts for b in p["bodies"]],
            "errors": sum(p["errors"] for p in parts),
            "wall_s": max(p["wall_s"] for p in parts), "client_processes": n_proc}


def pct(xs) -> dict:
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if len(a) == 0:
        return {}
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return {"p50_ms": round(float(p50), 2), "p95_ms": round(float(p95), 2),
            "p99_ms": round(float(p99), 2), "mean_ms": round(float(a.mean()), 2)}


def load_queries() -> list[str]:
    out = []
    for p in sorted((EVAL / "queries").glob("*_queries.jsonl")):
        out += [json.loads(l)["query"] for l in p.read_text().splitlines() if l.strip()]
    return out


def zipf_stream(items: list[str], n: int, s: float = 1.1, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, len(items) + 1) ** s
    order = rng.permutation(len(items))
    return [items[order[i]] for i in rng.choice(len(items), n, p=w / w.sum())]


# ------------------------------------------------------------------ experiments
async def embed_req(client, text, url):
    r = await client.post(f"{url}/embed", json={"texts": [text], "encoding": "base64"})
    r.raise_for_status()
    return r.json()


def run_embed_batching(queries, levels, n_req, port) -> dict:
    out = {}
    for mode, env in [("unbatched", {"EMBED_BATCHING": 0}),
                      ("batched_wait5ms", {"EMBED_BATCHING": 1, "EMBED_MAX_BATCH": 32,
                                           "EMBED_MAX_WAIT_MS": 5}),
                      # greedy: no timer; a batch is whatever queued while the
                      # previous one was running
                      ("batched_greedy", {"EMBED_BATCHING": 1, "EMBED_MAX_BATCH": 32,
                                          "EMBED_MAX_WAIT_MS": 0})]:
        out[mode] = {}
        with Service("embedding_service", port, {**env, "EMBED_CACHE_SIZE": 0}, f"embed-{mode}") as svc:
            for c in levels:
                payloads = [queries[i % len(queries)] for i in range(max(n_req, 16 * c))]

                res = drive("embed", svc.url, payloads, c,
                            before_start=lambda u: httpx.post(f"{u}/metrics/reset"))
                m = httpx.get(f"{svc.url}/metrics").json()
                row = {"requests": len(res["lat"]), "errors": res["errors"],
                       "client_processes": res.get("client_processes", 1),
                       "throughput_rps": round(len(res["lat"]) / res["wall_s"], 1),
                       "client": pct(res["lat"]),
                       "server_inference": m["latency"]["stages"].get("inference", {}),
                       "server_queue_wait": m["latency"]["stages"].get("queue_wait", {}),
                       "mean_batch_size": (m["batcher"] or {}).get("mean_batch_size", 1.0)}
                out[mode][c] = row
                print(f"  {mode:16s} c={c:3d} {row['throughput_rps']:7.1f} rps  "
                      f"p50={row['client']['p50_ms']:7.2f} p95={row['client']['p95_ms']:7.2f} "
                      f"p99={row['client']['p99_ms']:7.2f}  batch={row['mean_batch_size']}")
    return out


def run_embed_cache(queries, n_req, concurrency, port) -> dict:
    stream = zipf_stream(queries, n_req)
    env = {"EMBED_BATCHING": 1, "EMBED_MAX_WAIT_MS": 0, "EMBED_CACHE_SIZE": 10_000}
    with Service("embedding_service", port, env, "embed-cache") as svc:
        # the warm-up must not pre-fill the cache
        res = drive("embed", svc.url, stream, concurrency, before_start=lambda u: httpx.post(
            f"{u}/metrics/reset", params={"clear_cache": True}))
        hits = [l for l, b in zip(res["lat"], res["bodies"]) if b["cached"][0]]
        miss = [l for l, b in zip(res["lat"], res["bodies"]) if not b["cached"][0]]
        m = httpx.get(f"{svc.url}/metrics").json()
    out = {"requests": len(res["lat"]), "distinct_queries": len(set(stream)),
           "zipf_s": 1.1, "concurrency": concurrency,
           "cache": m["cache"], "throughput_rps": round(len(res["lat"]) / res["wall_s"], 1),
           "client_all": pct(res["lat"]), "client_hit": pct(hits), "client_miss": pct(miss)}
    print(f"  cache: hit_rate={m['cache']['hit_rate']} hit p50={out['client_hit'].get('p50_ms')} "
          f"miss p50={out['client_miss'].get('p50_ms')}")
    return out


async def search_req(client, q, url):
    r = await client.get(f"{url}/search", params={"q": q, "top_k": 10})
    r.raise_for_status()
    return r.json()


def run_search_e2e(queries, levels, n_req, embed_port, search_port, vector_index,
                   cache: bool = False) -> dict:
    """cache=False: unique-ish traffic, every embed is real model work.
    cache=True: Zipf-distributed repeat traffic with the embedding cache on."""
    out = {}
    emb_env = {"EMBED_BATCHING": 1, "EMBED_MAX_WAIT_MS": 0,
               "EMBED_CACHE_SIZE": 10_000 if cache else 0}
    with Service("embedding_service", embed_port, emb_env, "embed-e2e") as emb:
        s_env = {"EMBEDDING_URL": emb.url, "VECTOR_INDEX": vector_index}
        with Service("main", search_port, s_env, "search-e2e") as api:
            for c in levels:
                n = max(n_req, 16 * c)
                payloads = (zipf_stream(queries, n, seed=c) if cache
                            else [queries[i % len(queries)] for i in range(n)])
                res = drive("search", api.url, payloads, c)
                stages = {}
                for b in res["bodies"]:
                    for k, v in (b.get("timings") or {}).items():
                        stages.setdefault(k, []).append(v)
                row = {"requests": len(res["lat"]), "errors": res["errors"],
                       "client_processes": res.get("client_processes", 1),
                       "throughput_rps": round(len(res["lat"]) / res["wall_s"], 1),
                       "client": pct(res["lat"]),
                       "stages": {k: pct(v) for k, v in stages.items()}}
                out[c] = row
                s = row["stages"]
                print(f"  search c={c:3d} {row['throughput_rps']:6.1f} rps "
                      f"client p50={row['client']['p50_ms']:.1f} p99={row['client']['p99_ms']:.1f} | "
                      f"embed p50={s['embed_ms']['p50_ms']:.1f} ann p50={s['ann_ms']['p50_ms']:.2f} "
                      f"bm25 p50={s['bm25_ms']['p50_ms']:.2f} fusion p50={s['fusion_ms']['p50_ms']:.2f}")
            out["server_metrics"] = httpx.get(f"{api.url}/metrics").json()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--experiments", nargs="+",
                    default=["embed_batching", "embed_cache", "search_e2e"])
    ap.add_argument("--vector-index", default=os.environ.get("VECTOR_INDEX", "flat"))
    ap.add_argument("--e2e-cache", action="store_true",
                    help="Zipf repeat traffic with the embedding cache on")
    ap.add_argument("--e2e-key", default="search_e2e",
                    help="result key, so runs with different VECTOR_INDEX can coexist")
    ap.add_argument("--out", default=str(EVAL / "results" / "load_test.json"))
    args = ap.parse_args()

    queries = load_queries()
    path = Path(args.out)
    results = json.loads(path.read_text()) if path.exists() else {}
    results.setdefault("config", {}).update({"levels": args.levels, "requests_per_level": args.requests,
                         "cpu_count": os.cpu_count(), "platform": sys.platform,
                         })
    results["config"][f"{args.e2e_key}_vector_index"] = args.vector_index
    results["config"][f"{args.e2e_key}_embed_cache"] = args.e2e_cache
    if "embed_batching" in args.experiments:
        print("embed_batching")
        results["embed_batching"] = run_embed_batching(queries, args.levels, args.requests, 8101)
    if "embed_cache" in args.experiments:
        print("embed_cache")
        results["embed_cache"] = run_embed_cache(queries, 3000, 16, 8102)
    if "search_e2e" in args.experiments:
        print("search_e2e")
        results[args.e2e_key] = run_search_e2e(queries, args.levels, args.requests,
                                               8103, 8104, args.vector_index, cache=args.e2e_cache)
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
