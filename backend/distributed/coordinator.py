"""Query coordinator: fan out to every shard, fail over between replicas, merge.

Per query:
  1. embed + tokenise once (the shards never see text);
  2. send the same request to one replica of every shard, concurrently;
  3. for each shard, if the chosen replica errors, times out, or does not hold
     the pinned index version, retry on the next replica of that shard;
  4. merge each arm's per-shard top-k into a global top-k (merge.py), then run
     the same weighted fusion a single node runs.

Failure handling, in order of severity:
  * one replica down    -> transparent. The request is retried on a sibling
                           replica; results are identical. A per-replica circuit
                           breaker (ReplicaState) takes the bad replica out of
                           rotation for `cooldown_s`, then lets a single probe
                           request through before readmitting it.
  * every replica of a shard down
                        -> degrade, don't fail. The query is answered from the
                           shards that responded and the response says so
                           (`partial: true`, `missing_shards`). With hash
                           sharding the missing documents are a uniform random
                           1/N of the corpus, so recall drops by ~1/N rather than
                           a whole topic disappearing. A caller that needs
                           completeness can set `require_all=True` and get an
                           error instead.
  * a replica on the wrong index version
                        -> treated as unavailable for this query. The coordinator
                           pins one version and every shard request names it, so
                           a response can never mix documents from two builds.

Transport is injected, so the failure logic is unit-tested in-process
(eval/test_distributed.py) without sockets.
"""
from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

import numpy as np

from fusion import weighted_fusion

from .merge import merge_topk


class ShardError(Exception):
    pass


class Transport(Protocol):
    async def search(self, replica: str, payload: dict, timeout_s: float) -> dict: ...


class HttpTransport:
    def __init__(self, max_connections: int = 64):
        import httpx

        self._httpx = httpx
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections,
                                max_keepalive_connections=max_connections))

    async def search(self, replica: str, payload: dict, timeout_s: float) -> dict:
        try:
            r = await self.client.post(f"{replica}/search", json=payload, timeout=timeout_s)
        except self._httpx.TimeoutException as e:
            raise ShardError(f"timeout after {timeout_s*1000:.0f} ms") from e
        except self._httpx.HTTPError as e:
            raise ShardError(f"{type(e).__name__}") from e
        if r.status_code != 200:
            raise ShardError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    async def get(self, url: str, timeout_s: float = 2.0) -> dict:
        r = await self.client.get(url, timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    async def post(self, url: str, body: dict, timeout_s: float = 120.0) -> dict:
        r = await self.client.post(url, json=body, timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    async def aclose(self):
        await self.client.aclose()


@dataclass
class ReplicaState:
    """Circuit breaker per replica: closed -> open (cooldown) -> half-open -> closed.

    closed     no recent failure; takes its share of traffic
    open       failed within the last `cooldown_s`; skipped while any sibling is usable
    half-open  cooldown over but not yet proven healthy: exactly ONE request is let
               through as a probe. Without this state every request routed to the
               replica in the first `timeout_s` after the cooldown pays the full
               timeout (measured: 18 of 900 requests > 500 ms with one hung replica).
    """
    url: str
    down_until: float = 0.0
    failures: int = 0
    requests: int = 0
    last_error: str | None = None
    probing: bool = False

    def state(self, now: float) -> str:
        if self.failures == 0:
            return "closed"
        return "open" if now < self.down_until else "half-open"

    def healthy(self, now: float) -> bool:
        st = self.state(now)
        return st == "closed" or (st == "half-open" and not self.probing)


@dataclass
class Coordinator:
    shards: dict[int, list[str]]              # shard id -> replica base URLs
    version: str
    transport: Transport
    timeout_s: float = 1.0
    cooldown_s: float = 5.0
    alpha: float = 0.5
    k_each: int = 100
    _replicas: dict[int, list[ReplicaState]] = field(default_factory=dict)
    _rr: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self._replicas = {s: [ReplicaState(u) for u in urls] for s, urls in self.shards.items()}
        self._rr = {s: 0 for s in self.shards}

    # ---------------------------------------------------------- replica choice
    def _order(self, shard: int) -> list[ReplicaState]:
        """Healthy replicas first, round-robin among them; circuit-open ones last
        (still tried if everything else fails -- a stale breaker must not turn a
        recovered shard into a missing one)."""
        reps = self._replicas[shard]
        start = self._rr[shard] % len(reps)
        self._rr[shard] += 1
        rotated = reps[start:] + reps[:start]
        now = time.monotonic()
        return [r for r in rotated if r.healthy(now)] + [r for r in rotated if not r.healthy(now)]

    async def _query_shard(self, shard: int, payload: dict) -> dict:
        attempts = []
        for rep in self._order(shard):
            now = time.monotonic()
            if rep.state(now) == "half-open":
                if rep.probing:
                    continue                 # someone else is already probing it
                rep.probing = True
            t0 = time.perf_counter()
            rep.requests += 1
            try:
                out = await self.transport.search(rep.url, payload, self.timeout_s)
                if out.get("version") != payload["version"]:
                    raise ShardError(f"version mismatch: {out.get('version')}")
                rep.down_until, rep.failures, rep.probing = 0.0, 0, False
                out["_meta"] = {"replica": rep.url, "attempts": attempts + [rep.url],
                                "ms": (time.perf_counter() - t0) * 1000, "failed_over": bool(attempts)}
                return out
            except ShardError as e:
                rep.failures += 1
                rep.last_error = str(e)
                rep.down_until = time.monotonic() + self.cooldown_s
                rep.probing = False
                attempts.append(rep.url)
        raise ShardError(f"shard {shard}: all {len(attempts) or len(self._replicas[shard])} "
                         f"replicas failed or unavailable")

    # ---------------------------------------------------------- search
    async def search(self, vec: np.ndarray, tokens: list[str], top_k: int = 10,
                     require_all: bool = False) -> dict:
        # Read the pinned version exactly once. A switch_version() can land while
        # this request is fanned out; the request must be answered -- and labelled
        # -- with the version it was sent with, not the one pinned when it returns.
        version = self.version
        payload = {"vector_b64": base64.b64encode(np.asarray(vec, "<f4").tobytes()).decode(),
                   "tokens": tokens, "k": self.k_each, "version": version}
        t0 = time.perf_counter()
        shard_ids = sorted(self._replicas)
        outs = await asyncio.gather(*(self._query_shard(s, payload) for s in shard_ids),
                                    return_exceptions=True)
        t1 = time.perf_counter()

        ok, missing, shard_info = [], [], {}
        for s, o in zip(shard_ids, outs):
            if isinstance(o, Exception):
                missing.append(s)
                shard_info[s] = {"status": "unavailable", "error": str(o)}
            else:
                ok.append(o)
                shard_info[s] = {"status": "ok", "version": o.get("version"),
                                 **o["_meta"], **o.get("timings", {})}
        if not ok or (missing and require_all):
            raise ShardError(f"shards unavailable: {missing}")

        dense = merge_topk([[tuple(h) for h in o["dense"]] for o in ok], self.k_each)
        sparse = merge_topk([[tuple(h) for h in o["bm25"]] for o in ok], self.k_each)
        t2 = time.perf_counter()
        fused = weighted_fusion(dense, sparse, self.alpha)[:top_k]
        t3 = time.perf_counter()
        return {
            "hits": fused, "dense": dense, "sparse": sparse,
            "partial": bool(missing), "missing_shards": missing, "version": version,
            "shards": shard_info,
            "timings": {"fanout_ms": (t1 - t0) * 1000, "merge_ms": (t2 - t1) * 1000,
                        "fusion_ms": (t3 - t2) * 1000},
        }

    # ---------------------------------------------------------- ops
    def status(self) -> dict:
        now = time.monotonic()
        return {"version": self.version, "shards": {
            s: [{"url": r.url, "state": r.state(now), "healthy": r.healthy(now), "failures": r.failures,
                 "requests": r.requests, "last_error": r.last_error} for r in reps]
            for s, reps in self._replicas.items()}}

    async def switch_version(self, version: str,
                             health: Callable[[str], Awaitable[dict]]) -> dict:
        """Pin a new index version, but only if every shard has a replica serving it.

        Replicas that do not hold it yet stay in the pool and simply 409 (and fail
        over) until they load it.
        """
        ready = {}
        for s, reps in self._replicas.items():
            have = []
            for r in reps:
                try:
                    if version in (await health(r.url)).get("versions", {}):
                        have.append(r.url)
                except Exception:
                    pass
            ready[s] = have
        not_ready = [s for s, h in ready.items() if not h]
        if not_ready:
            return {"switched": False, "version": self.version, "not_ready_shards": not_ready}
        old, self.version = self.version, version
        return {"switched": True, "from": old, "version": version, "replicas_ready": ready}
