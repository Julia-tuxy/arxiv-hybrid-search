"""Tests for the serving layer: micro-batcher, embedding cache, latency recorder.

    python -m pytest test_serving.py -q
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

import _path  # noqa: F401
from serving.batcher import MicroBatcher
from serving.cache import EmbeddingCache, cache_key
from serving.latency import LatencyRecorder, StageTimer


# ------------------------------------------------------------------ batcher
def _run(coro):
    return asyncio.run(coro)


def test_batcher_returns_each_caller_its_own_result():
    async def go():
        b = MicroBatcher(lambda xs: [x * 10 for x in xs], max_batch=8, max_wait_ms=20)
        out = await asyncio.gather(*(b.submit(i) for i in range(20)))
        await b.stop()
        return [r for r, _ in out], b
    results, b = _run(go())
    assert results == [i * 10 for i in range(20)]
    assert b.items_processed == 20
    assert max(b.batch_sizes) <= 8


def test_concurrent_requests_are_batched_together():
    calls = []

    def fn(xs):
        calls.append(len(xs))
        time.sleep(0.01)
        return xs

    async def go():
        b = MicroBatcher(fn, max_batch=32, max_wait_ms=20)
        out = await asyncio.gather(*(b.submit(i) for i in range(32)))
        await b.stop()
        return out
    out = _run(go())
    assert len(calls) <= 2                       # 32 concurrent -> 1-2 model calls
    assert all(info["batch_size"] >= 16 for _, info in out)


def test_lone_request_waits_at_most_max_wait():
    async def go():
        b = MicroBatcher(lambda xs: xs, max_batch=32, max_wait_ms=30)
        t0 = time.perf_counter()
        _, info = await b.submit("x")
        dt = (time.perf_counter() - t0) * 1000
        await b.stop()
        return dt, info
    dt, info = _run(go())
    assert info["batch_size"] == 1
    assert 25 <= dt < 200


def test_full_batch_is_flushed_without_waiting():
    async def go():
        b = MicroBatcher(lambda xs: xs, max_batch=4, max_wait_ms=5000)
        t0 = time.perf_counter()
        await asyncio.gather(*(b.submit(i) for i in range(4)))
        dt = time.perf_counter() - t0
        await b.stop()
        return dt
    assert _run(go()) < 1.0                      # not the 5 s wait


def test_batch_failure_propagates_and_worker_survives():
    def fn(xs):
        if "boom" in xs:
            raise ValueError("bad input")
        return xs

    async def go():
        b = MicroBatcher(fn, max_batch=1, max_wait_ms=1)
        with pytest.raises(ValueError):
            await b.submit("boom")
        ok, _ = await b.submit("fine")           # worker still alive
        await b.stop()
        return ok
    assert _run(go()) == "fine"


# ------------------------------------------------------------------ cache
def test_cache_lru_eviction_and_stats():
    c = EmbeddingCache(capacity=2)
    c.put("a", np.ones(3)); c.put("b", np.zeros(3))
    assert c.get("a") is not None                # a is now most recent
    c.put("c", np.ones(3))                       # evicts b
    assert c.get("b") is None and c.get("c") is not None
    s = c.stats()
    assert s["size"] == 2 and s["hits"] == 2 and s["misses"] == 1


def test_cache_key_only_collapses_whitespace():
    assert cache_key("  graph   neural\tnets ") == "graph neural nets"
    assert cache_key("GNN") != cache_key("gnn")  # case can matter to a cased model


def test_cached_vectors_are_read_only_and_disabled_cache_is_inert():
    c = EmbeddingCache(4)
    c.put("q", np.arange(3, dtype=np.float32))
    with pytest.raises(ValueError):
        c.get("q")[0] = 99
    off = EmbeddingCache(0)
    off.put("q", np.ones(2))
    assert off.get("q") is None and off.stats()["misses"] == 0


def test_cache_is_thread_safe():
    c = EmbeddingCache(capacity=50)

    def worker(seed):
        rng = np.random.default_rng(seed)
        for _ in range(2000):
            k = f"q{rng.integers(100)}"
            if c.get(k) is None:
                c.put(k, np.ones(4))
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
    s = c.stats()
    assert s["size"] <= 50 and s["hits"] + s["misses"] == 16000


# ------------------------------------------------------------------ latency
def test_latency_percentiles():
    r = LatencyRecorder()
    for v in range(1, 101):
        r.record("embed", float(v))
    r.incr("errors", 2)
    snap = r.snapshot()
    e = snap["stages"]["embed"]
    assert e["n"] == 100 and abs(e["p50_ms"] - 50.5) < 1e-6 and e["p99_ms"] > 98
    assert snap["counters"]["errors"] == 2


def test_stage_timer_header():
    t = StageTimer()
    with t.stage("embed"):
        time.sleep(0.005)
    t.add("ann", 1.5)
    assert t.timings["embed"] >= 4 and t.timings["ann"] == 1.5
    assert t.server_timing_header().startswith("embed;dur=")
