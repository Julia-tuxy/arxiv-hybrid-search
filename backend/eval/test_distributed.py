"""Tests for sharding, merging and coordinator failover (backend/distributed/).

    python -m pytest test_distributed.py -q

The coordinator is driven through an in-process fake transport backed by real
shard indexes built from synthetic data, so the merge is checked against a real
single-node search and failures are injected deterministically.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

import _path  # noqa: F401
from bm25_index import SparseBM25
from distributed import sharding
from distributed.coordinator import Coordinator, ShardError
from distributed.merge import merge_topk
from fusion import weighted_fusion

N, D, K = 3000, 32, 50


# ------------------------------------------------------------------ fixtures
def _corpus(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, D)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    vocab = [f"t{i}" for i in range(300)]
    toks = [[vocab[j] for j in rng.zipf(1.3, rng.integers(20, 60)) % 300] for _ in range(N)]
    keys = [f"{2000 + i // 100}.{i:05d}" for i in range(N)]
    return x, toks, keys


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    root = tmp_path_factory.mktemp("shards")
    x, toks, keys = _corpus()
    bm25 = SparseBM25.build(toks)
    sharding.build_version(root, "v1", x, keys, bm25, n_shards=2)
    from distributed.shard_server import ShardIndex
    shards = {s: ShardIndex(root, "v1", s) for s in (0, 1)}
    return root, x, toks, keys, bm25, shards


class FakeTransport:
    """replica url "shard{s}-{r}" -> local ShardIndex; failures injectable per url."""

    def __init__(self, shards):
        self.shards = shards
        self.down: set[str] = set()
        self.calls: list[str] = []

    async def search(self, replica, payload, timeout_s):
        self.calls.append(replica)
        if replica in self.down:
            raise ShardError("connection refused")
        s = int(replica.split("-")[0].removeprefix("shard"))
        idx = self.shards[s]
        if payload["version"] != idx.version:
            raise ShardError("HTTP 409")
        import base64
        vec = np.frombuffer(base64.b64decode(payload["vector_b64"]), dtype="<f4")
        out = idx.search(vec.astype(np.float32), payload["tokens"], payload["k"])
        out.update(shard=s, replica=replica, version=idx.version)
        return out


def _coord(transport, **kw):
    return Coordinator(shards={0: ["shard0-a", "shard0-b"], 1: ["shard1-a", "shard1-b"]},
                       version="v1", transport=transport, k_each=K, **kw)


def _single_node(x, bm25, q, tokens):
    s = x @ q
    top = np.argsort(-s)[:K]
    dense = [(int(i), float(s[i])) for i in top]
    ids, sc = bm25.search(tokens, K)
    return dense, [(int(i), float(v)) for i, v in zip(ids, sc)]


# ------------------------------------------------------------------ sharding
def test_assignment_is_stable_and_balanced():
    keys = [f"{2000 + i // 1000}.{i:05d}" for i in range(20000)]
    a = sharding.assign(keys, 2)
    assert (a == sharding.assign(keys, 2)).all()                 # deterministic
    assert abs(a.mean() - 0.5) < 0.02                             # ~balanced
    assert sharding.shard_of(keys[7], 2) == a[7]


def test_manifest_versioning_and_checksums(cluster):
    root, *_ = cluster
    m = sharding.read_manifest(root, "v1")
    assert m["n_shards"] == 2 and sum(s["n_docs"] for s in m["shards"]) == N
    assert sharding.current_version(root) == "v1"
    with pytest.raises(FileExistsError):                         # versions are immutable
        x, toks, keys = _corpus()
        sharding.build_version(root, "v1", x, keys, SparseBM25.build(toks), 2)
    # corrupt a file -> verify refuses to serve it
    f = root / "v1" / "shard-1" / "doc_ids.npy"
    good = f.read_bytes()
    f.write_bytes(good[:-8] + b"\0" * 8)
    with pytest.raises(ValueError, match="checksum"):
        sharding.verify(root, "v1", 1)
    f.write_bytes(good)
    sharding.verify(root, "v1", 1)


def test_shards_partition_the_corpus(cluster):
    _, _, _, _, _, shards = cluster
    ids = np.concatenate([s.doc_ids for s in shards.values()])
    assert sorted(ids.tolist()) == list(range(N))


# ------------------------------------------------------------------ merging
def test_merge_topk_basic():
    a = [(1, 0.9), (3, 0.5), (5, 0.1)]
    b = [(2, 0.8), (4, 0.5), (6, 0.0)]
    assert merge_topk([a, b], 4) == [(1, 0.9), (2, 0.8), (3, 0.5), (4, 0.5)]
    assert merge_topk([a, []], 10) == a
    assert merge_topk([], 3) == []


def test_sharded_search_equals_single_node(cluster):
    _, x, toks, _, bm25, shards = cluster
    coord = _coord(FakeTransport(shards))
    rng = np.random.default_rng(5)
    for qi in rng.choice(N, 25, replace=False):
        q = x[qi] + 0.3 * rng.standard_normal(D).astype(np.float32)
        q /= np.linalg.norm(q)
        tokens = toks[qi][:4]
        out = asyncio.run(coord.search(q, tokens, top_k=10))
        dense, sparse = _single_node(x, bm25, q, tokens)
        assert [d for d, _ in out["dense"]] == [d for d, _ in dense]
        assert np.allclose([s for _, s in out["sparse"]], [s for _, s in sparse], atol=1e-5)
        want = weighted_fusion(dense, sparse, 0.5)[:10]
        assert [h[0] for h in out["hits"]] == [h[0] for h in want]
        assert not out["partial"]


# ------------------------------------------------------------------ failures
def _query(cluster):
    _, x, toks, *_ = cluster
    return x[11], toks[11][:3]


def test_replica_failover_is_transparent(cluster):
    shards = cluster[5]
    t = FakeTransport(shards)
    coord = _coord(t)
    q, tok = _query(cluster)
    want = asyncio.run(coord.search(q, tok))["hits"]
    t.down = {"shard0-a"}                                         # one replica of shard 0
    for _ in range(4):
        out = asyncio.run(coord.search(q, tok))
        assert out["hits"] == want and not out["partial"]
    assert coord.status()["shards"][0][0]["healthy"] is False     # circuit opened
    # after the breaker opens, the dead replica is no longer tried first
    t.calls.clear()
    asyncio.run(coord.search(q, tok))
    assert t.calls.count("shard0-a") == 0


def test_whole_shard_down_returns_partial_results(cluster):
    shards = cluster[5]
    t = FakeTransport(shards)
    t.down = {"shard1-a", "shard1-b"}
    coord = _coord(t)
    q, tok = _query(cluster)
    out = asyncio.run(coord.search(q, tok))
    assert out["partial"] and out["missing_shards"] == [1]
    on_shard1 = set(shards[1].doc_ids.tolist())
    assert not any(doc in on_shard1 for doc, *_ in out["hits"])
    with pytest.raises(ShardError):
        asyncio.run(coord.search(q, tok, require_all=True))
    t.down |= {"shard0-a", "shard0-b"}
    with pytest.raises(ShardError):                               # nothing left to answer from
        asyncio.run(coord.search(q, tok))


def test_recovered_replica_is_used_again(cluster):
    shards = cluster[5]
    t = FakeTransport(shards)
    coord = _coord(t, cooldown_s=0.0)
    q, tok = _query(cluster)
    t.down = {"shard0-a"}
    asyncio.run(coord.search(q, tok))
    t.down = set()
    t.calls.clear()
    for _ in range(4):
        asyncio.run(coord.search(q, tok))
    assert "shard0-a" in t.calls                                  # back in rotation


def test_version_pin_never_mixes_builds(cluster):
    shards = cluster[5]
    t = FakeTransport(shards)
    coord = _coord(t)
    coord.version = "v2"                                          # no shard has v2
    q, tok = _query(cluster)
    with pytest.raises(ShardError):
        asyncio.run(coord.search(q, tok))

    async def health(url):
        return {"versions": {"v1": 1}}
    res = asyncio.run(coord.switch_version("v9", health))
    assert res["switched"] is False and coord.version == "v2"
    res = asyncio.run(coord.switch_version("v1", health))
    assert res["switched"] is True and coord.version == "v1"


def test_version_switch_during_fanout_keeps_the_request_version(cluster):
    """Regression: the response was labelled with the version pinned when it
    returned, not the one its shard requests were sent with."""
    shards = cluster[5]

    class SwitchMidFlight(FakeTransport):
        async def search(self, replica, payload, timeout_s):
            coord.version = "v2"                    # a switch lands mid-request
            return await super().search(replica, payload, timeout_s)

    coord = _coord(SwitchMidFlight(shards))
    q, tok = _query(cluster)
    out = asyncio.run(coord.search(q, tok))
    assert out["version"] == "v1"
    assert {s["version"] for s in out["shards"].values()} == {"v1"}


def test_half_open_breaker_sends_a_single_probe(cluster):
    """After the cooldown only one request may test a recovering replica; the
    rest go to the healthy sibling instead of all paying the timeout."""
    shards = cluster[5]

    class Hung(FakeTransport):
        def __init__(self, shards):
            super().__init__(shards)
            self.hung_calls = 0

        async def search(self, replica, payload, timeout_s):
            if replica == "shard1-a":
                self.hung_calls += 1
                await asyncio.sleep(0.05)               # "times out"
                raise ShardError("timeout")
            return await super().search(replica, payload, timeout_s)

    t = Hung(shards)
    coord = _coord(t, cooldown_s=0.0)                    # breaker half-opens immediately
    q, tok = _query(cluster)

    async def burst():
        return await asyncio.gather(*(coord.search(q, tok) for _ in range(20)))
    asyncio.run(coord.search(q, tok))                   # first failure opens the breaker
    t.hung_calls = 0
    outs = asyncio.run(burst())
    assert all(not o["partial"] for o in outs)
    assert t.hung_calls == 1                            # one probe, not ~10
    assert coord.status()["shards"][1][0]["state"] in ("open", "half-open")
