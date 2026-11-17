# Distributed search: two shards, two replicas each

A minimal sharded deployment of the hybrid search. Its goal is not a full
distributed vector database. It is the smallest system that has to answer the
questions a real one does:

* **Shard assignment**: which shard owns a document?
* **Fan-out and merge**: how do per-shard results become one ranking without
  losing accuracy?
* **Failure**: what does a query do when a shard server is slow, dead, or
  entirely gone?
* **Versioning**: how is a new index rolled out under traffic without any
  response mixing two builds?

```
                         ┌──────────── gateway (coordinator) ────────────┐
  GET /search ─────────► │ embed once ─► fan out ─► merge ─► fusion ─► top-k │
                         └───────┬───────────────────────────┬───────────┘
                     shard 0     │                           │     shard 1
               ┌─────────────────┴──────┐        ┌───────────┴────────────┐
               │ replica 0a   replica 0b │        │ replica 1a   replica 1b │
               │ dense + BM25 over ~50K  │        │ dense + BM25 over ~50K  │
               └─────────────────────────┘        └─────────────────────────┘
       embedding service ◄── gateway (vectors and tokens go to the shards, never text)
```

| file | role |
|---|---|
| `sharding.py` | assignment rule, versioned shard builds, manifests, checksums |
| `build_shards.py` | CLI: `python -m distributed.build_shards --version v1 --shards 2` |
| `shard_server.py` | serves one shard; can hold several index versions at once |
| `merge.py` | exact k-way merge of per-shard top-k lists |
| `coordinator.py` | fan-out, replica failover, circuit breaker, partial results, version pin |
| `gateway.py` | the coordinator as an HTTP API: `/search`, `/cluster`, `/admin/version`, `/metrics` |
| `run_cluster_demo.py` | launches 2 × 2 shard servers + embedding service + gateway and measures everything below |

```bash
cd backend
python -m distributed.build_shards --version v1 --shards 2 --kind flat
python -m distributed.run_cluster_demo           # -> eval/results/cluster.json  (~4 min)
python -m pytest eval/test_distributed.py -q     # 11 tests, in-process, no sockets
```

## Shard assignment

A document is on shard `crc32(arxiv_id) mod N`. The split is 50,228 / 49,772.

* The hash uses the **stable external id**, not the row number. A document's
  shard does not change when the corpus is re-sampled or re-ordered, and a new
  document's shard is known without a lookup table.
* **Hashing, not ranges or categories.** Sharding by category would put all of
  hep-th on one shard. That shard would hold most of the top results for
  physics queries and become the hot spot, and losing it would remove a whole
  field. With hashing, each shard holds a uniform random sample. Load is even,
  and an outage costs a uniform 1/N of recall (measured below).
* Modulo hashing moves ~N/(N+1) of the documents when N changes. That is
  acceptable here because every index version is a full offline rebuild.
  Consistent hashing would matter if shards were rebalanced online.
* **BM25 has to be sharded with global statistics.** Each shard holds rows of
  the weight matrix built over the *whole* corpus, so IDF and average document
  length are corpus-wide. If each shard computed IDF over its own half, the
  same term would weigh differently on each shard, and BM25 scores from
  different shards could not be merged.

## Fan-out and merge

The gateway embeds and tokenises the query **once**, then sends the vector and
the tokens to one replica of every shard concurrently. Each shard returns its
top 100 for each arm: dense (cosine) and BM25.

The merge is **exact, not approximate**. A document's dense score depends only
on the query and that document, and with global BM25 statistics so does its
BM25 score. Neither depends on which shard holds it. If a document is in the
global top-k, at most k−1 documents outscore it anywhere, so at most k−1 do on
its own shard, so it is in that shard's top-k. The union of per-shard top-k
lists therefore contains the global top-k, and a k-way heap merge recovers it.
Two details are required for this to hold:

* every shard returns **k**, not k/N. Asking each shard for k/N silently drops
  results whenever the true top-k is unevenly spread, which for k=100 over two
  shards is almost always;
* **fusion runs after the merge.** Min-max normalisation depends on the whole
  list, so normalising per shard and then merging would compare scores on
  different scales.

**Verified:** all 740 evaluation queries return the **identical top-10, in the
same order**, from the 2-shard cluster and from a single node over the
unsharded index. nDCG@10 is identical (title 0.9906 / paraphrased 0.9461 /
keyword 0.9285).

| | p50 | p99 | throughput |
|---|---|---|---|
| single node, c=1 (eval/README.md 6.2) | 16.1 ms | 17.9 ms | 62 rps |
| 2-shard cluster via gateway, c=1 | 20.5 ms | 22.2 ms | 49 rps |
| 2-shard cluster via gateway, c=8 | 100 ms | 133 ms | 80 rps |

At c=1 the gateway's time splits into embed 8.6 ms, fan-out 8.2 ms (of which
the slower shard's ANN search is 3.5 ms), merge 0.2 ms and fusion 0.14 ms. The
~4 ms of overhead over a single node is the extra network hop plus JSON
encoding of 2 arms × 100 hits per shard. Binary payloads or a smaller per-arm
k would shrink it.

**What this setup cannot show is scale-out.** All seven processes (gateway,
embedding service, four shard servers, load generator) share one 8-core laptop.
Per-shard search over 50K vectors measured 3.5 ms, no faster than 100K in a
single process. So the shards are contending for the same cores, not working
in parallel on separate machines. The cluster demonstrates correctness,
overhead and failure behaviour. Throughput scaling needs shards on separate
hosts.

## When a shard server is unavailable

Each shard has two replicas. The coordinator tries one replica (round-robin
over healthy replicas) and on an error, a timeout (1 s), or a version mismatch
retries the other one. Each replica sits behind a **circuit breaker**:

```
closed ──failure──► open (cooldown 5 s, skipped) ──cooldown over──► half-open
   ▲                                                                   │
   └─────────────── probe succeeds ◄── exactly ONE request is let through
                                       (probe fails ──► open again)
```

| scenario (measured under load, 4 concurrent clients) | errors | results identical to healthy cluster | tail cost |
|---|---|---|---|
| replica 0a **SIGKILLed** mid-load (400 requests) | **0** | **400 / 400** | 1 request failed over (+17 ms); max 74 ms |
| replica 1a **hangs** 3 s per request, past the 1 s timeout (900 requests, ~12 s) | **0** | **900 / 900** | 6 requests > 500 ms; p99 95 ms |
| **both** replicas of shard 0 down (740 queries) | 0, `partial: true` | n/a: half the corpus is missing | see below |

* **A dead process is cheap.** Connection refused returns immediately, so the
  one request that hit it failed over within the same call. The breaker then
  keeps traffic off the dead replica.
* **A hung process is the expensive case**, because detection waits for the
  timeout. The first version of the breaker readmitted a replica to full
  rotation when its cooldown expired. Every request routed to it in the next
  second then paid the full 1 s timeout: **18** requests over 500 ms and p99
  **1,049 ms**. With the half-open state, only one probe per cooldown goes to
  the suspect replica: **6** requests over 500 ms and p99 **95 ms**. The 6 that
  remain are the requests already in flight when the hang began, plus one
  probe per 5 s window. Hedged requests (send a duplicate to the sibling after
  ~p95) would remove those too, at the cost of extra load.
* **A whole shard down means degraded results, not a failed request.** The
  gateway answers from the shards that responded and marks the response
  `partial: true, missing_shards: [0]`. Because the hash sharding is uniform,
  the targets of 389 of 740 queries (52.6 %) are on the surviving shard, and
  nDCG@10 falls to roughly that fraction of healthy: title 0.546, paraphrased
  0.472, keyword 0.461 (healthy 0.991 / 0.946 / 0.929). A caller that needs
  completeness passes `require_all=true` and gets HTTP 503 instead. That
  choice belongs to the caller: a search box should prefer partial results,
  and a deduplication job should not accept them.
* A late reply is never mixed into a response. A replica is only used if it
  answers within the timeout, under the pinned version.

## Index versioning

Every build is an immutable directory `index/shards/<version>/` with a
`manifest.json`: shard count, assignment rule, docs per shard, embedding
model, index type, BM25 parameters, and a **sha256 for every file**. The
manifest is written last and the directory is renamed into place atomically,
so a crashed build never looks complete. Shard servers verify the checksums
before serving. `CURRENT` names the default version and is flipped with an
atomic rename.

A rollout is blue/green at the replica level:

1. build `v2` next to `v1` (`build_shards --version v2 --no-set-current`);
2. `POST /admin/load {"version": "v2"}` on every replica. Each replica now
   holds v1 and v2 and answers each request from the version the request
   names;
3. `POST /admin/version {"version": "v2"}` on the gateway. It switches only if
   every shard has at least one replica that has v2 loaded, and refuses
   otherwise;
4. unload v1.

The gateway reads the pinned version **once per query**, sends it with every
shard request, and labels the response with it. A replica that does not hold
that version answers 409, which is handled exactly like a dead replica (fail
over). The demo rolled v1 (Flat) → v2 (HNSW) under continuous traffic:

| | result |
|---|---|
| switch requested before replicas loaded v2 | **refused**: `not_ready_shards: [0, 1]` |
| requests during rollout (4 clients) | 1,500, **0 errors** |
| answered from v1 / v2 | 66 / 1,434 |
| responses mixing versions across shards | **0** |
| nDCG@10 on v2 | 0.9899 / 0.9461 / 0.9285 (HNSW ef=256; v1 was 0.9906 on title) |

The mixed-version count was **1** in the first run of this experiment. The
cause was a bug: the response was labelled with the version pinned *when it
returned*, not the one its shard requests were sent with, so a query in flight
during the switch carried v1 results labelled v2. It is fixed (the version is
read once per request) and covered by
`test_version_switch_during_fanout_keeps_the_request_version`.

## Not done

* **Scale-out measurement on separate hosts.** See the note above.
* **Hedged requests**, to remove the remaining timeout tail from hung replicas.
* **Active health checks.** Failure detection is passive, i.e. by real
  traffic; a background `/health` poll would find a dead replica before a user
  request does.
* **Online writes.** New documents arrive by building a new version. A
  write path with per-shard WALs and replica catch-up is the next layer.
* **Coordinator high availability.** The gateway is stateless apart from the
  pinned version and breaker state, so several can run behind a load balancer.
  They would need to agree on the pinned version, which a shared `CURRENT` or a
  small config store would provide.
