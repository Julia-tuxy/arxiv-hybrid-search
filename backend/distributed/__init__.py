"""Minimal distributed search: hash-sharded indexes, replicated shard servers,
a fan-out/merge coordinator with failover, and versioned index builds.

    sharding.py       shard assignment + building a versioned set of shard indexes
    shard_server.py   serves one shard (dense + BM25) over HTTP, any number of versions
    merge.py          exact top-k merging of per-shard result lists
    coordinator.py    fan-out, replica failover, partial results, version pinning
    gateway.py        the coordinator as an HTTP API (GET /search, GET /cluster)
    run_cluster_demo.py  launches 2 shards x 2 replicas and measures all of the above

See distributed/README.md for the design.
"""
