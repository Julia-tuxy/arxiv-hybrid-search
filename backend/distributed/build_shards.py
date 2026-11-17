"""Build a versioned, sharded copy of the index.

    python -m distributed.build_shards --version v1 --shards 2 --kind flat
    python -m distributed.build_shards --version v2 --shards 2 --kind hnsw --no-set-current

Reads index/faiss_index.bin, index/bm25_standard.* and index/metadata.parquet
(written by ingest.py and eval/build_index.py) and writes index/shards/<version>/.
See sharding.py for the layout and the assignment rule.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import faiss
import pandas as pd

from ann_index import IndexSpec
from bm25_index import SparseBM25
from distributed import sharding

BACKEND = Path(__file__).resolve().parents[1]


def build(version: str, n_shards: int, kind: str, root: str | Path,
          index_dir: str | Path, set_current: bool = True) -> dict:
    index_dir = Path(index_dir)
    flat = faiss.read_index(str(index_dir / "faiss_index.bin"))
    xb = flat.reconstruct_n(0, flat.ntotal)
    keys = pd.read_parquet(index_dir / "metadata.parquet",
                           columns=["arxiv_id"]).arxiv_id.astype(str).tolist()
    bm25 = SparseBM25.load(index_dir / "bm25_standard")
    t0 = time.time()
    m = sharding.build_version(root, version, xb, keys, bm25, n_shards,
                               index_spec=IndexSpec(kind), set_current=set_current)
    print(f"{version}: {n_shards} shards ({kind}), docs per shard "
          f"{[s['n_docs'] for s in m['shards']]}, built in {time.time() - t0:.0f}s")
    return m


def main() -> None:
    index_dir = Path(os.environ.get("INDEX_DIR", BACKEND / "index"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--kind", default="flat", choices=["flat", "ivf", "ivfpq", "hnsw"])
    ap.add_argument("--root", default=os.environ.get("SHARD_ROOT", str(index_dir / "shards")))
    ap.add_argument("--no-set-current", action="store_true",
                    help="build only; flip CURRENT later (blue/green)")
    args = ap.parse_args()
    build(args.version, args.shards, args.kind, args.root, index_dir,
          set_current=not args.no_set_current)


if __name__ == "__main__":
    main()
