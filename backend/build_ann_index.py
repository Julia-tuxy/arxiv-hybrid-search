"""Build an approximate vector index from the exact one written by ingest.py.

    python build_ann_index.py --kind hnsw          # -> index/faiss_hnsw.bin
    python build_ann_index.py --kind ivf --nlist 1024

The vectors are read back out of index/faiss_index.bin (IndexFlatIP stores them
verbatim), so no re-encoding is needed. Serve the result with
VECTOR_INDEX=hnsw (or ivf / ivfpq); see search.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import faiss

import ann_index
from ann_index import IndexSpec

INDEX_DIR = Path(os.environ.get("INDEX_DIR", Path(__file__).resolve().parent / "index"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=[k for k in ann_index.KINDS if k != "flat"], required=True)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--pq-m", type=int, default=48)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    args = ap.parse_args()

    flat = faiss.read_index(str(INDEX_DIR / "faiss_index.bin"))
    xb = flat.reconstruct_n(0, flat.ntotal)
    spec = IndexSpec(args.kind, nlist=args.nlist, pq_m=args.pq_m,
                     hnsw_m=args.hnsw_m, ef_construction=args.ef_construction)
    index, t = ann_index.build(spec, xb)
    out = INDEX_DIR / f"faiss_{args.kind}.bin"
    ann_index.save(index, out)
    print(f"{spec.label}: {index.ntotal:,} vectors, train {t['train_s']:.1f}s, "
          f"add {t['add_s']:.1f}s, {out.stat().st_size / 1e6:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
