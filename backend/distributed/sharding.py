"""Shard assignment and versioned shard builds.

Assignment: a document lives on shard  crc32(arxiv_id) mod N.

  * Hashing the *stable external id* (not the row position) means a document's
    shard does not change when the corpus is re-sampled or re-ordered, and a new
    document's shard is known without a lookup table.
  * Hashing spreads topics uniformly. Range- or category-based sharding would
    put all of, say, hep-th on one shard: that shard would hold most of the top
    results for physics queries and become the hot spot, and its loss would
    remove a whole field instead of a uniform random half.
  * The cost of plain modulo is resharding: going from N to N+1 shards moves
    ~N/(N+1) of the documents. That is acceptable here because every index
    version is a full offline rebuild anyway (below); consistent hashing would
    matter if shards were rebalanced online.

BM25 needs one more thing to shard correctly: each shard holds rows of the
*global* BM25 weight matrix, so IDF and average document length are corpus-wide
statistics. A shard that computed IDF over its own half would give the same term
a different weight on each shard, and per-shard BM25 scores would not be
comparable when merged.

Layout written by `build_version`:

    <root>/
      CURRENT                    name of the version gateways should serve
      <version>/
        manifest.json            shard map, doc counts, checksums, build params
        bm25.vocab.pkl           global vocabulary (shared by all shards)
        shard-0/ vectors.faiss  bm25.npz  doc_ids.npy
        shard-1/ ...

A version directory is immutable once its manifest is written: the manifest is
written last, so a crashed build never looks complete.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
import zlib
from pathlib import Path

import numpy as np

MANIFEST = "manifest.json"


def shard_of(doc_key: str, n_shards: int) -> int:
    return zlib.crc32(doc_key.encode("utf-8")) % n_shards


def assign(doc_keys, n_shards: int) -> np.ndarray:
    return np.fromiter((shard_of(k, n_shards) for k in doc_keys), dtype=np.int32,
                       count=len(doc_keys))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_version(root: str | Path, version: str, vectors: np.ndarray, doc_keys: list[str],
                  bm25, n_shards: int, index_spec=None, model: str = "all-MiniLM-L6-v2",
                  set_current: bool = True) -> dict:
    """Split one corpus into `n_shards` shard indexes under <root>/<version>/.

    `vectors` row i and `bm25.W` row i both belong to global doc id i, and
    `doc_keys[i]` is its stable external id (arxiv_id). Shard indexes keep the
    global id, so merged results need no id translation.
    """
    from scipy import sparse

    import ann_index
    from ann_index import IndexSpec

    spec = index_spec or IndexSpec("flat")
    root = Path(root)
    vdir = root / version
    if (vdir / MANIFEST).exists():
        raise FileExistsError(f"version {version} already exists and is immutable")
    tmp = root / f".{version}.building"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    owner = assign(doc_keys, n_shards)
    W_csr = bm25.W.tocsr()
    with open(tmp / "bm25.vocab.pkl", "wb") as f:
        pickle.dump({"vocab": bm25.vocab, "k1": bm25.k1, "b": bm25.b}, f)

    shards = []
    for s in range(n_shards):
        ids = np.flatnonzero(owner == s).astype(np.int64)
        sdir = tmp / f"shard-{s}"
        sdir.mkdir()
        t0 = time.perf_counter()
        index, _ = ann_index.build(spec, vectors[ids], ids=ids)
        ann_index.save(index, sdir / "vectors.faiss")
        sparse.save_npz(sdir / "bm25.npz", W_csr[ids].tocsc())
        np.save(sdir / "doc_ids.npy", ids)
        shards.append({
            "shard": s, "n_docs": int(len(ids)), "build_s": round(time.perf_counter() - t0, 2),
            "files": {f.name: {"bytes": f.stat().st_size, "sha256": _sha256(f)}
                      for f in sorted(sdir.iterdir())},
        })

    manifest = {
        "version": version, "created_unix": int(time.time()), "n_shards": n_shards,
        "assignment": "crc32(arxiv_id) mod n_shards", "n_docs": int(len(doc_keys)),
        "embedding_model": model, "dim": int(vectors.shape[1]),
        "index": {"kind": spec.kind, "label": spec.label},
        "bm25": {"k1": bm25.k1, "b": bm25.b, "vocab": len(bm25.vocab),
                 "statistics": "global (idf and avgdl over the full corpus)"},
        "shards": shards,
    }
    with open(tmp / MANIFEST, "w") as f:          # manifest last: marks the build complete
        json.dump(manifest, f, indent=2)
    os.replace(tmp, vdir)
    if set_current:
        set_current_version(root, version)
    return manifest


def set_current_version(root: str | Path, version: str) -> None:
    root = Path(root)
    if not (root / version / MANIFEST).exists():
        raise FileNotFoundError(f"no complete build for version {version}")
    tmp = root / ".CURRENT.tmp"
    tmp.write_text(version + "\n")
    os.replace(tmp, root / "CURRENT")              # atomic pointer flip


def current_version(root: str | Path) -> str:
    return (Path(root) / "CURRENT").read_text().strip()


def read_manifest(root: str | Path, version: str) -> dict:
    return json.loads((Path(root) / version / MANIFEST).read_text())


def verify(root: str | Path, version: str, shard: int) -> None:
    """Checksum a shard's files against the manifest before serving them."""
    m = read_manifest(root, version)
    entry = next(s for s in m["shards"] if s["shard"] == shard)
    sdir = Path(root) / version / f"shard-{shard}"
    for name, info in entry["files"].items():
        got = _sha256(sdir / name)
        if got != info["sha256"]:
            raise ValueError(f"{version}/shard-{shard}/{name}: checksum mismatch")
