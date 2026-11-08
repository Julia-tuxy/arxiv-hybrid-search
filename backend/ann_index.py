"""Vector index factory: exact (Flat) and approximate (IVF, IVF-PQ, HNSW) search.

Every index here uses inner product over L2-normalised vectors, i.e. cosine
similarity, so all of them rank by the same score the exact index does and an
ANN result is directly comparable to the exact one.

The served index type is chosen by measurement (eval/run_ann_bench.py, results
in eval/README.md section 5). This module only knows how to build, tune,
persist and update each type; it has no opinion about which one to use.

    kind      structure                         tunable at query time
    flat      brute-force scan of every vector  -
    ivf       k-means coarse quantiser, scan    nprobe   (lists visited)
              `nprobe` of `nlist` inverted lists
    ivfpq     IVF + product-quantised residuals nprobe
              (48 x 8-bit codes per vector)
    hnsw      navigable small-world graph       efSearch (beam width)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

KINDS = ("flat", "ivf", "ivfpq", "hnsw")


@dataclass
class IndexSpec:
    kind: str
    nlist: int = 1024          # ivf / ivfpq: number of k-means cells (~sqrt(N)*3)
    pq_m: int = 48             # ivfpq: sub-quantisers (384 / 48 = 8 dims each)
    pq_bits: int = 8
    hnsw_m: int = 32           # hnsw: graph out-degree
    ef_construction: int = 200
    train_size: int | None = None   # ivf*: vectors used for k-means (None = all)
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.kind in ("ivf", "ivfpq"):
            return f"{self.kind}{self.nlist}"
        if self.kind == "hnsw":
            return f"hnsw{self.hnsw_m}"
        return self.kind


def _check(vectors: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {v.shape}")
    return v


def build(spec: IndexSpec, vectors: np.ndarray, seed: int = 0,
          ids: np.ndarray | None = None) -> tuple[faiss.Index, dict]:
    """Build an index over `vectors`; ids are row positions 0..N-1 unless `ids`
    gives explicit ones (a shard holds global document ids -- see distributed/).

    Returns the index and a timing dict {train_s, add_s}. Training (k-means for
    IVF) and adding are timed separately because they scale differently: train
    once per index version, add on every update.
    """
    x = _check(vectors)
    d = x.shape[1]
    ip = faiss.METRIC_INNER_PRODUCT

    if spec.kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif spec.kind == "ivf":
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, spec.nlist, ip)
    elif spec.kind == "ivfpq":
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, spec.nlist,
                                 spec.pq_m, spec.pq_bits, ip)
    elif spec.kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, spec.hnsw_m, ip)
        index.hnsw.efConstruction = spec.ef_construction
    else:
        raise ValueError(f"unknown index kind {spec.kind!r}; expected one of {KINDS}")

    t0 = time.perf_counter()
    if not index.is_trained:
        sample = x
        if spec.train_size and spec.train_size < len(x):
            rng = np.random.default_rng(seed)
            sample = x[np.sort(rng.choice(len(x), spec.train_size, replace=False))]
        index.train(sample)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if ids is None:
        index.add(x)
    else:
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(x, np.ascontiguousarray(ids, dtype=np.int64))
    add_s = time.perf_counter() - t0
    return index, {"train_s": train_s, "add_s": add_s}


def set_search_params(index: faiss.Index, *, nprobe: int | None = None,
                      ef_search: int | None = None) -> None:
    """Set the query-time knob for whichever index type this is."""
    ivf = _as_ivf(index)
    if ivf is not None and nprobe is not None:
        ivf.nprobe = int(nprobe)
    hnsw = _as_hnsw(index)
    if hnsw is not None and ef_search is not None:
        hnsw.hnsw.efSearch = int(ef_search)


def kind_of(index: faiss.Index) -> str:
    if _as_hnsw(index) is not None:
        return "hnsw"
    ivf = _as_ivf(index)
    if ivf is not None:
        return "ivfpq" if isinstance(faiss.downcast_index(ivf), faiss.IndexIVFPQ) else "ivf"
    return "flat"


def _as_ivf(index):
    try:
        return faiss.extract_index_ivf(index)
    except RuntimeError:
        return None


def _as_hnsw(index):
    idx = faiss.downcast_index(index)
    if isinstance(idx, faiss.IndexIDMap):          # IndexIDMap2 subclasses IndexIDMap
        idx = faiss.downcast_index(idx.index)
    return idx if isinstance(idx, faiss.IndexHNSW) else None


def search(index: faiss.Index, queries: np.ndarray, k: int,
           exclude_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """k-NN search; `exclude_ids` filters tombstoned documents at query time.

    Filtering is how deletes work for HNSW, whose graph cannot drop a node:
    the deleted id stays in the graph as a routing waypoint but is never
    returned. Flat and IVF can delete physically (see `remove`).
    """
    q = _check(queries)
    k = min(k, index.ntotal)
    if exclude_ids is None or len(exclude_ids) == 0:
        return index.search(q, k)
    sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(np.asarray(exclude_ids, dtype=np.int64)))
    kind = kind_of(index)
    if kind == "hnsw":
        params = faiss.SearchParametersHNSW(sel=sel, efSearch=_as_hnsw(index).hnsw.efSearch)
    elif kind in ("ivf", "ivfpq"):
        params = faiss.SearchParametersIVF(sel=sel, nprobe=_as_ivf(index).nprobe)
    else:
        params = faiss.SearchParameters(sel=sel)
    return index.search(q, k, params=params)


def remove(index: faiss.Index, ids: np.ndarray) -> int:
    """Physically delete ids (Flat / IVF). Raises for HNSW -- use tombstones.

    Note that IndexFlat compacts on removal, which renumbers every later row;
    IVF keeps explicit ids, so its remaining ids are stable.
    """
    if kind_of(index) == "hnsw":
        raise NotImplementedError("HNSW cannot remove nodes; filter with exclude_ids")
    return int(index.remove_ids(faiss.IDSelectorBatch(np.asarray(ids, dtype=np.int64))))


def serialized_bytes(index: faiss.Index) -> int:
    return int(faiss.serialize_index(index).nbytes)


def save(index: faiss.Index, path: str | Path) -> None:
    faiss.write_index(index, str(path))


def load(path: str | Path, *, mmap: bool = False) -> faiss.Index:
    flags = faiss.IO_FLAG_MMAP if mmap else 0
    return faiss.read_index(str(path), flags)


def recall_at_k(found: np.ndarray, exact: np.ndarray, k: int) -> float:
    """Mean fraction of the exact top-k that the ANN top-k also returned.

    This is the index's own recall -- agreement with brute force -- and is
    distinct from the retrieval-quality Recall@10 in eval/metrics.py, which is
    measured against relevance judgements.
    """
    found, exact = np.asarray(found)[:, :k], np.asarray(exact)[:, :k]
    hits = [len(set(f.tolist()) & set(e.tolist())) for f, e in zip(found, exact)]
    return float(np.mean(hits) / k)
