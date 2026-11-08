"""Correctness tests for the vector index layer (backend/ann_index.py).

    python -m pytest test_ann.py -q

Uses small synthetic data so it runs in seconds without the real index.
"""
from __future__ import annotations

import numpy as np
import pytest

import _path  # noqa: F401
import ann_index
from ann_index import IndexSpec


def _unit(n: int, d: int = 32, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


XB, XQ = _unit(4000), _unit(50, seed=1)


def _exact(k=10):
    return np.argsort(-(XQ @ XB.T), axis=1)[:, :k]


def test_recall_at_k_definition():
    exact = np.array([[1, 2, 3], [4, 5, 6]])
    assert ann_index.recall_at_k(exact, exact, 3) == 1.0
    assert ann_index.recall_at_k(np.array([[3, 2, 9], [9, 9, 9]]), exact, 3) == pytest.approx(2 / 6)
    assert ann_index.recall_at_k(np.array([[1], [9]]), exact, 1) == 0.5


def test_flat_is_exact():
    index, _ = ann_index.build(IndexSpec("flat"), XB)
    _, ids = index.search(XQ, 10)
    assert ann_index.recall_at_k(ids, _exact(), 10) == 1.0


def test_ivf_with_every_list_probed_is_exact():
    index, _ = ann_index.build(IndexSpec("ivf", nlist=16), XB)
    ann_index.set_search_params(index, nprobe=16)
    _, ids = index.search(XQ, 10)
    assert ann_index.recall_at_k(ids, _exact(), 10) == 1.0


def test_ivf_recall_grows_with_nprobe():
    index, _ = ann_index.build(IndexSpec("ivf", nlist=64), XB)
    recalls = []
    for p in (1, 4, 16, 64):
        ann_index.set_search_params(index, nprobe=p)
        recalls.append(ann_index.recall_at_k(index.search(XQ, 10)[1], _exact(), 10))
    assert recalls == sorted(recalls) and recalls[-1] == 1.0 and recalls[0] < 1.0


def test_hnsw_high_ef_is_near_exact_and_kind_detection():
    index, _ = ann_index.build(IndexSpec("hnsw", hnsw_m=16), XB)
    assert ann_index.kind_of(index) == "hnsw"
    ann_index.set_search_params(index, ef_search=256)
    assert ann_index.recall_at_k(index.search(XQ, 10)[1], _exact(), 10) >= 0.95


def test_kind_detection():
    for kind, spec in [("flat", IndexSpec("flat")), ("ivf", IndexSpec("ivf", nlist=8)),
                       ("ivfpq", IndexSpec("ivfpq", nlist=8, pq_m=8, pq_bits=4))]:
        assert ann_index.kind_of(ann_index.build(spec, XB)[0]) == kind


def test_incremental_add_keeps_positional_ids():
    index, _ = ann_index.build(IndexSpec("hnsw", hnsw_m=16), XB[:3000])
    index.add(XB[3000:])
    ann_index.set_search_params(index, ef_search=128)
    _, ids = index.search(XB[3500:3501], 1)      # a vector finds itself
    assert ids[0, 0] == 3500


def test_hnsw_tombstones_are_never_returned():
    index, _ = ann_index.build(IndexSpec("hnsw", hnsw_m=16), XB)
    dead = np.unique(_exact(5))
    _, ids = ann_index.search(index, XQ, 10, exclude_ids=dead)
    assert not np.isin(ids, dead).any()
    with pytest.raises(NotImplementedError):
        ann_index.remove(index, dead)


def test_ivf_physical_delete_keeps_other_ids_stable():
    index, _ = ann_index.build(IndexSpec("ivf", nlist=16), XB)
    ann_index.set_search_params(index, nprobe=16)
    dead = _exact(1)[:, 0]
    assert ann_index.remove(index, dead) == len(np.unique(dead))
    _, ids = index.search(XQ, 10)
    assert not np.isin(ids, dead).any()
    # the second-best exact neighbour is now first, under its original id
    assert (ids[:, 0] == _exact(2)[:, 1]).mean() > 0.9


def test_save_load_roundtrip(tmp_path):
    index, _ = ann_index.build(IndexSpec("ivf", nlist=16), XB)
    ann_index.save(index, tmp_path / "i.bin")
    again = ann_index.load(tmp_path / "i.bin")
    assert again.ntotal == index.ntotal
    assert ann_index.serialized_bytes(index) > XB.nbytes      # vectors + centroids
