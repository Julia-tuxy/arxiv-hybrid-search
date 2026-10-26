"""Build the corpus-wide BM25 indexes used by the ablation (and by fused search).

Reads metadata.parquet produced by ingest.py and writes two sparse indexes into
the same index/ directory:

  bm25_simple    str.lower().split() over "[TITLE] … [ABSTRACT] …"
                 -- reproduces the tokenisation the original pipeline used, so
                 the legacy hybrid baseline is comparable.
  bm25_standard  regex tokens, stopwords removed, over title + abstract
                 -- the tokenizer the new global BM25 arm uses.

Both are far smaller and faster to load than the 139 MB pickled token list they
replace, because the BM25 weights are precomputed (see bm25_index.py).
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import pandas as pd

import _path  # noqa: F401  -- puts backend/ on sys.path
from bm25_index import SparseBM25
from tokenizer import simple_tokens, standard_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(Path(__file__).resolve().parents[1] / "index"))
    ap.add_argument("--metadata", default=None)
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    df = pd.read_parquet(args.metadata or index_dir / "metadata.parquet")
    print(f"{len(df):,} documents")

    jobs = [
        ("bm25_simple", simple_tokens,
         ("[TITLE] " + df.title + " [ABSTRACT] " + df.abstract).tolist()),
        ("bm25_standard", standard_tokens, (df.title + " " + df.abstract).tolist()),
    ]
    for name, tokenize, texts in jobs:
        t0 = time.time()
        tokens = [tokenize(t) for t in texts]
        idx = SparseBM25.build(tokens)
        idx.save(index_dir / name)
        size = sum(os.path.getsize(index_dir / f"{name}{ext}")
                   for ext in (".npz", ".vocab.pkl")) / 1e6
        print(f"{name}: vocab={len(idx.vocab):,} nnz={idx.W.nnz:,} "
              f"{size:.0f} MB in {time.time()-t0:.1f}s")
        del tokens, idx
        gc.collect()


if __name__ == "__main__":
    main()
