"""
One-time ingestion pipeline:
  1. Reservoir-sample TARGET papers from the arXiv JSONL snapshot
  2. Batch-encode title+abstract with Sentence-BERT
  3. Build a FAISS IndexFlatIP and save to index/
  4. Save metadata as Parquet and build the corpus-wide sparse BM25 index

Step 4 used to pickle the tokenised corpus (140 MB) so that BM25 could be
re-fitted per request. It now builds a SparseBM25 index (35 MB) that BM25 can
actually retrieve from -- see eval/README.md for why that mattered.
"""

import json
import os
import random
import sys

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bm25_index import SparseBM25
from tokenizer import standard_tokens

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "arxiv-metadata-oai-snapshot.json")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")

TARGET = int(os.environ.get("TARGET", 100_000))
BATCH_SIZE = 256
SEED = 42
MODEL_NAME = "all-MiniLM-L6-v2"


def clean_text(s: str) -> str:
    return " ".join(s.split())


def parse_authors(authors_parsed: list) -> str:
    parts = []
    for a in authors_parsed[:5]:
        name = " ".join(filter(None, a[:2])).strip()
        if name:
            parts.append(name)
    return ", ".join(parts)


def reservoir_sample(path: str, target: int, seed: int) -> list[dict]:
    random.seed(seed)
    reservoir = []
    kept = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Sampling", unit="lines")):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record.get("abstract") or not record.get("categories"):
                continue

            kept += 1
            if len(reservoir) < target:
                reservoir.append(record)
            else:
                j = random.randint(0, kept - 1)
                if j < target:
                    reservoir[j] = record

    print(f"Sampled {len(reservoir)} papers from {kept} valid records ({i+1} lines total)")
    return reservoir


def build_metadata(records: list[dict]) -> list[dict]:
    meta = []
    for r in records:
        meta.append({
            "arxiv_id":  r["id"],
            "title":     clean_text(r.get("title", "")),
            "abstract":  clean_text(r.get("abstract", "")),
            "authors":   parse_authors(r.get("authors_parsed", [])),
            "year":      (r.get("update_date") or "")[:4],
            "categories": r.get("categories", ""),
        })
    return meta


def build_texts(meta: list[dict]) -> list[str]:
    return [f"[TITLE] {m['title']} [ABSTRACT] {m['abstract']}" for m in meta]


def encode_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    print(f"Encoding {len(texts)} texts with {MODEL_NAME}…")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def save_artifacts(index: faiss.Index, meta: list[dict]) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss_index.bin"))
    print(f"Saved FAISS index ({index.ntotal} vectors)")

    df = pd.DataFrame(meta)
    df.to_parquet(os.path.join(INDEX_DIR, "metadata.parquet"), index=False)
    print(f"Saved metadata ({len(df)} rows)")

    # BM25 indexes the same fields as the dense arm, but without the [TITLE] /
    # [ABSTRACT] markers: they occur in every document, so they carry no IDF.
    corpus = [standard_tokens(f"{m['title']} {m['abstract']}") for m in meta]
    bm25 = SparseBM25.build(corpus)
    bm25.save(os.path.join(INDEX_DIR, "bm25_standard"))
    print(f"Saved BM25 index (vocab {len(bm25.vocab):,}, nnz {bm25.W.nnz:,})")


def main():
    print(f"Target sample size: {TARGET}")

    records = reservoir_sample(DATA_PATH, TARGET, SEED)
    meta = build_metadata(records)
    texts = build_texts(meta)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = encode_texts(texts, model)

    index = build_faiss_index(embeddings)
    save_artifacts(index, meta)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
