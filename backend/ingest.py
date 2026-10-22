"""
One-time ingestion pipeline:
  1. Reservoir-sample TARGET papers from the arXiv JSONL snapshot
  2. Batch-encode title+abstract with Sentence-BERT
  3. Build a FAISS IndexFlatIP and save to index/
  4. Save metadata as Parquet and BM25 tokenized corpus as pickle
"""

import json
import os
import pickle
import random
import sys

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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


def save_artifacts(index: faiss.Index, meta: list[dict], texts: list[str]) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss_index.bin"))
    print(f"Saved FAISS index ({index.ntotal} vectors)")

    df = pd.DataFrame(meta)
    df.to_parquet(os.path.join(INDEX_DIR, "metadata.parquet"), index=False)
    print(f"Saved metadata ({len(df)} rows)")

    corpus_tokens = [t.lower().split() for t in texts]
    with open(os.path.join(INDEX_DIR, "bm25_corpus.pkl"), "wb") as f:
        pickle.dump(corpus_tokens, f)
    print("Saved BM25 corpus")


def main():
    print(f"Target sample size: {TARGET}")

    records = reservoir_sample(DATA_PATH, TARGET, SEED)
    meta = build_metadata(records)
    texts = build_texts(meta)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = encode_texts(texts, model)

    index = build_faiss_index(embeddings)
    save_artifacts(index, meta, texts)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
