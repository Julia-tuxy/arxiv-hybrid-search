# arXiv Hybrid Semantic Search

Search ~100,000 arXiv papers with a two-arm retrieval system: a dense
Sentence-BERT index and a corpus-wide BM25 index retrieve independently, and
their ranked lists are fused. FastAPI backend, React + Vite frontend.

The interesting part of this project is not that it does hybrid retrieval — it
is that an offline evaluation harness decides *how*. The fusion strategy and its
weight were selected by measurement, and the first thing that measurement found
was a bug in the original architecture.

![fusion weight sweep](backend/eval/results/alpha_sweep.png)

## Headline results

Known-item retrieval over the 100K-paper index, nDCG@10 by query type
(full method, caveats and significance tests: **[backend/eval/README.md](backend/eval/README.md)**):

| System | title queries | paraphrased | keyword (~5 words) |
|---|---|---|---|
| BM25 only (corpus-wide) | 0.995 | 0.884 | 0.821 |
| Dense only (MiniLM + FAISS) | 0.949 | 0.919 | 0.860 |
| Original hybrid (dense → BM25 rescoring, α=0.7) | 0.996 | 0.935 | 0.882 |
| **Weighted score fusion, α=0.5** ← served | 0.991 | 0.946 | **0.928** |

Three things the ablation established:

1. **Neither arm wins alone.** BM25 beats dense by 4.6 nDCG points on
   title-shaped queries and loses by 4.0 on short keyword queries. The ranking
   flips with query type, which is the empirical case for fusing them.
2. **The original "hybrid" wasn't one.** It pulled 100 candidates from FAISS and
   re-fitted BM25 *on those 100 documents* per request — estimating IDF from a
   100-document sample, and making a lexically-matching paper the embedding
   missed unreachable at any weight. Letting both arms retrieve over the full
   corpus lifted keyword-query nDCG@10 from 0.882 to 0.928 (paired bootstrap
   p = 0.014) and Recall@10 to 1.000. On 53 of 120 keyword queries, a top-10
   result now comes from a document the dense arm never returned.
3. **The hard-coded α = 0.7 was wrong for every query type.** The optimum is
   0.2 / 0.6 / 0.5 for title / paraphrased / keyword queries respectively.

BM25 was also rebuilt as a precomputed sparse weight matrix: the index went from
a 140 MB pickle to 35 MB, and corpus-wide lexical retrieval costs 0.6 ms p50
against ~30 ms for the dense arm (12.9 ms query encoding + 17.5 ms exact FAISS
search) — so the second arm is effectively free.

## Architecture

```
query
  ├─► SentenceTransformer (all-MiniLM-L6-v2) ─► FAISS IndexFlatIP ─► top 100
  └─► tokenizer ─────────────────────────────► SparseBM25         ─► top 100
                                                       │
                       min-max normalise each arm, blend α·dense + (1-α)·bm25
                                                       │
                                                  top-k results
```

```
backend/
  main.py           FastAPI app: GET /search, GET /health
  search.py         the served retrieval path
  fusion.py         score fusion — imported by both the API and the harness
  bm25_index.py     corpus-wide BM25 over a precomputed sparse weight matrix
  tokenizer.py      tokenisation shared by indexing and querying
  ingest.py         one-time build: sample → encode → FAISS + BM25 + metadata
  eval/             evaluation harness, query sets, ablation results
frontend/           React + Vite search UI
```

`fusion.py` is deliberately dependency-free and side-effect-free: `search.py`
loads ~200 MB of artifacts at import time, so the ranking math lives where it
can be unit-tested — and where the served code and the measured code are the
same function rather than two copies that drift. Running the evaluation metrics
through the production `hybrid_search()` reproduces the ablation to four decimal
places on all three query sets.

## Running it

Requires the [arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv)
at `data/arxiv-metadata-oai-snapshot.json`.

```bash
# backend
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

cd backend
python ingest.py                 # builds index/ (FAISS + BM25 + metadata); ~30 min for 100K
uvicorn main:app --reload        # http://localhost:8000

# frontend
cd frontend && npm install && npm run dev    # http://localhost:5173
```

Retrieval is configured by environment variable, so α can be re-tuned without a
code change:

| Variable | Default | Meaning |
|---|---|---|
| `HYBRID_ALPHA` | `0.5` | weight on the dense arm (0 = BM25 only, 1 = dense only) |
| `CANDIDATES_PER_ARM` | `100` | documents each arm contributes to the fusion |
| `MODEL_NAME` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `INDEX_DIR` | `backend/index` | where the artifacts live |

## Evaluation

```bash
cd backend/eval
pip install -r requirements-eval.txt
python build_index.py                                  # BM25 indexes -> ../index/
python build_queries.py --metadata ../index/metadata.parquet
python run_eval.py                                     # -> results/ablation.json
python -m pytest test_eval.py -q                       # 11 correctness tests
```

The harness scores BM25, dense, the original hybrid, weighted fusion across an
α sweep, and RRF, on three query sets, reporting Recall@10/@100, MRR@10,
nDCG@10, latency percentiles and paired-bootstrap significance.

## Known limitations

* Evaluation is **single-positive known-item retrieval**: absolute scores are
  optimistic and are not comparable to BEIR numbers. Only the relative
  comparison between systems is claimed.
* The dense index is `IndexFlatIP` — exact brute-force search. It is already
  58 % of dense-arm latency at 100K documents and scales linearly, so it will
  not survive the full 2.7M corpus (which is also 4.1 GB of float32 vectors);
  the HNSW / IVF-PQ recall-latency trade-off is unmeasured.
* Cross-encoder reranking is implemented (`eval/retrievers.py`) but has not been
  run, so no second-stage numbers are claimed.
* No metadata filtering (category, year) in the query path yet.

## API

```
GET /search?q=<query>&top_k=<1-50>
```

```json
{
  "query": "graph neural network dislocation dynamics",
  "total_results": 10,
  "results": [
    {
      "arxiv_id": "2208.03296",
      "title": "Accelerating discrete dislocation dynamics simulations with graph neural networks",
      "abstract_snippet": "...",
      "authors": "...", "year": "2023", "categories": "cond-mat.mtrl-sci",
      "score": 1.0, "vector_score": 1.0, "bm25_score": 1.0,
      "retrieved_by": "both"
    }
  ]
}
```

`retrieved_by` reports which arm surfaced the document (`both` / `dense` / `bm25`).
