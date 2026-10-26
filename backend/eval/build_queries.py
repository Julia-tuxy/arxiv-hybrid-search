"""Build the two evaluation query sets.

Set A -- `title`: the paper's own title is the query. Lexically almost fully
contained in the indexed text, so it measures the easy, keyword-shaped end of
the query distribution (and is deliberately biased in BM25's favour).

Set B -- `nl`: natural-language information needs written from the abstract by
an LLM annotator, paraphrased so the title's distinctive wording is not reused
(~12 words, 79% of query terms occur in the target document).

Set C -- `short`: 3-6 keyword-style queries over the same 120 papers, i.e. what
users actually type. Few terms means many plausible matches, so this is the
discriminative end of the query-length spectrum.

Both are single-positive: the source paper is the only judged-relevant doc, so
absolute scores understate real relevance. Relative comparison between systems
-- which is what the ablation is for -- is unaffected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def build_title_set(df: pd.DataFrame, n: int, seed: int) -> list[dict]:
    alen = df.abstract.str.split().str.len()
    tlen = df.title.str.split().str.len()
    pool = df[(alen >= 80) & (alen <= 260) & (tlen >= 5) & (tlen <= 22)]
    pool = pool[~pool.title.duplicated(keep=False)]
    sample = pool.sample(n, random_state=seed)
    return [{"qid": f"title-{pos}", "query": row.title, "relevant": [int(pos)],
             "type": "title"}
            for pos, row in zip(sample.index, sample.itertuples())]


def build_authored_set(tsv: Path, qtype: str) -> list[dict]:
    out = []
    for line in tsv.read_text().splitlines():
        if not line.strip():
            continue
        pos, query = line.split("\t")
        out.append({"qid": f"{qtype}-{pos}", "query": query, "relevant": [int(pos)],
                    "type": qtype})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-dir", default=str(HERE / "queries"))
    ap.add_argument("--n-title", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    df = pd.read_parquet(args.metadata)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = [("title_queries", build_title_set(df, args.n_title, args.seed))]
    for tsv in sorted(out_dir.glob("*_queries.tsv")):
        sets.append((tsv.stem, build_authored_set(tsv, tsv.stem.replace("_queries", ""))))

    for name, rows in sets:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} queries -> {path}")


if __name__ == "__main__":
    main()
