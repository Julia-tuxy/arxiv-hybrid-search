"""Render the recall/latency frontier from results/ann_bench.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
SERIES = {
    "flat":  ("#0b0b0b", "Flat (exact)", ""),
    "ivf":   ("#2a78d6", "IVF-Flat, nlist=1024", "nprobe"),
    "ivfpq": ("#9b59b6", "IVF-PQ 48x8, nlist=1024", "nprobe"),
    "hnsw":  ("#eb6834", "HNSW, M=32", "ef"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ann_bench.json")
    ap.add_argument("--out", default="results/ann_frontier.png")
    args = ap.parse_args()
    r = json.loads(Path(args.results).read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, (metric, ylabel) in zip(axes, [
            (lambda s: s["recall@10"], "index recall@10 (vs exact search)"),
            (lambda s: s["ndcg@10"]["short"]["hybrid"], "served hybrid nDCG@10, keyword queries")]):
        ax.set_facecolor(SURFACE)
        for kind, (color, label, pname) in SERIES.items():
            e = r["indexes"].get(kind)
            if not e:
                continue
            xs = [s["latency"]["p50_ms"] for s in e["sweep"]]
            ys = [metric(s) for s in e["sweep"]]
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
                    label=f"{label} ({e['size_mb']:.0f} MB)")
            if pname:
                for s, x, y in zip(e["sweep"], xs, ys):
                    ax.annotate(f"{s['param']}", (x, y), textcoords="offset points",
                                xytext=(4, -10), fontsize=6.5, color=color)
        ax.set_xscale("log")
        ax.set_xlabel("p50 search latency per query, ms (log, 1 thread, encoding excluded)",
                      color=INK_2, fontsize=8.5)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=8.5)
        ax.grid(True, color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=8)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(f"ANN trade-off on {r['config']['n_vectors']:,} x {r['config']['dim']}-d "
                 f"MiniLM vectors, {r['config']['n_queries']} queries",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
