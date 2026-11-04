"""Render the quality/latency frontier from results/rerank.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
SERIES = {                       # categorical slots 1-3
    "title": ("#2a78d6", "title-as-query"),
    "nl":    ("#eb6834", "paraphrased (~12 w)"),
    "short": ("#1baf7a", "keyword (~5 w)"),
}
STAGES = ["first_stage", "top10", "top20", "top50", "top100"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/rerank.json")
    ap.add_argument("--out", default="results/rerank_frontier.png")
    args = ap.parse_args()

    r = json.loads(Path(args.results).read_text())
    ce = next(iter(r["rerank"]))
    lat = r["measured_latency_p50_ms"]

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for qtype, (color, label) in SERIES.items():
        xs = [lat[s] for s in STAGES]
        ys = ([r["first_stage"][qtype]["ndcg@10"]]
              + [r["rerank"][ce][qtype][s]["ndcg@10"] for s in STAGES[1:]])
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                zorder=3, label=label)
        ax.annotate(f"{ys[0]:.3f}", (xs[0], ys[0]), textcoords="offset points",
                    xytext=(10, -4), fontsize=8.5, color=color, ha="left")
        ax.annotate(f"{ys[1]:.3f}", (xs[1], ys[1]), textcoords="offset points",
                    xytext=(0, 11), fontsize=8.5, color=color, ha="center")

    ax.set_xscale("log")
    ax.set_xlim(22, 9000)
    ax.set_ylim(0.915, 1.012)
    ax.set_xlabel("end-to-end query latency, p50 (log scale) — 2 vCPU, single process",
                  fontsize=9.5, color=INK_2, labelpad=8)
    ax.set_ylabel("nDCG@10", fontsize=9.5, color=INK_2)
    ax.set_title("Reranking the top 10 captures the whole gain; deeper only buys latency",
                 fontsize=11.5, color=INK, loc="left", pad=26)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_xticks([33, 601, 1316, 2910, 5469])
    ax.set_xticklabels(["no rerank\n33 ms", "rerank top 10\n0.6 s",
                        "top 20\n1.3 s", "top 50\n2.9 s", "top 100\n5.5 s"])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False,
              fontsize=9, labelcolor=INK_2, handlelength=1.6, columnspacing=1.8)

    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
