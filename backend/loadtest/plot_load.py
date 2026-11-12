"""Render eval/results/load_test.json: batching throughput/latency and the e2e stage split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
MODES = {"unbatched": ("#52514e", "no batching"),
         "batched_wait5ms": ("#2a78d6", "micro-batch, 5 ms window"),
         "batched_greedy": ("#eb6834", "micro-batch, greedy")}
STAGES = [("embed_ms", "#2a78d6", "embed (service round trip)"),
          ("ann_ms", "#eb6834", "ANN search"),
          ("bm25_ms", "#1baf7a", "BM25"),
          ("fusion_ms", "#9b59b6", "fusion"),
          ("format_ms", "#c9a227", "format")]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)


def main() -> None:
    here = Path(__file__).resolve().parents[1] / "eval" / "results"
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(here / "load_test.json"))
    ap.add_argument("--out", default=str(here / "load_test.png"))
    args = ap.parse_args()
    r = json.loads(Path(args.results).read_text())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    eb = r["embed_batching"]
    for mode, (color, label) in MODES.items():
        if mode not in eb:
            continue
        levels = sorted(eb[mode], key=int)
        x = [int(c) for c in levels]
        axes[0].plot(x, [eb[mode][c]["throughput_rps"] for c in levels], color=color,
                     lw=2, marker="o", ms=6, mec=SURFACE, label=label)
        axes[1].plot(x, [eb[mode][c]["client"]["p99_ms"] for c in levels], color=color,
                     lw=2, marker="o", ms=6, mec=SURFACE, label=label)
    for ax, yl in zip(axes[:2], ["embedding throughput, requests/s",
                                 "embedding p99 latency, ms (client side)"]):
        _style(ax)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("concurrent clients", color=INK_2, fontsize=8.5)
        ax.set_ylabel(yl, color=INK_2, fontsize=8.5)
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7.5)

    ax = axes[2]
    variants = [(k, lab) for k, lab in (("search_e2e", "Flat"), ("search_e2e_hnsw", "HNSW"))
                if k in r]
    levels = sorted((c for c in r[variants[0][0]] if c.isdigit()), key=int)
    width = 0.8 / len(variants)
    for vi, (key, lab) in enumerate(variants):
        e2e = r[key]
        xs = np.arange(len(levels)) + (vi - (len(variants) - 1) / 2) * width
        bottom = np.zeros(len(levels))
        for skey, color, slabel in STAGES:
            vals = np.array([e2e[c]["stages"].get(skey, {}).get("p50_ms", 0.0) for c in levels])
            ax.bar(xs, vals, bottom=bottom, color=color, width=width * 0.92,
                   label=slabel if vi == 0 else None, hatch=None if vi == 0 else "///",
                   edgecolor=SURFACE, lw=0.4)
            bottom += vals
        for x, c in zip(xs, levels):
            ax.annotate(f"{lab}\n{e2e[c]['throughput_rps']:.0f} rps", (x, bottom[levels.index(c)]),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=5.5,
                        color=INK_2)
    _style(ax)
    ax.set_xticks(range(len(levels)), [f"c={c}" for c in levels])
    ax.set_ylabel("search API p50 by stage, ms (Flat left, HNSW right)", color=INK_2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Embedding service batching and search API stage latency under concurrent load",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
