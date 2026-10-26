"""Render the fusion-weight sweep from results/ablation.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
SERIES = {                      # categorical slots 1-3
    "title": ("#2a78d6", "title-as-query"),
    "nl":    ("#eb6834", "paraphrased (~12 w)"),
    "short": ("#1baf7a", "keyword (~5 w)"),
}
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ablation.json")
    ap.add_argument("--out", default="results/alpha_sweep.png")
    args = ap.parse_args()

    runs = json.loads(Path(args.results).read_text())["runs"]

    def key(a: float) -> str:
        return f"fusion_weighted_a{a:g}"

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for qtype, (color, label) in SERIES.items():
        ys = [runs[key(a)][qtype]["ndcg@10"] for a in ALPHAS]
        ax.plot(ALPHAS, ys, color=color, lw=2, zorder=3, label=label)
        best = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot([ALPHAS[best]], [ys[best]], "o", ms=8, color=color,
                mec=SURFACE, mew=2, zorder=4)
        dx, dy = {"title": (0, -26), "nl": (0, 13), "short": (46, -22)}[qtype]
        ax.annotate(f"best \u03b1={ALPHAS[best]:.1f} \u00b7 {ys[best]:.3f}",
                    (ALPHAS[best], ys[best]), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=8.5, color=color, zorder=5)

    ax.axvline(0.7, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.annotate("shipped \u03b1 = 0.7", (0.7, 0.806), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK_2)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.80, 1.005)
    ax.set_xlabel("α  —  weight on the dense arm   (0 = BM25 only, 1 = dense only)",
                  fontsize=9.5, color=INK_2)
    ax.set_ylabel("nDCG@10", fontsize=9.5, color=INK_2)
    ax.set_title("Score fusion beats either arm — and the best weight moves with query length",
                 fontsize=11.5, color=INK, loc="left", pad=26)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False,
              fontsize=9, labelcolor=INK_2, handlelength=1.6, columnspacing=1.8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
