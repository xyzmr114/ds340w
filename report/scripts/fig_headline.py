"""Headline results figure for the final report (single column): pooled test MAE for the five
headline models, sorted low to high, read straight from results_v2/final/summary.json. Meant to sit
near the top of the Results section so a reader gets the main comparison before the dense tables.

Usage: python -B fig_headline.py -> ../figures/fig_headline.{pdf,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DS340W_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
S = json.load(open(os.path.join(ROOT, "results_v2", "final", "summary.json")))

ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
accent, neutral = "#2a78d6", "#9a9890"

ROWS = [
    ("Persistence", S["pooled"]["persistence"]["mae"], neutral),
    ("Climatology", S["pooled"]["climatology"]["mae"], neutral),
    ("Earthformer", S["pooled_neural"]["earthformer"]["mae"], neutral),
    ("GBM (tree model)", S["pooled"]["gbm_base"]["mae"], accent),
    ("ConvLSTM (neural net)", S["pooled_neural"]["convlstm"]["mae"], accent),
]
ROWS.sort(key=lambda r: r[1], reverse=True)


def main():
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(3.4, 2.3))
    ax = fig.add_axes([0.40, 0.18, 0.56, 0.74])
    y = list(range(len(ROWS)))[::-1]
    labels = [r[0] for r in ROWS]
    vals = [r[1] for r in ROWS]
    cols = [r[2] for r in ROWS]
    ax.barh(y, vals, color=cols, height=0.58, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.02, yi, f"{v:.3f}", va="center", ha="left", fontsize=6.8, color=ink)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Pooled test MAE (NDTI units; lower is better)", fontsize=7)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.grid(axis="x", color=grid, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    out = os.path.join(os.path.dirname(HERE), "figures", "fig_headline")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=250)
    print("wrote", out)


if __name__ == "__main__":
    main()
