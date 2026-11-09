"""Results figure for the final report (full text width), redrawn from results_v2/final/summary.json.

(a) pooled test MAE by lead; the ConvLSTM line is drawn thin and dashed over a wide, light GBM line so both stay
    visible. (b) mean block MAE difference from persistence with the 95% super-block bootstrap interval and
    wins/32. (c) the same against the GBM for ConvLSTM, Earthformer and the five weather variants. Labels follow
    the report's tables. Same numbers as final_analyze.figure(); only the layout and labels differ.
Usage: python -B fig_results.py -> ../figures/fig_final_results.{pdf,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(os.environ.get("DS340W_ROOT", os.path.abspath(os.path.join(HERE, "..", ".."))), "results_v2", "final", "summary.json")))
LABEL = {"persistence": "Persistence", "persistence_ng": "Non-green persistence", "movavg6": "Six-dekad mean",
         "seasonal": "Seasonal persistence", "climatology": "Climatology", "ridge": "Ridge", "gbm_base": "GBM",
         "gbm_rain": "GBM +rain", "gbm_soil": "GBM +soil", "gbm_temp": "GBM +temp", "gbm_chirps": "GBM +CHIRPS",
         "gbm_all": "GBM +all", "convlstm": "ConvLSTM", "earthformer": "Earthformer"}
ink, ink2, grid, dot = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def forest(ax, rows, xlabel, title, step, tx):
    yy = np.arange(len(rows))[::-1]
    for y, (m, r) in zip(yy, rows):
        ax.plot(r["ci95"], [y, y], color=ink2, lw=1.3, solid_capstyle="round")
        ax.plot(r["mean_diff"], y, "o", color=dot, ms=4.5, mec="white", mew=0.8)
        ax.text(1.02, y, f"{r['wins']}/{r['blocks']}", transform=ax.get_yaxis_transform(), va="center",
                ha="left", fontsize=6.5, color=ink2)
    ax.axvline(0, color=ink, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([LABEL[m] for m, _ in rows], fontsize=6.5)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.tick_params(axis="x", labelsize=6.5)
    ax.text(tx, 1.035, title, transform=ax.transAxes, fontsize=7.5, color=ink, ha="left", va="bottom")
    ax.grid(axis="x", color=grid, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.16, 2.6))
    a1 = fig.add_axes([0.07, 0.17, 0.2, 0.74])
    a2 = fig.add_axes([0.448, 0.17, 0.18, 0.74])
    a3 = fig.add_axes([0.82, 0.17, 0.13, 0.74])
    # (a)
    series = [("persistence", "#2a78d6", "o", 1.8, "-"), ("climatology", "#eb6834", "s", 1.8, "-"),
              ("ridge", "#1baf7a", "^", 1.8, "-")]
    for m, col, mk, lw, ls in series:
        y = [S["by_lead"][m][str(L)]["mae"] for L in (1, 2, 3)]
        a1.plot([1, 2, 3], y, color=col, lw=lw, ls=ls, marker=mk, ms=4, mec="white", mew=0.8, label=LABEL[m])
    y = [S["by_lead"]["gbm_base"][str(L)]["mae"] for L in (1, 2, 3)]
    a1.plot([1, 2, 3], y, color="#eda100", lw=4.5, alpha=0.55, marker="D", ms=6, mec="none", label="GBM", zorder=2)
    for m, col, mk in (("earthformer", "#b8327a", "P"), ("convlstm", "#1a1a1a", "v")):
        bl = S["by_lead_neural"][m]
        a1.plot([1, 2, 3], [bl[str(L)]["mae"] for L in (1, 2, 3)], color=col, lw=1.0, ls=(0, (3, 1.5)),
                marker=mk, ms=3.5, mfc="white" if m == "convlstm" else col, mec=col, mew=0.8, label=LABEL[m],
                zorder=3)
    a1.set_xticks([1, 2, 3]); a1.set_xlabel("Lead (dekads)"); a1.set_ylabel("MAE (NDTI)")
    a1.text(-0.3, 1.035, "(a) Pooled test MAE by lead", transform=a1.transAxes, fontsize=7.5, color=ink, ha="left", va="bottom")
    lo, hi = a1.get_ylim()
    lo = min(lo, 0.018)
    a1.set_ylim(lo, hi + 0.45 * (hi - lo))
    a1.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.8,
              labelspacing=0.3)
    a1.tick_params(labelsize=6.5)
    a1.grid(axis="y", color=grid, lw=0.6); a1.set_axisbelow(True)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    # (b)
    models = ["persistence_ng", "movavg6", "seasonal", "climatology", "ridge", "gbm_base", "gbm_rain", "gbm_soil",
              "gbm_temp", "gbm_chirps", "gbm_all", "convlstm", "earthformer"]
    rows = sorted([(m, S["tests"][f"{m}_vs_persistence"]["mae"]) for m in models], key=lambda r: r[1]["mean_diff"])
    forest(a2, rows, "Block MAE minus persistence", "(b) Difference from persistence", 0.02, -0.85)
    # (c)
    models = ["convlstm", "earthformer", "gbm_rain", "gbm_soil", "gbm_temp", "gbm_chirps", "gbm_all"]
    rows = sorted([(m, S["tests"][f"{m}_vs_gbm_base"]["mae"]) for m in models], key=lambda r: r[1]["mean_diff"])
    forest(a3, rows, "Block MAE minus GBM", "(c) Difference from the GBM", 0.001, -0.85)
    out = os.path.join(os.path.dirname(HERE), "figures", "fig_final_results")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=250)
    print("wrote", out)


if __name__ == "__main__":
    main()
