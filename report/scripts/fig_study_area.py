"""Study-area figure for the final report, drawn at print size (one IEEE column, 3.5 in wide).

Same content as delta hw/tools/fig_study_area_final.py (479-cell grid coloured by super-block split and the blocks
used in the final experiment outlined by split), with fonts sized for the column, the legend below the map and the
division labels moved off the block outlines. Reads the project files read-only.
Usage: python -B fig_study_area.py -> ../figures/fig_final_study_area.{pdf,png}
"""
import json
import os
import sys

sys.dont_write_bytecode = True
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle

ROOT = os.environ.get("DS340W_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, os.path.join(ROOT, "pipelines"))
from mh_grid import blocks, bbox, split_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FILL = {"train": "#deebf7", "val": "#fee6ce", "test": "#fcbba1"}
EDGE = {"train": "#2171b5", "val": "#d94801", "test": "#000000"}
NAME = {"train": "training", "val": "validation", "test": "test"}


def rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        yield poly[0]


def main():
    fit = json.load(open(os.path.join(ROOT, "results_v2", "final", "fit.json")))
    used = {"train": sorted({b for b, _ in fit["train_block_seasons"]}),
            "val": sorted({b for b, _ in fit["val_block_seasons"]}),
            "test": sorted({b for b, _ in fit["test_block_seasons"]})}
    gj = json.load(open(os.path.join(ROOT, "data", "maharashtra.geojson"), encoding="utf-8"))
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(3.5, 3.2))
    ax = fig.add_axes([0.088, 0.14, 0.905, 0.85])
    for b, _, _ in blocks():
        w, s, e, n = bbox(b)
        ax.add_patch(Rectangle((w, s), .25, .25, facecolor=FILL[split_of(b)], edgecolor="white", lw=0.25, zorder=1))
    for f in gj["features"]:
        for r in rings(f["geometry"]):
            xs, ys = zip(*r)
            ax.plot(xs, ys, color="#9e9e9e", lw=0.3, zorder=2)
    for k in ("train", "val", "test"):
        for b in used[k]:
            assert split_of(b) == k, (b, k)
            w, s, e, n = bbox(b)
            ax.add_patch(Rectangle((w + .015, s + .015), .22, .22, fill=False, edgecolor=EDGE[k],
                                   lw=1.0 if k == "test" else 0.8, zorder=4))
    label_xy = {"Konkan": (72.55, 18.25), "Pune": (74.95, 16.45), "Nashik": (73.6, 21.45),
                "Marathwada": (76.85, 18.72), "Vidarbha": (79.2, 19.05)}
    for d, (x, y) in label_xy.items():
        ax.text(x, y, d, fontsize=7, style="italic", color="#303030", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
    ax.set_xlim(72.4, 81.1); ax.set_ylim(15.4, 22.2); ax.set_aspect(1 / np.cos(np.radians(19)))
    ax.set_xticks(range(73, 81, 2)); ax.set_yticks(range(16, 23, 2))
    ax.set_xticklabels([f"{v}°E" for v in range(73, 81, 2)]); ax.set_yticklabels([f"{v}°N" for v in range(16, 23, 2)])
    ax.tick_params(labelsize=7, length=2, pad=1.5)
    handles = [Patch(facecolor=FILL[k], edgecolor=EDGE[k], lw=1.0, label=f"{NAME[k]} ({len(used[k])} used)")
               for k in ("train", "val", "test")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               fontsize=7, frameon=False, handlelength=1.4, columnspacing=1.2, labelspacing=0.35)
    out = os.path.join(os.path.dirname(HERE), "figures", "fig_final_study_area")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=250)
    print("wrote", out, {k: len(v) for k, v in used.items()})


if __name__ == "__main__":
    main()
