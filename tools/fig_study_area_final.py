"""Study-area figure for the final report: the 479-block grid coloured by super-block split and the blocks used in
the final experiment outlined by split (32 train, 10 val, 32 test).

Run from project root: python tools/fig_study_area_final.py -> results_v2/figures/fig_final_study_area.{pdf,png}
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipelines"))
from mh_grid import blocks, bbox, split_of  # noqa: E402

FILL = {"train": "#deebf7", "val": "#fee6ce", "test": "#fcbba1"}
EDGE = {"train": "#2171b5", "val": "#d94801", "test": "#000000"}


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
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for b, _, _ in blocks():
        w, s, e, n = bbox(b)
        ax.add_patch(Rectangle((w, s), .25, .25, facecolor=FILL[split_of(b)], edgecolor="white", lw=0.3, zorder=1))
    for f in gj["features"]:
        for r in rings(f["geometry"]):
            xs, ys = zip(*r)
            ax.plot(xs, ys, color="#9e9e9e", lw=0.35, zorder=2)
    for k in ("train", "val", "test"):
        for b in used[k]:
            assert split_of(b) == k, (b, k)
            w, s, e, n = bbox(b)
            ax.add_patch(Rectangle((w + .01, s + .01), .23, .23, fill=False, edgecolor=EDGE[k],
                                   lw=1.4 if k == "test" else 1.1, zorder=4))
    label_xy = {"Konkan": (72.95, 17.6), "Pune": (74.6, 17.1), "Nashik": (74.3, 20.9),
                "Marathwada": (76.5, 18.6), "Vidarbha": (78.9, 19.4)}
    for d, (x, y) in label_xy.items():
        ax.text(x, y, d, fontsize=8, style="italic", color="#424242", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.6))
    ax.set_xlim(72.4, 81.1); ax.set_ylim(15.4, 22.2); ax.set_aspect(1 / np.cos(np.radians(19)))
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    handles = [Patch(facecolor=FILL[k], edgecolor="none", label=f"{k} super-blocks") for k in FILL]
    handles += [Patch(facecolor="none", edgecolor=EDGE[k], lw=1.3, label=f"{k} block used ({len(used[k])})")
                for k in ("train", "val", "test")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.6, frameon=True)
    fig.tight_layout(pad=0.3)
    out = os.path.join(ROOT, "results_v2", "figures", "fig_final_study_area")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
    print("wrote", out, {k: len(v) for k, v in used.items()})


if __name__ == "__main__":
    main()
