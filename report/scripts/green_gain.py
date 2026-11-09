"""Pixel-level split of the pooled absolute-error reduction of the GBM against persistence and climatology
by whether the last clear input was still green (NDVI >= 0.30).

Reads the full test caches (local_data/cache_final/v1_full_<b>_<y>.npz) and models.pkl written by
pipelines/final_eval.py, re-predicts gbm_base exactly as final_eval.predict_all does (three quantile models,
sorted per row), checks the pooled MAEs against rows.csv and writes green_gain.json next to this script.
Read only with respect to the project folder.
Usage: python -B green_gain.py
"""
import json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ROOT = os.environ.get("DS340W_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))) + "/"
R = ROOT + "results_v2/final/"
CACHE = ROOT + "local_data/cache_final/"
QS = (0.1, 0.5, 0.9)
NDVI_LAST, NDVI_MAX = 4, 0.30
_M = None


def job(by):
    global _M
    b, y = by
    if _M is None:
        _M = pickle.load(open(R + "models.pkl", "rb"))["models"]
    with np.load(CACHE + f"v1_full_{b:03d}_{y}.npz") as d:
        X, yy, lead, clim, units = d["X"], d["y"].astype(np.float64), d["lead"], d["clim"], d["units"]
    Xg = np.ascontiguousarray(np.hstack([X, lead[:, None].astype(np.float32)]), dtype=np.float64)
    q = np.sort(np.stack([_M["gbm_base"][t].predict(Xg) for t in QS], 1), 1).astype(np.float32)
    gbm = q[:, 1].astype(np.float64)
    pers = X[:, 0].astype(np.float64)
    green = X[:, NDVI_LAST] >= NDVI_MAX
    out = {}
    ae_p, ae_g = np.abs(pers - yy), np.abs(gbm - yy)
    for k, msk in (("green", green), ("notgreen", ~green)):
        out[k] = {"n": int(msk.sum()), "ae_pers": float(ae_p[msk].sum()), "ae_gbm": float(ae_g[msk].sum())}
    return b, y, out


def main():
    rows = pd.read_csv(R + "rows.csv")
    rows = rows[(rows.split == "test") & (rows.model == "persistence")]
    pairs = sorted({(int(b), int(s)) for b, s in zip(rows.block, rows.season)})
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
        res = list(ex.map(job, pairs))
    tot = {k: {"n": 0, "ae_pers": 0.0, "ae_gbm": 0.0} for k in ("green", "notgreen")}
    for _, _, o in res:
        for k in tot:
            for f in tot[k]:
                tot[k][f] += o[k][f]
    N = tot["green"]["n"] + tot["notgreen"]["n"]
    gain = {k: tot[k]["ae_pers"] - tot[k]["ae_gbm"] for k in tot}
    out = {"n": N, "parts": tot,
           "pers_mae": (tot["green"]["ae_pers"] + tot["notgreen"]["ae_pers"]) / N,
           "gbm_mae": (tot["green"]["ae_gbm"] + tot["notgreen"]["ae_gbm"]) / N,
           "green_pixel_share": tot["green"]["n"] / N,
           "green_share_of_gain_vs_pers": gain["green"] / (gain["green"] + gain["notgreen"]),
           "rows_n": int(rows.n.sum()), "rows_pers_mae": float(rows.sae.sum() / rows.n.sum())}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "green_gain.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
