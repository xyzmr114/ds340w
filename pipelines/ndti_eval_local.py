"""Score NDTI residue-cover baselines on the local dekadal cubes (ndti_local.py).

Task (EXPERIMENT_PROTOCOL.md, Task A): from six input dekads ending at the issue dekad t,
forecast NDTI at t+1, t+2, t+3.  A target pixel is scored only if it is WorldCover cropland,
clear in the target dekad, and NDVI < 0.30 in the target dekad (non-green, residue-relevant),
and has at least one clear observation in the input window (so every model sees some data).
Every model is scored on exactly the same target pixels.

Seasons: train 2018-2022, validation 2023, test 2024 and 2025 (season Y = Sep 11 Y .. Jan 31 Y+1).
Blocks: mh_grid.split_of (2x2 super-blocks, seed 340).

Models
  persistence   last clear input NDTI (P10 = P50 = P90)
  climatology   per-pixel quantiles of the masked target NDTI at the same dekad in 2018-2022
                (block-wide dekad quantiles where a pixel has no training value)
  ridge         linear model on issue-time features (point forecast only)
  gbm           gradient-boosted quantile regression (P10/P50/P90) on issue-time features;
                iterations chosen by validation pinball loss on validation blocks, 2023

Usage: python ndti_eval_local.py --plan ../results_v2/ndti_local/plan.json
Writes ../results_v2/ndti_local/{rows.csv, summary.json}
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
CUBES = os.path.join(HERE, "..", "local_data", "harvest20")
OUT = os.path.join(HERE, "..", "results_v2", "ndti_local")
T_IN, LEADS, NDVI_MAX = 6, (1, 2, 3), 0.30
TRAIN, VAL, TEST = [2018, 2019, 2020, 2021, 2022], [2023], [2024, 2025]
QS = (0.1, 0.5, 0.9)
FEATS = ["ndti_last", "age", "ndti_mean", "ndti_delta", "ndvi_last", "ndvi_max", "since_max", "dekad", "n_obs"]


def load(b, y):
    p = os.path.join(CUBES, f"blk_{b:03d}_{y}.npz")
    if not os.path.exists(p):
        return None
    try:
        d = np.load(p)
        ndti, ndvi, n = d["ndti"].astype(np.float32), d["ndvi"].astype(np.float32), d["nclr"]
    except Exception:        # cube still being written
        return None
    clear = (n > 0) & np.isfinite(ndti) & np.isfinite(ndvi)
    return np.where(clear, ndti, np.nan), np.where(clear, ndvi, np.nan)


def crop_mask(b, season=None):
    """WorldCover cropland on the cube's own grid. Cubes were built on either UTM 43N or 44N;
    cube_crs.json records which (matched by NDVI/WorldCover agreement), and wc_<b>_<epsg>.npy
    holds WorldCover on that grid."""
    m = os.path.join(CUBES, "cube_crs.json")
    if season is not None and os.path.exists(m):
        crs = json.load(open(m)).get(f"{b}_{season}")
        if crs:
            return np.load(os.path.join(CUBES, f"wc_{b:03d}_{crs}.npy")) == 40
    return np.load(os.path.join(CUBES, f"wc_{b:03d}.npy")) == 40


def target(ndti, ndvi, crop, t):
    return np.where(crop & np.isfinite(ndti[t]) & (ndvi[t] < NDVI_MAX), ndti[t], np.nan)


def features(ndti, ndvi, t):
    """Issue-time features for every pixel, using dekads <= t only."""
    win = ndti[t - T_IN + 1:t + 1]
    vwin = ndvi[t - T_IN + 1:t + 1]
    obs = np.isfinite(win)
    n_obs = obs.sum(0)
    last_idx = np.where(obs.any(0), T_IN - 1 - np.argmax(obs[::-1], 0), -1)
    take = lambda a, i: np.take_along_axis(a, np.clip(i, 0, None)[None], 0)[0]
    ndti_last = np.where(last_idx >= 0, take(win, last_idx), np.nan)
    # second-to-last observation for the recent change
    obs2 = obs.copy()
    np.put_along_axis(obs2, np.clip(last_idx, 0, None)[None], False, 0)
    idx2 = np.where(obs2.any(0), T_IN - 1 - np.argmax(obs2[::-1], 0), -1)
    ndti_prev = np.where(idx2 >= 0, take(win, idx2), np.nan)
    vobs = np.isfinite(vwin)
    vidx = np.where(vobs.any(0), T_IN - 1 - np.argmax(vobs[::-1], 0), -1)
    ndvi_last = np.where(vidx >= 0, take(vwin, vidx), np.nan)
    season = ndvi[:t + 1]
    with np.errstate(all="ignore"):
        ndvi_max = np.nanmax(np.where(np.isfinite(season), season, -np.inf), 0)
        ndvi_max = np.where(np.isfinite(ndvi_max), ndvi_max, np.nan)
        arg = np.argmax(np.where(np.isfinite(season), season, -np.inf), 0)
        ndti_mean = np.nanmean(win, 0)
    f = {
        "ndti_last": ndti_last, "age": np.where(last_idx >= 0, T_IN - 1 - last_idx, np.nan),
        "ndti_mean": ndti_mean, "ndti_delta": ndti_last - ndti_prev, "ndvi_last": ndvi_last,
        "ndvi_max": ndvi_max, "since_max": np.where(np.isfinite(ndvi_max), t - arg, np.nan),
        "dekad": np.full(ndti_last.shape, t, np.float32), "n_obs": n_obs.astype(np.float32),
    }
    return f, n_obs > 0


def climatology(b, crop):
    """(14, 3, H, W) per-pixel training quantiles and (14, 3) block-wide fallback."""
    stack = []
    for y in TRAIN:
        d = load(b, y)
        if d is None:
            continue
        stack.append(np.stack([target(d[0], d[1], crop, t) for t in range(d[0].shape[0])]))
    if not stack:
        return None, None, 0
    s = np.stack(stack)                                   # (years, 14, H, W)
    with np.errstate(all="ignore"):
        q = np.moveaxis(np.nanquantile(s, QS, axis=0), 0, 1)              # (14, 3, H, W)
        fb = np.stack([np.nanquantile(s[:, t], QS) for t in range(s.shape[1])])  # (14, 3)
    return q.astype(np.float32), fb.astype(np.float32), len(stack)


def samples(blocks, years, rng, per=4000):
    """Training/validation rows: (lead -> X, y) sampled from scored pixels."""
    out = {L: ([], []) for L in LEADS}
    for b in blocks:
        crop = crop_mask(b)
        for y in years:
            d = load(b, y)
            if d is None:
                continue
            ndti, ndvi = d
            for t in range(T_IN - 1, ndti.shape[0] - max(LEADS)):
                f, has = features(ndti, ndvi, t)
                for L in LEADS:
                    tg = target(ndti, ndvi, crop, t + L)
                    ok = np.flatnonzero((has & np.isfinite(tg)).ravel())
                    if len(ok) == 0:
                        continue
                    pick = rng.choice(ok, min(per, len(ok)), replace=False)
                    out[L][0].append(np.stack([f[k].ravel()[pick] for k in FEATS], 1))
                    out[L][1].append(tg.ravel()[pick])
    return {L: (np.concatenate(v[0]), np.concatenate(v[1])) for L, v in out.items() if v[0]}


def pinball(y, q, tau):
    e = y - q
    return np.maximum(tau * e, (tau - 1) * e)


def fit_models(train_blocks, val_blocks, rng):
    tr, va = samples(train_blocks, TRAIN, rng), samples(val_blocks, VAL, rng)
    models, info = {}, {}
    for L in LEADS:
        X, y = tr[L]
        med = np.nanmedian(X, 0)
        fill = lambda A: np.where(np.isfinite(A), A, med)
        ridge = Ridge(alpha=1.0).fit(fill(X), y)
        gb = {}
        for tau in QS:
            best = None
            for it in (100, 200, 400):
                m = HistGradientBoostingRegressor(loss="quantile", quantile=tau, max_iter=it,
                                                  learning_rate=0.08, max_leaf_nodes=31,
                                                  min_samples_leaf=100, random_state=0).fit(X, y)
                if L in va:
                    loss = float(pinball(va[L][1], m.predict(va[L][0]), tau).mean())
                else:
                    loss = 0.0
                if best is None or loss < best[0]:
                    best = (loss, it, m)
            gb[tau] = best[2]
            info[f"lead{L}_q{tau}"] = {"iters": best[1], "val_pinball": round(best[0], 5)}
        models[L] = {"ridge": ridge, "med": med, "gbm": gb}
        info[f"lead{L}_train_rows"] = int(len(y))
        info[f"lead{L}_val_rows"] = int(len(va[L][1])) if L in va else 0
    return models, info


def score_block(b, models, div, frac):
    crop = crop_mask(b)
    q, fb, nyears = climatology(b, crop)
    rows = []
    for y in TEST:
        d = load(b, y)
        if d is None:
            continue
        ndti, ndvi = d
        for t in range(T_IN - 1, ndti.shape[0] - max(LEADS)):
            f, has = features(ndti, ndvi, t)
            X = np.stack([f[k].ravel() for k in FEATS], 1)
            clear_rate = float(np.isfinite(ndti[t - T_IN + 1:t + 1])[:, crop].mean())
            for L in LEADS:
                tg = target(ndti, ndvi, crop, t + L)
                ok = (has & np.isfinite(tg)).ravel()
                n = int(ok.sum())
                if n == 0:
                    continue
                yv = tg.ravel()[ok]
                Xk = X[ok]
                preds = {"persistence": np.repeat(Xk[:, 0:1], 3, 1)}
                if q is not None:
                    cq = q[t + L].reshape(3, -1)[:, ok].T
                    cq = np.where(np.isfinite(cq), cq, fb[t + L][None])
                    preds["climatology"] = np.sort(cq, 1)
                M = models[L]
                Xf = np.where(np.isfinite(Xk), Xk, M["med"])
                r = M["ridge"].predict(Xf)
                preds["ridge"] = np.repeat(r[:, None], 3, 1)
                g = np.sort(np.stack([M["gbm"][tau].predict(Xk) for tau in QS], 1), 1)
                preds["gbm"] = g
                for name, p in preds.items():
                    if not np.all(np.isfinite(p)):
                        p = np.where(np.isfinite(p), p, np.nanmedian(yv))
                    e = p[:, 1] - yv
                    pin = np.mean([pinball(yv, p[:, i], tau) for i, tau in enumerate(QS)], 0)
                    rows.append(dict(model=name, block=b, division=div, cropfrac=round(frac, 3), season=y,
                                     issue=t, lead=L, n=n, clear_rate=round(clear_rate, 3),
                                     sae=float(np.abs(e).sum()), sse=float((e ** 2).sum()), se=float(e.sum()),
                                     spin=float(pin.sum()),
                                     cover=int(((yv >= p[:, 0]) & (yv <= p[:, 2])).sum()),
                                     below=int((yv < p[:, 0]).sum()), clim_years=nyears))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(340)
    models, info = fit_models(plan["train"], plan["val"], rng)
    rows = []
    for b in plan["test"]:
        rows += score_block(b, models, plan["division"][str(b)], plan["cropfrac"][str(b)])
        print("scored", b, flush=True)
    with open(os.path.join(OUT, "rows.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump({"fit": info, "features": FEATS, "plan": plan}, open(os.path.join(OUT, "fit.json"), "w"), indent=1)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
