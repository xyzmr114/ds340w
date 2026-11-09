"""Final-report evaluation of the non-neural NDTI residue-cover forecasts (v2 cubes, EXPERIMENT_PROTOCOL.md).

Task (as ndti_eval_local.py): issue dekads t = 5..10, inputs = dekads t-5..t, leads L = 1, 2, 3.
Target = NDTI at t+L on WorldCover cropland that is clear and non-green (NDVI < 0.30) at t+L.
A target pixel is scored only if it has >= 1 clear input in the window.

Splits (fixed before training): fit on train super-blocks, seasons 2018-2022; select iterations /
learning rate / ridge alpha on val super-blocks, season 2023 (pinball loss); score test super-blocks,
seasons 2024 and 2025 (season Y = 11 Sep Y .. 31 Jan Y+1).

Identical pixels: for every test (block, season, issue, lead) unit the scored pixel set is computed
once, in extract(), and stored in the cache; every model's prediction is an array over exactly
those rows, with a documented fallback wherever a model has no value of its own, so all models in
a unit share n and pixels (asserted in score_bs). Weather variants are emitted only for units
whose block-season has the weather series; when emitted they use the same rows.

Models
  persistence      last clear input NDTI
  persistence_ng   last clear input whose NDVI < 0.30 (fallback: persistence)
  movavg6          mean of the clear inputs in the six-dekad window
  seasonal         masked target NDTI at the same pixel and target dekad, previous season
                   (fallback: persistence)
  climatology      per-pixel P10/P50/P90 of the masked target at the same dekad over the block's
                   2018-2022 seasons (pixel needs >= --clim-min-years values); fallback block-wide
                   dekad quantiles; last resort the pooled training-row quantiles for that dekad
  ridge            point forecast; 9 issue-time features (+missing flag) + lead one-hot, alpha by val MAE
  gbm_base         HistGradientBoostingRegressor quantile loss (P10/P50/P90), 9 features + lead
  gbm_rain/_soil/_temp/_chirps/_all   gbm_base + weather (ERA5 Open-Meteo, CHIRPS block mean):
                   rain = ERA5 precipitation summed over the input window and over the issue dekad,
                   soil = ERA5 0-7 cm soil moisture, mean of the issue dekad,
                   temp = ERA5 2 m temperature, mean of the issue dekad,
                   chirps = CHIRPS block-mean rain summed over the input window and the issue dekad,
                   all = every weather column above

Usage (run from pipelines/):
  python final_eval.py                 # extract (cached) -> fit (cached) -> score (resumable)
  python final_eval.py --quick         # smoke test on a few blocks with whatever cubes exist
Outputs: ../results_v2/final/{rows.csv, fit.json, models.pkl}; cache ../local_data/cache_final/
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import ndti_eval_local as E          # target(), features(), FEATS, T_IN, NDVI_MAX
from mh_grid import superblock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUBES = os.path.join(ROOT, "local_data", "harvest20v2")
CACHE = os.path.join(ROOT, "local_data", "cache_final")
WX2 = os.path.join(ROOT, "local_data", "weather_v2")
WX1 = os.path.join(ROOT, "local_data", "weather")
PLAN = os.path.join(ROOT, "results_v2", "ndti_local", "plan.json")
CACHE_VER = 1
# all-NaN windows/dekads are expected (no clear observation); their NaN results are handled explicitly
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

ISSUES, LEADS, QS = tuple(range(5, 11)), (1, 2, 3), (0.1, 0.5, 0.9)
TRAIN, VAL, TEST = [2018, 2019, 2020, 2021, 2022], [2023], [2024, 2025]
FEATS = E.FEATS
WCOLS = ["rain6", "rain1", "soil", "temp", "chirps6", "chirps1"]
WSETS = {"gbm_rain": ["rain6", "rain1"], "gbm_soil": ["soil"], "gbm_temp": ["temp"],
         "gbm_chirps": ["chirps6", "chirps1"], "gbm_all": WCOLS}
BASELINES = ["persistence", "persistence_ng", "movavg6", "seasonal", "climatology"]
MODELS = BASELINES + ["ridge", "gbm_base"] + list(WSETS)


# ============================================================================ cubes
def cube_path(b, y):
    return os.path.join(CUBES, f"blk_{b:03d}_{y}.npz")


def have(b, y):
    return os.path.exists(cube_path(b, y))


def load_cube(b, y):
    """(ndti, ndvi) float32 (14,640,640), NaN where not clear; starts. None if absent/unreadable."""
    p = cube_path(b, y)
    if not os.path.exists(p):
        return None
    try:
        with np.load(p) as d:
            ndti, ndvi, n = d["ndti"].astype(np.float32), d["ndvi"].astype(np.float32), d["nclr"]
            starts = d["starts"].astype(str)
    except Exception:
        return None
    clear = (n > 0) & np.isfinite(ndti) & np.isfinite(ndvi)
    return np.where(clear, ndti, np.nan), np.where(clear, ndvi, np.nan), starts


def crop_mask(b):
    return np.load(os.path.join(CUBES, f"wc_{b:03d}.npy")) == 40


def nanquantiles(s, qs, min_n):
    """Quantiles along axis 0 ignoring NaN (numpy 'linear' method); NaN where < min_n values."""
    k = np.isfinite(s).sum(0)
    srt = np.sort(s, 0)                                   # NaN sorted last
    out = []
    for q in qs:
        pos = q * np.maximum(k - 1, 0)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, np.maximum(k - 1, 0))
        vlo = np.take_along_axis(srt, lo[None], 0)[0]
        vhi = np.take_along_axis(srt, hi[None], 0)[0]
        out.append(np.where(k >= min_n, vlo + (pos - lo) * (vhi - vlo), np.nan))
    return np.stack(out)


def climatology(b, crop, min_years):
    """(3,14,H,W) per-pixel quantiles, (3,14) block-wide dekad quantiles, training years used."""
    stack, years = [], []
    for y in TRAIN:
        d = load_cube(b, y)
        if d is None:
            continue
        stack.append(np.stack([E.target(d[0], d[1], crop, t) for t in range(d[0].shape[0])]))
        years.append(y)
    if not stack:
        return None, None, []
    s = np.stack(stack)                                   # (Y,14,H,W)
    pix = nanquantiles(s, QS, min_years).astype(np.float32)
    fb = np.full((3, s.shape[1]), np.nan, np.float32)
    for t in range(s.shape[1]):
        v = s[:, t][np.isfinite(s[:, t])]
        if v.size:
            fb[:, t] = np.quantile(v, QS)
    return pix, fb, years


# ============================================================================ extraction (cached)
def deps_of(b, y, mode):
    """What a cache depends on besides the cube itself (so it is rebuilt when more cubes arrive)."""
    if mode == "sample":
        return {}
    return {"prev": have(b, y - 1), "clim_years": [t for t in TRAIN if have(b, t)]}


def cache_path(b, y, mode, per):
    tag = "full" if mode == "full" else f"s{per}"
    return os.path.join(CACHE, f"v{CACHE_VER}_{tag}_{b:03d}_{y}.npz")


def cache_ok(b, y, mode, per, min_years):
    p = cache_path(b, y, mode, per)
    if not os.path.exists(p):
        return False
    try:
        with np.load(p) as d:
            meta = json.loads(str(d["meta"]))
    except Exception:
        return False
    want = deps_of(b, y, mode)
    if mode == "full":
        want["clim_min_years"] = min_years
    return meta.get("deps") == want and os.path.getmtime(p) >= os.path.getmtime(cube_path(b, y))


def extract(b, y, mode, per, min_years, clim_cache=None):
    """Scored rows of one block-season. mode 'full' = every scored pixel (+ baselines needing other
    seasons); 'sample' = up to `per` random scored pixels per (issue, lead), seed (340, b, y)."""
    d = load_cube(b, y)
    if d is None:
        return None
    ndti, ndvi, starts = d
    crop = crop_mask(b)
    full = mode == "full"
    rng = np.random.default_rng([340, b, y])
    prev = load_cube(b, y - 1) if full else None
    if full:
        if clim_cache is not None and clim_cache.get("b") == b:
            pix, fb, years = clim_cache["v"]
        else:
            pix, fb, years = climatology(b, crop, min_years)
            if clim_cache is not None:
                clim_cache.update(b=b, v=(pix, fb, years))
    cols = {k: [] for k in ("X", "y", "issue", "lead", "png", "seas", "clim", "clim_src")}
    units, clear_rate = [], []
    for t in ISSUES:
        f, has = E.features(ndti, ndvi, t)
        Xt = np.stack([f[k].ravel() for k in FEATS], 1).astype(np.float32)
        win, vwin = ndti[t - E.T_IN + 1:t + 1], ndvi[t - E.T_IN + 1:t + 1]
        ng = np.isfinite(win) & (vwin < E.NDVI_MAX)
        idx = np.where(ng.any(0), E.T_IN - 1 - np.argmax(ng[::-1], 0), -1)
        png = np.where(idx >= 0, np.take_along_axis(win, np.clip(idx, 0, None)[None], 0)[0], np.nan).ravel()
        clear_rate.append(float(np.isfinite(win)[:, crop].mean()) if crop.any() else float("nan"))
        for L in LEADS:
            tg = E.target(ndti, ndvi, crop, t + L).ravel()
            ok = np.flatnonzero(has.ravel() & np.isfinite(tg))
            if not full and len(ok) > per:
                ok = np.sort(rng.choice(ok, per, replace=False))
            units.append((t, L, len(ok)))
            cols["X"].append(Xt[ok]); cols["y"].append(tg[ok])
            cols["issue"].append(np.full(len(ok), t, np.int8)); cols["lead"].append(np.full(len(ok), L, np.int8))
            cols["png"].append(png[ok].astype(np.float32))
            if full:
                if prev is not None:
                    s = E.target(prev[0], prev[1], crop, t + L).ravel()[ok]
                else:
                    s = np.full(len(ok), np.nan, np.float32)
                cols["seas"].append(s.astype(np.float32))
                c = np.full((len(ok), 3), np.nan, np.float32)
                src = np.full(len(ok), 2, np.int8)
                if pix is not None:
                    c = pix[:, t + L].reshape(3, -1)[:, ok].T.copy()
                    have_pix = np.isfinite(c).all(1)
                    src[have_pix] = 0
                    if np.isfinite(fb[:, t + L]).all():
                        c[~have_pix] = fb[:, t + L]
                        src[~have_pix] = 1
                cols["clim"].append(c); cols["clim_src"].append(src)
    out = {k: np.concatenate(v) for k, v in cols.items() if v}
    if full:
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(crop[None], ndvi, np.nan).reshape(ndvi.shape[0], -1), 1)
        peak = int(np.nanargmax(med)) if np.isfinite(med).any() else -1
        deps = deps_of(b, y, mode)
        deps["clim_min_years"] = min_years
    else:
        peak, deps = -1, {}
    meta = {"b": b, "y": y, "mode": mode, "per": per, "deps": deps, "starts": list(starts),
            "clear_rate": clear_rate, "ndvi_peak": peak, "wc_cropfrac": float(crop.mean()),
            "clim_years": years if full else [], "season_clear": float(np.isfinite(ndti)[:, crop].mean())}
    out["units"] = np.array(units, np.int64).reshape(-1, 3)
    out["meta"] = np.array(json.dumps(meta))
    return out


def ensure(b, y, mode, per, min_years, clim_cache=None):
    """Return the cached rows (building the cache if missing or stale), or None if no cube."""
    p = cache_path(b, y, mode, per)
    if not have(b, y):
        return None
    if not cache_ok(b, y, mode, per, min_years):
        out = extract(b, y, mode, per, min_years, clim_cache)
        if out is None:
            return None
        tmp = p[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, p)
    with np.load(p) as d:
        r = {k: d[k] for k in d.files}
    r["meta"] = json.loads(str(r["meta"]))
    return r


def _extract_job(args):
    b, years, mode, per, min_years = args
    cc = {}
    done = []
    for y in years:
        t0 = time.time()
        r = ensure(b, y, mode, per, min_years, cc)
        if r is not None:
            done.append((y, len(r["y"]), round(time.time() - t0, 1)))
    return b, mode, done


def extract_all(jobs, workers):
    t0 = time.time()
    if workers <= 1:
        res = [_extract_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_extract_job, jobs))
    for b, mode, done in res:
        print(f"  cache {mode:6s} blk {b}: " + ", ".join(f"{y}:{n} rows {s}s" for y, n, s in done), flush=True)
    print(f"extract: {time.time() - t0:.0f}s", flush=True)


# ============================================================================ weather
def _read_wx(b):
    """(era5 DataFrame or None, chirps Series or None, sources) for block b; v2 preferred."""
    era, chi, src = None, None, {}
    for dirn, name in ((WX2, f"era5_blk_{b}.json"), (WX1, f"blk_{b}.json")):
        p = os.path.join(dirn, name)
        if os.path.exists(p):
            try:
                d = json.load(open(p))["daily"]
                era = pd.DataFrame({k: pd.to_numeric(pd.Series(d[k]), errors="coerce")
                                    for k in ("precipitation_sum", "temperature_2m_mean",
                                              "soil_moisture_0_to_7cm_mean")})
                era.index = pd.to_datetime(d["time"])
                src["era5"] = os.path.basename(dirn)
                break
            except Exception:
                era = None
    for dirn in (WX2, WX1):
        p = os.path.join(dirn, f"chirps_blk_{b}.json")
        if os.path.exists(p):
            try:
                c = json.load(open(p))
                chi = pd.Series(pd.to_numeric(pd.Series(c["precip_mm_blockmean"]), errors="coerce").values,
                                index=pd.to_datetime(c["time"]))
                src["chirps"] = os.path.basename(dirn)
                break
            except Exception:
                chi = None
    return era, chi, src


_WX = {}


def weather(b, starts):
    """(14, 6) weather columns WCOLS for each issue dekad (NaN where a series does not cover >= 90%
    of the window's days). Window = input dekads t-5..t = [starts[t-5], starts[t+1])."""
    if b not in _WX:
        _WX[b] = _read_wx(b)
    era, chi, _ = _WX[b]
    st = pd.to_datetime(list(starts))
    out = np.full((len(st), len(WCOLS)), np.nan)

    def agg(s, a, z, how):
        if s is None:
            return np.nan
        w = s[(s.index >= a) & (s.index < z)]
        need = (z - a).days
        if need <= 0 or w.notna().sum() < 0.9 * need:
            return np.nan
        return float(w.sum() / w.notna().sum() * need) if how == "sum" else float(w.mean())

    for t in ISSUES:
        a6, a1, z = st[t - E.T_IN + 1], st[t], st[t + 1]
        if era is not None:
            out[t, 0] = agg(era.precipitation_sum, a6, z, "sum")
            out[t, 1] = agg(era.precipitation_sum, a1, z, "sum")
            out[t, 2] = agg(era.soil_moisture_0_to_7cm_mean, a1, z, "mean")
            out[t, 3] = agg(era.temperature_2m_mean, a1, z, "mean")
        out[t, 4] = agg(chi, a6, z, "sum")
        out[t, 5] = agg(chi, a1, z, "sum")
    return out


def wx_rows(r):
    W = weather(r["meta"]["b"], r["meta"]["starts"])
    return W[r["issue"].astype(int)]


def wx_sig(b, starts):
    """Content signature of a block-season's issue-dekad weather (refit / rescore if the files change)."""
    return hashlib.sha1(np.round(weather(b, starts)[list(ISSUES)], 4).tobytes()).hexdigest()[:12]


def wx_names_for(models, b, starts, units):
    """Weather variants score_bs will emit: judged on the issues that have scored rows (as score_bs does)."""
    iss = sorted({int(t) for t, _, n in units if n > 0})
    return variants_available(models, weather(b, starts)[iss]) if iss else []


# ============================================================================ design matrices
def gbm_X(r, W, extra):
    base = np.hstack([r["X"], r["lead"][:, None].astype(np.float32)])
    if not extra:
        return base
    return np.hstack([base, W[:, [WCOLS.index(c) for c in extra]].astype(np.float32)])


def ridge_prep_fit(X, lead):
    """Median fill + missing flags + lead one-hot, standardised with training statistics (plain dict,
    so models.pkl holds only numpy/sklearn objects)."""
    pp = {"med": np.nanmedian(X, 0), "flagcols": (~np.isfinite(X)).any(0)}
    Z = _ridge_raw(pp, X, lead)
    pp["mu"], pp["sd"] = Z.mean(0), Z.std(0) + 1e-9
    return pp


def _ridge_raw(pp, X, lead):
    miss = ~np.isfinite(X)
    oh = np.stack([lead == L for L in LEADS], 1).astype(np.float32)
    return np.hstack([np.where(miss, pp["med"], X), miss[:, pp["flagcols"]].astype(np.float32), oh])


def ridge_prep(pp, X, lead):
    return (_ridge_raw(pp, X, lead) - pp["mu"]) / pp["sd"]


def pinball(y, q, tau):
    e = y - q
    return np.maximum(tau * e, (tau - 1) * e)


def stack_rows(rs):
    keys = ("X", "y", "lead", "issue")
    out = {k: np.concatenate([r[k] for r in rs]) for k in keys}
    out["W"] = np.concatenate([wx_rows(r) for r in rs])
    return out


# ============================================================================ fitting
def fit_gbm(Xtr, ytr, Xva, yva, tau, lrs, cap, leaf, minleaf):
    """Fit one quantile GBM per learning rate (cap iterations), pick (lr, iterations) by validation
    pinball over the staged predictions, truncate the chosen model to that many iterations."""
    best, curves = None, {}
    for lr in lrs:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=tau, learning_rate=lr, max_iter=cap,
                                          max_leaf_nodes=leaf, min_samples_leaf=minleaf,
                                          early_stopping=False, random_state=340)
        m.fit(Xtr, ytr)
        loss = np.array([pinball(yva, p, tau).mean() for p in m.staged_predict(Xva)])
        k = int(np.argmin(loss)) + 1
        curves[str(lr)] = {"best_iter": k, "val_pinball": float(loss[k - 1]),
                           "val_pinball_at_cap": float(loss[-1])}
        if best is None or loss[k - 1] < best[0]:
            best = (float(loss[k - 1]), lr, k, m, loss)
    loss, lr, k, m, curve = best
    m._predictors = m._predictors[:k]                     # n_iter_ is len(_predictors)
    assert np.isclose(pinball(yva, m.predict(Xva), tau).mean(), loss, rtol=1e-6, atol=1e-9), "truncation"
    return m, {"lr": lr, "iters": k, "val_pinball": loss, "grid": curves}


def fit_all(train_rows, val_rows, a):
    tr, va = stack_rows(train_rows), stack_rows(val_rows)
    info = {"train_rows": int(len(tr["y"])), "val_rows": int(len(va["y"]))}
    models = {}
    t0 = time.time()
    # ridge (point); alpha by validation MAE
    prep = ridge_prep_fit(tr["X"], tr["lead"])
    Ztr, Zva = ridge_prep(prep, tr["X"], tr["lead"]), ridge_prep(prep, va["X"], va["lead"])
    best = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        m = Ridge(alpha=alpha).fit(Ztr, tr["y"])
        mae = float(np.abs(m.predict(Zva) - va["y"]).mean())
        if best is None or mae < best[0]:
            best = (mae, alpha, m)
    models["ridge"] = (prep, best[2])
    info["ridge"] = {"alpha": best[1], "val_mae": best[0]}
    print(f"  ridge alpha={best[1]} val MAE {best[0]:.4f}", flush=True)
    # regional climatology fallback: pooled training target quantiles per target dekad
    tdek = (tr["issue"] + tr["lead"]).astype(int)
    models["clim_regional"] = {int(d): np.quantile(tr["y"][tdek == d], QS).astype(np.float32)
                               for d in np.unique(tdek)}
    # gbm_base: (lr, iterations) per quantile
    models["gbm_base"], info["gbm_base"] = {}, {}
    for tau in QS:
        t1 = time.time()
        m, inf = fit_gbm(gbm_X(tr, tr["W"], []), tr["y"], gbm_X(va, va["W"], []), va["y"], tau,
                         a.lrs, a.cap, a.leaf, a.minleaf)
        inf["fit_s"] = round(time.time() - t1, 1)
        models["gbm_base"][tau], info["gbm_base"][str(tau)] = m, inf
        print(f"  gbm_base q{tau}: lr={inf['lr']} iters={inf['iters']} val pinball {inf['val_pinball']:.5f} "
              f"({inf['fit_s']}s)", flush=True)
    # weather variants: same learning rate as gbm_base for that quantile; iterations by val pinball.
    # Rows lacking the variant's weather are dropped from its training/validation (coverage logged).
    for name, extra in WSETS.items():
        ci = [WCOLS.index(c) for c in extra]
        oktr = np.isfinite(tr["W"][:, ci]).all(1)
        okva = np.isfinite(va["W"][:, ci]).all(1)
        info[name] = {"train_cover": float(oktr.mean()), "val_cover": float(okva.mean())}
        if oktr.sum() < 1000 or okva.sum() < 200:
            print(f"  {name}: skipped (weather rows train {oktr.sum()}, val {okva.sum()})", flush=True)
            info[name]["skipped"] = True
            continue
        models[name] = {}
        for tau in QS:
            t1 = time.time()
            lr = info["gbm_base"][str(tau)]["lr"]
            m, inf = fit_gbm(gbm_X(tr, tr["W"], extra)[oktr], tr["y"][oktr],
                             gbm_X(va, va["W"], extra)[okva], va["y"][okva], tau,
                             (lr,), a.cap, a.leaf, a.minleaf)
            inf["fit_s"] = round(time.time() - t1, 1)
            models[name][tau] = m
            info[name][str(tau)] = inf
            print(f"  {name} q{tau}: iters={inf['iters']} val pinball {inf['val_pinball']:.5f} "
                  f"(train cover {oktr.mean():.2f}, {inf['fit_s']}s)", flush=True)
    info["fit_seconds"] = round(time.time() - t0, 1)
    return models, info


# ============================================================================ scoring
def variants_available(models, W):
    """Weather variants that can be scored on every row of a block-season (all their columns finite)."""
    if len(W) == 0:
        return []
    return [n for n in WSETS if n in models and np.isfinite(W[:, [WCOLS.index(c) for c in WSETS[n]]]).all()]


def predict_all(r, models, W, wx_names):
    """{model: (N,3) P10/P50/P90} over exactly the cached rows of one block-season, plus fallback
    counts per row (bool arrays)."""
    X, y, lead, issue = r["X"], r["y"], r["lead"], r["issue"]
    N = len(y)
    pers = X[:, FEATS.index("ndti_last")]
    assert np.isfinite(pers).all(), "persistence must exist on every scored pixel (>=1 clear input)"
    rep = lambda v: np.repeat(v[:, None], 3, 1)
    P, FB = {}, {}
    P["persistence"], FB["persistence"] = rep(pers), np.zeros(N, bool)
    fb = ~np.isfinite(r["png"])
    P["persistence_ng"], FB["persistence_ng"] = rep(np.where(fb, pers, r["png"])), fb
    P["movavg6"], FB["movavg6"] = rep(X[:, FEATS.index("ndti_mean")]), np.zeros(N, bool)
    fb = ~np.isfinite(r["seas"])
    P["seasonal"], FB["seasonal"] = rep(np.where(fb, pers, r["seas"])), fb
    c = r["clim"].copy()
    none = r["clim_src"] == 2
    if none.any():
        tdek = (issue + lead).astype(int)
        c[none] = np.stack([models["clim_regional"][int(d)] for d in tdek[none]])
    P["climatology"], FB["climatology"] = np.sort(c, 1), r["clim_src"] > 0
    prep, ridge = models["ridge"]
    P["ridge"], FB["ridge"] = rep(ridge.predict(ridge_prep(prep, X, lead)).astype(np.float32)), np.zeros(N, bool)
    names = ["gbm_base"] + list(wx_names)
    for name in names:
        extra = WSETS.get(name, [])
        Xg = np.ascontiguousarray(gbm_X(r, W, extra), dtype=np.float64)   # avoid per-model casts
        P[name] = np.sort(np.stack([models[name][tau].predict(Xg) for tau in QS], 1), 1).astype(np.float32)
        FB[name] = np.zeros(N, bool)
    for k, p in P.items():
        assert p.shape == (N, 3) and np.isfinite(p).all(), f"{k}: non-finite prediction"
    return P, FB


def subsample(r, max_px, b, y):
    """Optional identical subsample for every model: up to max_px random rows per unit, seed (340, b, y)."""
    units = r["units"]
    if max_px <= 0 or (units[:, 2] <= max_px).all():
        return r
    rng = np.random.default_rng([340, b, y, 1])
    bounds = np.concatenate([[0], np.cumsum(units[:, 2])])
    keep, nu = [], units.copy()
    for j, (t, L, n) in enumerate(units):
        idx = np.arange(bounds[j], bounds[j + 1])
        if n > max_px:
            idx = np.sort(rng.choice(idx, max_px, replace=False))
        keep.append(idx)
        nu[j, 2] = len(idx)
    keep = np.concatenate(keep)
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == bounds[-1] else v)
           for k, v in r.items()}
    out["units"] = nu
    return out


def score_bs(b, y, r, models, plan, max_px=0):
    r = subsample(r, max_px, b, y)
    meta = r["meta"]
    W = wx_rows(r)
    wx_names = variants_available(models, W)
    P, FB = predict_all(r, models, W, wx_names)
    yv = r["y"]
    units = r["units"]
    bounds = np.concatenate([[0], np.cumsum(units[:, 2])])
    nonempty = units[:, 2] > 0
    starts = bounds[:-1][nonempty]
    green = (r["X"][:, FEATS.index("ndvi_last")] >= E.NDVI_MAX).astype(np.int64)
    seg = lambda v: np.add.reduceat(v, starts) if len(starts) else np.zeros(0)
    lastgreen = seg(green)
    out = []
    for name, p in P.items():
        e = p[:, 1] - yv
        pin = (pinball(yv, p[:, 0], .1) + pinball(yv, p[:, 1], .5) + pinball(yv, p[:, 2], .9)) / 3
        stats = {"sae": seg(np.abs(e)), "sse": seg(e ** 2), "se": seg(e), "spin": seg(pin),
                 "cover": seg(((yv >= p[:, 0]) & (yv <= p[:, 2])).astype(np.int64)),
                 "below": seg((yv < p[:, 0]).astype(np.int64)), "nfb": seg(FB[name].astype(np.int64))}
        for j, (t, L, n) in enumerate(units[nonempty]):
            out.append({"model": name, "block": b, "season": y, "issue": int(t), "lead": int(L), "n": int(n),
                        "sae": float(stats["sae"][j]), "sse": float(stats["sse"][j]), "se": float(stats["se"][j]),
                        "spin": float(stats["spin"][j]), "cover": int(stats["cover"][j]),
                        "below": int(stats["below"][j]), "nfb": int(stats["nfb"][j]),
                        "lastgreen": int(lastgreen[j]),
                        "clear_rate": round(meta["clear_rate"][ISSUES.index(int(t))], 4),
                        "cropfrac": plan["cropfrac"].get(str(b), np.nan),
                        "wc_cropfrac": round(meta["wc_cropfrac"], 4),
                        "region": plan["division"].get(str(b), ""), "superblock": superblock(b),
                        "ndvi_peak": meta["ndvi_peak"], "season_clear": round(meta["season_clear"], 4),
                        "clim_years": len(meta["clim_years"]), "seas_prev": int(meta["deps"].get("prev", False))})
    # identical pixels: every model in a unit carries the same n (rows are shared by construction)
    if out:
        df = pd.DataFrame(out)
        assert (df.groupby(["issue", "lead"]).n.nunique() == 1).all()
        assert (df.groupby(["issue", "lead"]).model.nunique() == len(P)).all()
    return out, wx_names


# ============================================================================ main
def fingerprint(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def roles(plan, quick, max_blocks):
    """(train [(b,y)], val [(b,y)], test [(b,y)], pseudo flag)."""
    tr = [(b, y) for b in plan["train"] for y in TRAIN if have(b, y)]
    va = [(b, y) for b in plan["val"] for y in VAL if have(b, y)]
    te = [(b, y) for b in plan["test"] for y in TEST if have(b, y)]
    if not quick:
        return tr, va, te, False
    if tr and va and te:
        tb = sorted({b for b, _ in tr})[:max_blocks]
        vb = sorted({b for b, _ in va})[:max(1, max_blocks // 2)]
        sb = sorted({b for b, _ in te})[:max_blocks]
        return ([p for p in tr if p[0] in tb], [p for p in va if p[0] in vb],
                [p for p in te if p[0] in sb], False)
    # Not enough cubes yet for the real split: pseudo split of whatever exists (smoke test only).
    avail = sorted({(b, y) for b in plan["test"] + plan["train"] + plan["val"]
                    for y in TRAIN + VAL + TEST if have(b, y)})
    blocks = sorted({b for b, _ in avail})[:max(3, 2 * max_blocks)]
    nb = len(blocks)
    trb, vab, teb = blocks[:max(1, nb // 2)], blocks[max(1, nb // 2):max(1, nb // 2) + 1], blocks[max(1, nb // 2) + 1:]
    pick = lambda bs: [p for p in avail if p[0] in bs]
    return pick(trb), pick(vab), pick(teb), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: few blocks, small GBM")
    ap.add_argument("--max-blocks", type=int, default=3)
    ap.add_argument("--stage", choices=["extract", "fit", "score", "all"], default="all")
    ap.add_argument("--per", type=int, default=500, help="training rows sampled per block-season-issue-lead")
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--cap", type=int, default=800, help="max GBM iterations (best chosen on val)")
    ap.add_argument("--leaf", type=int, default=31)
    ap.add_argument("--minleaf", type=int, default=200)
    ap.add_argument("--clim-min-years", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-px", type=int, default=0,
                    help="score at most this many random pixels per unit (same pixels for every model); 0 = all")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.quick:
        a.per, a.lrs, a.cap = min(a.per, 200), [0.1], min(a.cap, 150)
    out_dir = a.out or os.path.join(ROOT, "results_v2", "final_quick" if a.quick else "final")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    plan = json.load(open(PLAN))
    T0 = time.time()
    tr, va, te, pseudo = roles(plan, a.quick, a.max_blocks)
    print(f"block-seasons with cubes: train {len(tr)}, val {len(va)}, test {len(te)}"
          + ("  [PSEUDO SPLIT - smoke test only]" if pseudo else ""), flush=True)
    if not (tr and va and te):
        raise SystemExit("need train, val and test cubes")
    # Cubes the protocol expects but that are absent (download unfinished or failed). A missing test-block
    # 2018-2022 season weakens its climatology and a missing 2023 cube removes seasonal persistence for
    # 2024, silently, via fallbacks -- so say so loudly and record it in fit.json.
    expect = ([(b, y) for b in plan["train"] for y in TRAIN] + [(b, y) for b in plan["val"] for y in VAL]
              + [(b, y) for b in plan["test"] for y in TRAIN + VAL + TEST])
    missing = [p for p in expect if not have(*p)]
    if missing and not a.quick:
        print(f"WARNING: {len(missing)} expected cubes are missing (first: {missing[:8]}); results will use "
              f"fallbacks for them. Re-run after the download finishes (caches/models/parts rebuild "
              f"automatically).", flush=True)

    # ---- extract (parallel, cached)
    group = lambda pairs: sorted({b: None for b, _ in pairs})
    jobs = [(b, sorted(y for bb, y in tr + va if bb == b), "sample", a.per, a.clim_min_years)
            for b in group(tr + va)]
    jobs += [(b, sorted(y for bb, y in te if bb == b), "full", a.per, a.clim_min_years) for b in group(te)]
    t1 = time.time()
    extract_all(jobs, a.workers)
    t_extract = time.time() - t1
    if a.stage == "extract":
        return

    # ---- fit (cached by fingerprint of the training set + settings + weather coverage)
    # keep (b, y) aligned with the loaded rows (an unreadable cube must not shift the pairing)
    def load_pairs(pairs):
        got = [(p, ensure(*p, "sample", a.per, a.clim_min_years)) for p in pairs]
        return [p for p, r in got if r is not None], [r for _, r in got if r is not None]
    tr, train_rows = load_pairs(tr)
    va, val_rows = load_pairs(va)
    wx_cov = {f"{b}_{y}": bool(np.isfinite(weather(b, r["meta"]["starts"])[list(ISSUES)]).all())
              for (b, y), r in zip(tr + va, train_rows + val_rows)}
    # the fingerprint covers weather *content* and source, not just coverage, so a rewritten weather
    # file forces a refit instead of silently pairing old models with new inputs
    wx_fp = {f"{b}_{y}": [wx_sig(b, r["meta"]["starts"]), _WX[b][2]]
             for (b, y), r in zip(tr + va, train_rows + val_rows)}
    fp = fingerprint({"tr": tr, "va": va, "per": a.per, "lrs": a.lrs, "cap": a.cap, "leaf": a.leaf,
                      "minleaf": a.minleaf, "wx": wx_cov, "wxsig": wx_fp, "ver": CACHE_VER})
    mpath = os.path.join(out_dir, "models.pkl")
    models = info = None
    if os.path.exists(mpath) and not a.refit:
        with open(mpath, "rb") as fh:
            saved = pickle.load(fh)
        if saved["fp"] == fp:
            models, info = saved["models"], saved["info"]
            print(f"fit: reusing models.pkl ({fp})", flush=True)
    t1 = time.time()
    if models is None:
        models, info = fit_all(train_rows, val_rows, a)
        with open(mpath + ".tmp", "wb") as fh:
            pickle.dump({"fp": fp, "models": models, "info": info}, fh)
        os.replace(mpath + ".tmp", mpath)
    t_fit = time.time() - t1
    del train_rows, val_rows
    if a.stage == "fit":
        return

    # ---- score (resumable per block-season; parts keyed by model fingerprint)
    pdir = os.path.join(out_dir, "parts", f"{fp}_px{a.max_px}")
    os.makedirs(pdir, exist_ok=True)
    t1 = time.time()
    rows, timing = [], []
    for b, y in te:
        part = os.path.join(pdir, f"{b:03d}_{y}.csv")
        side = part[:-4] + ".json"
        cp = cache_path(b, y, "full", a.per)
        if (os.path.exists(part) and os.path.exists(side) and cache_ok(b, y, "full", a.per, a.clim_min_years)
                and os.path.getmtime(part) >= os.path.getmtime(cp)):
            with np.load(cp) as d:
                starts, units = json.loads(str(d["meta"]))["starts"], d["units"]
            sd = json.load(open(side))
            if sd["wx"] == wx_names_for(models, b, starts, units) and sd.get("wxsig") == wx_sig(b, starts):
                if sd.get("n_out", 1) > 0:
                    rows += pd.read_csv(part).to_dict("records")
                continue
        t2 = time.time()
        r = ensure(b, y, "full", a.per, a.clim_min_years)
        rr, wx_names = score_bs(b, y, r, models, plan, a.max_px)
        pd.DataFrame(rr).to_csv(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)
        json.dump({"wx": wx_names, "wxsig": wx_sig(b, r["meta"]["starts"]), "n_rows": int(len(r["y"])),
                   "n_out": len(rr)}, open(side, "w"))
        rows += rr
        nsc = sum(x["n"] for x in rr if x["model"] == "persistence")
        timing.append((nsc, time.time() - t2))
        print(f"  scored blk {b} {y}: {nsc} of {len(r['y'])} pixel-rows, wx={len(wx_names)}/{len(WSETS)}, "
              f"{time.time() - t2:.1f}s", flush=True)
    t_score = time.time() - t1
    df = pd.DataFrame(rows)
    df.insert(1, "split", "test")
    df.to_csv(os.path.join(out_dir, "rows.csv"), index=False)
    fbs = df.groupby("model").apply(lambda g: g.nfb.sum() / g.n.sum(), include_groups=False)
    fit_json = {"fingerprint": fp, "pseudo_split": pseudo, "quick": a.quick,
                "settings": {k: v for k, v in vars(a).items()}, "features": FEATS + ["lead"],
                "weather_columns": WCOLS, "weather_sets": WSETS, "models": MODELS,
                "train_block_seasons": tr, "val_block_seasons": va, "test_block_seasons": te,
                "weather_sources": {str(b): _WX[b][2] for b in sorted(_WX)},
                "train_val_weather_complete": wx_cov, "missing_cubes": missing,
                "fit": info, "fallback_share": {k: float(v) for k, v in fbs.items()},
                "timing_s": {"extract": round(t_extract, 1), "fit": round(t_fit, 1), "score": round(t_score, 1),
                             "total": round(time.time() - T0, 1),
                             "score_rows_per_s": (sum(n for n, _ in timing) / max(1e-9, sum(s for _, s in timing)))
                             if timing else None}}
    json.dump(fit_json, open(os.path.join(out_dir, "fit.json"), "w"), indent=1, default=str)
    print(f"rows.csv: {len(df)} rows; models {sorted(df.model.unique())}", flush=True)
    print(f"timing: extract {t_extract:.0f}s, fit {t_fit:.0f}s, score {t_score:.0f}s, total {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
