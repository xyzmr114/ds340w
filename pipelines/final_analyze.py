"""Analysis of final_eval.py rows (EXPERIMENT_PROTOCOL.md statistics and error analysis).

  * pooled metrics per model, and by lead, season, region, clear-observation tercile, cropland tercile
    (MAE, RMSE, bias, mean pinball over P10/P50/P90, 80% interval coverage, share below P10)
  * block-level comparisons vs persistence, climatology and gbm_base: win rate, paired Wilcoxon
    signed-rank on per-block MAE and pinball, mean block difference with a 95% bootstrap CI that
    resamples super-blocks (2000 draws, seed 340)
  * weather ablation: change in block pinball / MAE of each gbm weather variant vs gbm_base
  * regional correlates: Spearman of block error vs clear rate, cropland fraction, NDVI-peak dekad,
    and the share of forecasts whose last clear input was green (NDVI >= 0.30)

Every comparison uses only the units (block, season, issue, lead) where all compared models are
present, so pixels are identical (final_eval.py shares rows within a unit).

Neural models (optional, <dir>/neural_rows.csv from final_neural.py, same schema, model "<arch>_s<seed>"):
  * a run is complete when it is listed in <dir>/neural_runs.json AND its rows cover every core unit;
    incomplete runs are skipped (printed note) but their rows still enter the one-n-per-unit check
  * seed aggregate per arch ("convlstm", "earthformer"): on units where every complete seed is present,
    sae/sse/se/spin/cover/below/nfb/lastgreen are averaged over seeds (n and covariates copied). Pooled
    MAE, pinball, bias, coverage and below-P10 of the aggregate therefore equal the mean over seeds, while
    the aggregate RMSE is sqrt(mean over seeds of MSE), not the mean of the seed RMSEs. It is NOT the
    error of an averaged (ensemble) forecast. Neural outputs are P10/P50/P90, so they are not in POINT.
  * the baseline tables above are untouched; each aggregate gets its own tables (*_neural) on
    common(core models + arch), with gbm_base / persistence on those same units (neural_same_units)
  * block tests: aggregate vs persistence / climatology / gbm_base, earthformer vs convlstm, and each
    complete seed vs gbm_base and persistence (neural_seed_tests); seed spread in neural_runs
Without neural_rows.csv every output is byte-identical to the baseline-only version.

Usage: python final_analyze.py [--dir ../results_v2/final]
Writes <dir>/summary.json, <dir>/final_numbers.tex, ../results_v2/figures/fig_final_results.{pdf,png}
"""
import argparse
import io
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UNIT = ["block", "season", "issue", "lead"]
ORDER = ["persistence", "persistence_ng", "movavg6", "seasonal", "climatology", "ridge", "gbm_base",
         "gbm_rain", "gbm_soil", "gbm_temp", "gbm_chirps", "gbm_all", "convlstm", "earthformer"]
KEY = {"persistence": "Pers", "persistence_ng": "PersNG", "movavg6": "MovAvg", "seasonal": "Seas",
       "climatology": "Clim", "ridge": "Ridge", "gbm_base": "Gbm", "gbm_rain": "GbmRain", "gbm_soil": "GbmSoil",
       "gbm_temp": "GbmTemp", "gbm_chirps": "GbmChirps", "gbm_all": "GbmAll", "convlstm": "ConvLSTM",
       "earthformer": "Earthformer"}
LABEL = {"persistence": "Persistence", "persistence_ng": "Non-green persistence", "movavg6": "6-dekad mean",
         "seasonal": "Seasonal persistence", "climatology": "Climatology", "ridge": "Ridge",
         "gbm_base": "GBM", "gbm_rain": "GBM + ERA5 rain", "gbm_soil": "GBM + soil moisture",
         "gbm_temp": "GBM + temperature", "gbm_chirps": "GBM + CHIRPS rain", "gbm_all": "GBM + all weather",
         "convlstm": "ConvLSTM", "earthformer": "Earthformer"}
NEURAL = ("convlstm", "earthformer")                  # seed aggregates; per-seed runs "<arch>_s<seed>"
SEEDWORD = {340: "SeedOne", 341: "SeedTwo", 342: "SeedThree"}
MET = ["sae", "sse", "se", "spin", "cover", "below", "nfb", "lastgreen"]   # summed per unit, averaged over seeds
NB, SEED = 2000, 340
# point forecasts (P10 = P50 = P90): their "80% coverage" only counts exact ties and their "below P10" is
# the share below the point forecast, so interval metrics are reported as NaN ("--") for them; their
# pinball loss is the degenerate-quantile value (= MAE / 2), which is the standard comparison.
POINT = {"persistence", "persistence_ng", "movavg6", "seasonal", "ridge"}


def metrics(g, model=None):
    n = g.n.sum()
    pt = model in POINT
    return {"n": int(n), "mae": g.sae.sum() / n, "rmse": (g.sse.sum() / n) ** .5, "bias": g.se.sum() / n,
            "pinball": g.spin.sum() / n, "cover80": float("nan") if pt else g.cover.sum() / n,
            "below10": float("nan") if pt else g.below.sum() / n, "fallback": g.nfb.sum() / n}


def table(df, by):
    out = {}
    for keys, g in df.groupby(["model"] + by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        d = out.setdefault(keys[0], {})
        for k in keys[1:-1]:
            d = d.setdefault(str(k), {})
        d[str(keys[-1]) if by else "all"] = metrics(g, keys[0])
    return out


def common(df, models):
    """Rows of `models` restricted to units where all of them are present."""
    d = df[df.model.isin(models)]
    cnt = d.groupby(UNIT).model.nunique()
    keep = cnt[cnt == len(models)].index
    return d.set_index(UNIT).loc[keep].reset_index()


def block_metric(d, metric):
    g = d.groupby(["model", "block"])
    s = g.sae.sum() if metric == "mae" else g.spin.sum()
    return (s / g.n.sum()).unstack(0)


def boot_ci(diff, sb, rng):
    """95% CI of the mean block difference, resampling super-blocks with replacement."""
    groups = [diff[sb == s] for s in np.unique(sb)]
    k = len(groups)
    means = np.empty(NB)
    for i in range(NB):
        pick = rng.integers(0, k, k)
        means[i] = np.concatenate([groups[j] for j in pick]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(df, a, b, sbmap):
    d = common(df, [a, b])
    if d.empty:
        return None
    out = {"units": int(d.groupby(UNIT).ngroups)}
    for metric in ("mae", "pinball"):
        bm = block_metric(d, metric)
        diff = (bm[a] - bm[b]).values
        sb = np.array([sbmap[x] for x in bm.index])
        nz = diff[diff != 0]
        p = float(wilcoxon(diff).pvalue) if len(nz) >= 1 and len(diff) >= 2 else float("nan")
        lo, hi = boot_ci(diff, sb, np.random.default_rng(SEED))
        out[metric] = {"blocks": int(len(diff)), "wins": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
                       "win_rate": float((diff < 0).mean()), "mean_diff": float(diff.mean()),
                       "median_diff": float(np.median(diff)), "ci95": [lo, hi], "wilcoxon_p": p,
                       "rel_diff": float(diff.mean() / bm[b].mean())}
    return out


def check_pixels(df):
    """identical pixels: one n per unit across every model (baselines and neural runs)"""
    k = df.groupby(UNIT).n.nunique()
    if (k == 1).all():
        return
    d = df.set_index(UNIT).loc[k[k > 1].index].reset_index()
    d = d.merge(d.groupby(UNIT).n.agg(lambda s: s.mode().iloc[0]).rename("n_ref").reset_index(), on=UNIT)
    d = d[d.n != d.n_ref]
    msg = "; ".join(f"{m}: {len(g)} units differ from the majority n, e.g. (block, season, issue, lead) = "
                    f"{tuple(int(x) for x in g[UNIT].iloc[0])} n={int(g.n.iloc[0])} vs {int(g.n_ref.iloc[0])}"
                    for m, g in d.groupby("model"))
    raise AssertionError(f"models scored on different pixels ({int((k > 1).sum())} units): {msg}")


def _slurp(path):
    with open(path, "rb") as fh:
        return fh.read()


def load_neural(dirpath, core):
    """(test rows of neural_rows.csv or None, neural_runs.json, {arch: [complete seeds]}, skipped runs)"""
    path = os.path.join(dirpath, "neural_rows.csv")
    if not os.path.exists(path):
        return None, {}, {}, []
    # final_neural.py replaces these two files with os.replace, which fails on Windows while another process
    # holds them open (even with FILE_SHARE_DELETE). Read each in one call and close at once to keep that
    # window to a single read; while training is still running, prefer --dir on a copy.
    nn = pd.read_csv(io.BytesIO(_slurp(path)))
    nn = nn[nn.split == "test"]
    dup = nn.duplicated(["model"] + UNIT)
    assert not dup.any(), f"neural_rows.csv repeats units: {nn[dup].model.unique().tolist()}"
    num_ok = nn[["n"] + MET].apply(pd.to_numeric, errors="coerce")
    bad = num_ok.isna().any(axis=1)
    assert not bad.any(), (f"neural_rows.csv has incomplete or non-numeric rows (file being written?): "
                           f"{nn[bad].groupby('model').size().to_dict()} rows, e.g. "
                           f"{nn.loc[bad, ['model'] + UNIT].head(5).to_dict('records')}")
    nn[["n"] + MET] = num_ok
    jp = os.path.join(dirpath, "neural_runs.json")
    runs = json.loads(_slurp(jp)) if os.path.exists(jp) else {}
    cu = set(core.groupby(UNIT).size().index)
    use, skip = {}, []
    for run in sorted(set(nn.model)):
        arch, _, s = run.rpartition("_s")
        if arch not in NEURAL or not s.isdigit():
            print(f"note: neural model {run} is not <arch>_s<seed> with arch in {NEURAL}; ignored")
            skip.append(run); continue
        miss = len(cu - set(nn.loc[nn.model == run, UNIT].itertuples(index=False, name=None)))
        if run not in runs or miss:
            why = "not in neural_runs.json" if run not in runs else f"{miss} of {len(cu)} core units missing"
            print(f"note: neural run {run} incomplete ({why}); left out of the aggregate and seed tables")
            skip.append(run); continue
        use.setdefault(arch, []).append(int(s))
    return nn, runs, use, skip


def seed_mean(nn, arch, seeds):
    """seed-aggregate rows: MET averaged over seeds on units where every seed is present"""
    runs = [f"{arch}_s{s}" for s in seeds]
    d = nn[nn.model.isin(runs)]
    cnt = d.groupby(UNIT).model.nunique()
    d = d.set_index(UNIT).loc[cnt[cnt == len(runs)].index]
    out = d[d.model == runs[0]].drop(columns=MET).join(d.groupby(UNIT)[MET].mean())
    return out.assign(model=arch).reset_index()[list(nn.columns)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results_v2", "final"))
    ap.add_argument("--figdir", default=os.path.join(ROOT, "results_v2", "figures"))
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.dir, "rows.csv"))
    df = df[df.split == "test"]
    fit = json.load(open(os.path.join(a.dir, "fit.json")))
    models = [m for m in ORDER if m in set(df.model)]
    sbmap = df.groupby("block").superblock.first().to_dict()
    core_models = [m for m in models if not m.startswith("gbm_") or m == "gbm_base"]
    allc = common(df, models)                     # units where every model (incl. weather) exists
    core = common(df, core_models)
    nn, runs, use, skipped = load_neural(a.dir, core)
    dfa = df if nn is None else pd.concat([df, nn], ignore_index=True)   # baselines + per-seed neural runs
    # identical pixels: one n per unit across every model (baselines and every neural run)
    check_pixels(dfa)
    archs = [m for m in NEURAL if m in use]
    if archs:
        dfa = pd.concat([dfa] + [seed_mean(nn, m, use[m]) for m in archs], ignore_index=True)

    S = {"source_fit": {k: fit.get(k) for k in ("fingerprint", "pseudo_split", "quick", "fallback_share")},
         "counts": {"test_blocks": int(df.block.nunique()), "superblocks": int(df.superblock.nunique()),
                    "units_core": int(core.groupby(UNIT).ngroups), "units_all_models": int(allc.groupby(UNIT).ngroups),
                    "pixel_forecasts": int(core[core.model == "persistence"].n.sum()),
                    "train_rows": fit["fit"].get("train_rows"), "val_rows": fit["fit"].get("val_rows"),
                    "seasons": sorted(int(s) for s in df.season.unique()),
                    "regions": sorted(df.region.dropna().unique().tolist())}}
    # ---- pooled tables (core units: every non-weather model; weather variants on the all-models units)
    core = core.copy()
    S["pooled"] = {m: v["all"] for m, v in table(core, []).items()}
    S["pooled_all_models_units"] = {m: v["all"] for m, v in table(allc, []).items()}
    S["by_lead"] = table(core, ["lead"])
    S["by_season"] = table(core, ["season"])
    S["by_region"] = table(core, ["region"])
    S["by_superblock"] = table(core, ["superblock"])          # protocol error analysis: region and super-block
    S["by_lead_all_models_units"] = table(allc, ["lead"])
    # pooled MAE relative to persistence on the same pixels (MASE-style skill with a test-period naive scale;
    # the protocol's in-sample-naive MASE needs training-period errors that rows.csv does not hold)
    pm = S["pooled"]["persistence"]["mae"]
    S["rel_mae_vs_persistence"] = {m: v["mae"] / pm for m, v in S["pooled"].items()}
    S["rel_mae_vs_persistence"] |= {m: v["mae"] / S["pooled_all_models_units"]["persistence"]["mae"]
                                    for m, v in S["pooled_all_models_units"].items() if m not in S["pooled"]}
    # clear-observation and cropland terciles (unit-level clear rate; block-level cropland fraction)
    u = core.groupby(UNIT).clear_rate.first()
    cuts = np.quantile(u.values, [1 / 3, 2 / 3])
    core["clear_tercile"] = np.digitize(core.clear_rate, cuts)
    bc = core.groupby("block").cropfrac.first()
    ccuts = np.quantile(bc.values, [1 / 3, 2 / 3])
    core["crop_tercile"] = np.digitize(core.cropfrac, ccuts)
    S["by_clear_tercile"] = table(core, ["clear_tercile"])
    S["by_crop_tercile"] = table(core, ["crop_tercile"])
    S["tercile_cuts"] = {"clear_rate": cuts.tolist(), "cropfrac": ccuts.tolist()}
    # ---- block-level tests
    S["tests"] = {}
    for ref in ("persistence", "climatology", "gbm_base"):
        if ref not in models:
            continue
        for m in models:
            if m != ref:
                r = compare(df, m, ref, sbmap)
                if r:
                    S["tests"][f"{m}_vs_{ref}"] = r
    S["weather_ablation"] = {m: S["tests"].get(f"{m}_vs_gbm_base") for m in models if m.startswith("gbm_")
                             and m != "gbm_base"}
    # ---- regional correlates (core units)
    p = core[core.model == "persistence"].groupby("block")
    blk = pd.DataFrame({"clear_rate": core[core.model == "persistence"].groupby(["block", "season", "issue"])
                        .clear_rate.first().groupby("block").mean(),
                        "cropfrac": p.cropfrac.first(), "region": p.region.first(), "superblock": p.superblock.first(),
                        "ndvi_peak": core[core.model == "persistence"].groupby(["block", "season"]).ndvi_peak.first()
                        .groupby("block").mean(),
                        "green_last_share": p.lastgreen.sum() / p.n.sum()})
    for m in ("persistence", "climatology", "gbm_base"):
        if m in models:
            blk[f"mae_{m}"] = block_metric(core[core.model == m], "mae")[m]
    if "gbm_base" in models:
        blk["gain_gbm_vs_pers"] = blk["mae_persistence"] - blk["mae_gbm_base"]
    S["regional"] = {}
    for err in [c for c in blk.columns if c.startswith("mae_") or c.startswith("gain_")]:
        for cov in ("clear_rate", "cropfrac", "ndvi_peak", "green_last_share"):
            ok = blk[[err, cov]].dropna()
            if len(ok) >= 3:
                rho, pv = spearmanr(ok[err], ok[cov])
                S["regional"][f"{err}__{cov}"] = {"rho": float(rho), "p": float(pv), "blocks": int(len(ok))}
    S["covariate_ranges"] = {c: [float(blk[c].min()), float(blk[c].max())]
                             for c in ("clear_rate", "cropfrac", "ndvi_peak", "green_last_share")}
    S["region_block_means"] = {r: {c: float(g[c].mean()) for c in blk.columns if c.startswith(("mae_", "gain_"))}
                               | {"blocks": int(len(g))} for r, g in blk.groupby("region")}
    S["blocks"] = {str(b): {k: (v if isinstance(v, str) else float(v)) for k, v in row.items()}
                   for b, row in blk.iterrows()}
    S["fit_selected"] = {m: {q: {k: v[k] for k in ("lr", "iters", "val_pinball")} for q, v in fit["fit"][m].items()
                             if isinstance(v, dict) and "iters" in v} for m in models if m.startswith("gbm_")
                         and m in fit["fit"]}
    S["fit_selected"]["ridge"] = fit["fit"].get("ridge")
    if nn is not None:
        neural(S, dfa, core_models, archs, use, runs, skipped, sbmap)
    json.dump(S, open(os.path.join(a.dir, "summary.json"), "w"), indent=1, default=float)
    write_tex(S, models, os.path.join(a.dir, "final_numbers.tex"))
    figure(S, models + archs, a.figdir)
    # console digest (f guards None / NaN)
    f = lambda v, spec: format(float("nan") if v is None else v, spec)
    print(f"{'model':16s} {'MAE':>7s} {'RMSE':>7s} {'bias':>8s} {'pinball':>8s} {'cov80':>6s} {'<P10':>6s} {'fb':>5s}")
    for m in models + archs:
        v = S["pooled"].get(m) or S["pooled_all_models_units"].get(m) or S["pooled_neural"][m]
        print(f"{m:16s} {f(v['mae'], '7.4f')} {f(v['rmse'], '7.4f')} {f(v['bias'], '8.4f')} {f(v['pinball'], '8.4f')} "
              f"{f(v['cover80'], '6.3f')} {f(v['below10'], '6.3f')} {f(v['fallback'], '5.2f')}")
    tests = dict(S["tests"]) | S.get("neural_seed_tests", {})
    for k, v in tests.items():
        if k.endswith("_vs_persistence") or k.endswith("_vs_gbm_base") or k == "earthformer_vs_convlstm":
            t = v["mae"]
            print(f"{k:34s} MAE diff {f(t['mean_diff'], '+.4f')} CI [{f(t['ci95'][0], '+.4f')},{f(t['ci95'][1], '+.4f')}] "
                  f"wins {t['wins']}/{t['blocks']} p={f(t['wilcoxon_p'], '.4f')} | pinball diff "
                  f"{f(v['pinball']['mean_diff'], '+.4f')}")
    if nn is not None:
        R = S["neural_runs"]
        print(f"neural seeds used {S['neural_seeds_used']}, skipped {S['neural_seeds_skipped']}")
        for r, v in R["runs"].items():
            print(f"  {r:16s} MAE {f(v['mae'], '.4f')} pinball {f(v['pinball'], '.4f')} cov80 {f(v['cover80'], '.3f')} "
                  f"best epoch {v['best_epoch']}/{v['epochs_run']} ({v['stopped']}) {f(v['train_minutes'], '.0f')} min "
                  f"peak {f(v['peak_gpu_gb'], '.2f')} GB {v['params']} params")
        for m, v in R["archs"].items():
            print(f"  {m:16s} {v['seeds']} seeds: MAE {f(v['mae_mean'], '.4f')} sd {f(v['mae_sd'], '.6f')} | pinball "
                  f"{f(v['pinball_mean'], '.4f')} sd {f(v['pinball_sd'], '.6f')} ({f(v['pinball_cv_pct'], '.2f')}%) "
                  f"[{f(v['pinball_min'], '.5f')}, {f(v['pinball_max'], '.5f')}] | same units {S['neural_same_units'][m]['units']}, gbm_base pinball "
                  f"{f(S['neural_same_units'][m]['pooled'].get('gbm_base', {}).get('pinball'), '.4f')}")
    print(json.dumps(S["counts"]))


def neural(S, dfa, core_models, archs, use, runs, skipped, sbmap):
    """neural keys of the summary; never touches the baseline keys (except adding arch tests to S["tests"])"""
    S["neural_seeds_used"] = {m: use[m] for m in archs}
    S["neural_seeds_skipped"] = skipped
    for k in ("pooled_neural", "by_lead_neural", "by_season_neural", "by_region_neural", "neural_same_units",
              "rel_mae_vs_persistence_neural"):
        S[k] = {}
    for m in archs:
        nu = common(dfa, core_models + [m])
        t = {k: table(nu, by) for k, by in (("pooled", []), ("by_lead", ["lead"]), ("by_season", ["season"]),
                                             ("by_region", ["region"]))}
        S["pooled_neural"][m] = t["pooled"][m]["all"]
        for k in ("by_lead", "by_season", "by_region"):
            S[f"{k}_neural"][m] = t[k][m]
        refs = [r for r in ("gbm_base", "persistence") if r in t["pooled"]]
        S["neural_same_units"][m] = {"units": int(nu.groupby(UNIT).ngroups),
                                     "pixel_forecasts": int(nu[nu.model == m].n.sum()),
                                     "pooled": {r: t["pooled"][r]["all"] for r in refs},
                                     "by_lead": {r: t["by_lead"][r] for r in refs}}
        if "persistence" in t["pooled"]:
            S["rel_mae_vs_persistence_neural"][m] = t["pooled"][m]["all"]["mae"] / t["pooled"]["persistence"]["all"]["mae"]
    # block-level tests: aggregates vs the three references, earthformer vs convlstm, each seed vs gbm / pers
    have = set(dfa.model)
    for ref in ("persistence", "climatology", "gbm_base"):
        for m in archs:
            r = compare(dfa, m, ref, sbmap) if ref in have else None
            if r:
                S["tests"][f"{m}_vs_{ref}"] = r
    if "earthformer" in archs and "convlstm" in archs:
        r = compare(dfa, "earthformer", "convlstm", sbmap)
        if r:
            S["tests"]["earthformer_vs_convlstm"] = r
    S["neural_seed_tests"] = {}
    S["neural_runs"] = {"runs": {}, "archs": {}}
    for m in archs:
        for s in use[m]:
            run = f"{m}_s{s}"
            for ref in ("gbm_base", "persistence"):
                r = compare(dfa, run, ref, sbmap) if ref in have else None
                if r:
                    S["neural_seed_tests"][f"{run}_vs_{ref}"] = r
            nu = common(dfa, core_models + [run])
            v = metrics(nu[nu.model == run], run)
            S["neural_runs"]["runs"][run] = {"arch": m, "seed": s, "units": int(nu.groupby(UNIT).ngroups)} | \
                {k: v[k] for k in ("mae", "rmse", "bias", "pinball", "cover80", "below10")} | \
                {k: runs[run].get(k) for k in ("params", "best_epoch", "epochs_run", "best_val_pinball",
                                                "train_minutes", "peak_gpu_gb", "stopped")}
        rr = [S["neural_runs"]["runs"][f"{m}_s{s}"] for s in use[m]]
        st = {"seeds": len(rr), "seed_list": use[m]}
        for k in ("mae", "pinball"):
            x = np.array([r[k] for r in rr], float)
            sd = float(x.std(ddof=1)) if len(x) >= 2 else float("nan")
            st |= {f"{k}_mean": float(x.mean()), f"{k}_sd": sd, f"{k}_min": float(x.min()), f"{k}_max": float(x.max()),
                   f"{k}_cv_pct": 100 * sd / float(x.mean())}
        S["neural_runs"]["archs"][m] = st


# ============================================================================ LaTeX
WORD = {1: "One", 2: "Two", 3: "Three"}


def num(v, d=4):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def pval(p):
    if p is None or not np.isfinite(p):
        return "--"
    return "$<$0.0001" if p < 1e-4 else f"{p:.4f}"


def yearword(y):
    return {2024: "TwentyFour", 2025: "TwentyFive", 2023: "TwentyThree"}.get(int(y), "Y")


def write_tex(S, models, path):
    L = ["% generated by final_analyze.py from results_v2/final/rows.csv -- do not edit"]
    c = lambda name, val: L.append(r"\newcommand{\fin%s}{%s}" % (name, val))
    for k, v in S["counts"].items():
        if isinstance(v, (int, np.integer)):
            c("N" + "".join(w.capitalize() for w in k.split("_")), f"{int(v):,}".replace(",", "{,}"))
    for m in models:
        k = KEY[m]
        v = S["pooled"].get(m) or S["pooled_all_models_units"].get(m)
        for nm, key in (("MAE", "mae"), ("RMSE", "rmse"), ("Bias", "bias"), ("Pin", "pinball"),
                        ("Cov", "cover80"), ("Below", "below10"), ("Fallback", "fallback")):
            c(k + nm, num(v[key]))
        c(k + "RelMAE", num(S["rel_mae_vs_persistence"].get(m)))
        bl =S["by_lead"].get(m) or S["by_lead_all_models_units"].get(m, {})
        for ld, w in WORD.items():
            if str(ld) in bl:
                c(f"{k}MAELead{w}", num(bl[str(ld)]["mae"]))
                c(f"{k}PinLead{w}", num(bl[str(ld)]["pinball"]))
        for y, v in S["by_season"].get(m, {}).items():
            c(f"{k}MAEYr{yearword(y)}", num(v["mae"]))
            c(f"{k}PinYr{yearword(y)}", num(v["pinball"]))
        for r, v in S["by_region"].get(m, {}).items():
            c(f"{k}MAE{r.replace(' ', '')}", num(v["mae"]))
        for t, w in enumerate(("Low", "Mid", "High")):
            v = S["by_clear_tercile"].get(m, {}).get(str(t))
            if v:
                c(f"{k}MAEClear{w}", num(v["mae"]))
            v = S["by_crop_tercile"].get(m, {}).get(str(t))
            if v:
                c(f"{k}MAECrop{w}", num(v["mae"]))
    refs = {"persistence": "Pers", "climatology": "Clim", "gbm_base": "Gbm", "convlstm": "ConvLSTM"}

    def tests(base, t, full=True):
        for metric, tag in (("mae", ""), ("pinball", "Pin")):
            r = t[metric]
            c(base + tag + "Diff", num(r["mean_diff"]))
            if full:
                c(base + tag + "Lo", num(r["ci95"][0]))
                c(base + tag + "Hi", num(r["ci95"][1]))
                c(base + tag + "Rel", num(100 * r["rel_diff"], 1))
            c(base + tag + "Wins", str(r["wins"]))
            c(base + tag + "Blocks", str(r["blocks"]))
            c(base + tag + "P", pval(r["wilcoxon_p"]))
    for name, t in S["tests"].items():
        m, ref = name.split("_vs_")
        if m not in NEURAL and ref not in NEURAL:        # neural tests go in the neural block at the end
            tests(f"{KEY[m]}Vs{refs[ref]}", t)
    covk = {"clear_rate": "Clear", "cropfrac": "Crop", "ndvi_peak": "Peak", "green_last_share": "Green"}
    errk = {"mae_persistence": "Pers", "mae_climatology": "Clim", "mae_gbm_base": "Gbm", "gain_gbm_vs_pers": "Gain"}
    for name, r in S["regional"].items():
        err, cov = name.split("__")
        c(f"Rho{errk[err]}{covk[cov]}", num(r["rho"], 2))
        c(f"RhoP{errk[err]}{covk[cov]}", pval(r["p"]))
    for cov, (lo, hi) in S["covariate_ranges"].items():
        d = 1 if cov == "ndvi_peak" else 4
        c(f"Range{covk[cov]}Lo", num(lo, d))
        c(f"Range{covk[cov]}Hi", num(hi, d))
    for m, qs in S["fit_selected"].items():
        if m == "ridge" or not qs:
            continue
        for q, v in qs.items():
            w = {"0.1": "Ten", "0.5": "Fifty", "0.9": "Ninety"}[q]
            c(f"{KEY[m]}Iters{w}", str(v["iters"]))
            c(f"{KEY[m]}Lr{w}", num(v["lr"], 2))
    if S["fit_selected"].get("ridge"):
        c("RidgeAlpha", f"{S['fit_selected']['ridge']['alpha']:g}")
    if "pooled_neural" in S:
        write_tex_neural(S, L, c, tests)
    names = [re.match(r"\\newcommand\{\\fin(.*?)\}\{", x).group(1) for x in L if x.startswith("\\newcommand")]
    dup = sorted({x for x in names if names.count(x) > 1})
    bad = [x for x in names if not re.fullmatch("[A-Za-z]+", x)]
    assert not dup and not bad, f"macro names duplicated {dup} / not letters-only {bad}"
    open(path, "w").write("\n".join(L) + "\n")


def write_tex_neural(S, L, c, tests):
    """macros of the neural seed aggregates (on their own units), their tests, and per-seed run facts"""
    L.append("% neural models from results_v2/final/neural_rows.csv + neural_runs.json: <Arch> = mean over complete "
             "seeds on the core units; <Arch>GbmSameUnits*/PersSameUnits* = baselines on those units")
    cnt = lambda v: f"{int(v):,}".replace(",", "{,}")
    R = S["neural_runs"]
    for m in [x for x in NEURAL if x in S["pooled_neural"]]:
        k, v = KEY[m], S["pooled_neural"][m]
        for nm, key in (("MAE", "mae"), ("RMSE", "rmse"), ("Bias", "bias"), ("Pin", "pinball"),
                        ("Cov", "cover80"), ("Below", "below10"), ("Fallback", "fallback")):
            c(k + nm, num(v[key]))
        c(k + "RelMAE", num(S["rel_mae_vs_persistence_neural"].get(m)))
        bl = S["by_lead_neural"][m]
        for ld, w in WORD.items():
            if str(ld) in bl:
                c(f"{k}MAELead{w}", num(bl[str(ld)]["mae"]))
                c(f"{k}PinLead{w}", num(bl[str(ld)]["pinball"]))
        for y, v in S["by_season_neural"][m].items():
            c(f"{k}MAEYr{yearword(y)}", num(v["mae"]))
            c(f"{k}PinYr{yearword(y)}", num(v["pinball"]))
        for r, v in S["by_region_neural"][m].items():
            c(f"{k}MAE{r.replace(' ', '')}", num(v["mae"]))
        su = S["neural_same_units"][m]
        c(k + "NUnits", cnt(su["units"]))
        for rk, ref in (("Gbm", "gbm_base"), ("Pers", "persistence")):
            if ref in su["pooled"]:
                c(f"{k}{rk}SameUnitsPin", num(su["pooled"][ref]["pinball"]))
                c(f"{k}{rk}SameUnitsMAE", num(su["pooled"][ref]["mae"]))
        a = R["archs"][m]
        c(k + "Seeds", str(a["seeds"]))
        # seed spread is far below the 4th decimal: SD at 6 decimals, range at 5, and SD as % of the seed mean
        for nm, key, d in (("PinSd", "pinball_sd", 6), ("PinMin", "pinball_min", 5), ("PinMax", "pinball_max", 5),
                           ("PinSdRel", "pinball_cv_pct", 2), ("MAESd", "mae_sd", 6), ("MAEMin", "mae_min", 5),
                           ("MAEMax", "mae_max", 5), ("MAESdRel", "mae_cv_pct", 2)):
            c(k + nm, num(a[key], d))
        for s in a["seed_list"]:
            run, w = f"{m}_s{s}", SEEDWORD.get(s)
            if w is None:
                print(f"note: no macro word for seed {s} ({run}); its per-seed macros are skipped")
                continue
            v, b = R["runs"][run], f"{k}{w}"
            c(b + "Pin", num(v["pinball"]))
            c(b + "MAE", num(v["mae"]))
            c(b + "Cov", num(v["cover80"]))
            if v["best_epoch"] is not None:
                c(b + "BestEpoch", str(int(v["best_epoch"])))
            if v["params"] is not None:
                c(b + "Params", cnt(v["params"]))
            c(b + "TrainMin", num(v["train_minutes"], 0))
            c(b + "PeakGpu", num(v["peak_gpu_gb"], 2))
            for ref, rk in (("gbm_base", "Gbm"), ("persistence", "Pers")):
                t = S["neural_seed_tests"].get(f"{run}_vs_{ref}")
                if t:
                    tests(f"{b}Vs{rk}", t, full=False)
    refs = {"persistence": "Pers", "climatology": "Clim", "gbm_base": "Gbm", "convlstm": "ConvLSTM"}
    for name, t in S["tests"].items():
        m, ref = name.split("_vs_")
        if m in NEURAL or ref in NEURAL:
            tests(f"{KEY[m]}Vs{refs[ref]}", t)


# ============================================================================ figure
def figure(S, models, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "font.family": "DejaVu Sans"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) pooled MAE by lead for four reference models (fixed categorical order)
    series = [("persistence", "#2a78d6", "o"), ("climatology", "#eb6834", "s"), ("ridge", "#1baf7a", "^"),
              ("gbm_base", "#eda100", "D")]
    for m, col, mk in series:
        if m not in S["by_lead"]:
            continue
        y = [S["by_lead"][m][str(L)]["mae"] for L in (1, 2, 3)]
        a1.plot([1, 2, 3], y, color=col, lw=2, marker=mk, ms=5, mec="white", mew=1, label=LABEL[m])
    # neural seed aggregates (own units = core units when every seed is complete): dashed, thinner, drawn
    # beneath the baselines (zorder 1.9 < 2) so a near-identical GBM line stays visible; colours chosen to
    # stay apart from the four baselines under simulated deuteranopia / protanopia / tritanopia
    nser = 0
    for m, col, mk in (("convlstm", "#2b2b2b", "v"), ("earthformer", "#b8327a", "P")):
        bl = S.get("by_lead_neural", {}).get(m)
        if bl:
            ld = [L for L in (1, 2, 3) if str(L) in bl]
            a1.plot(ld, [bl[str(L)]["mae"] for L in ld], color=col, lw=1.4, ls=(0, (4, 2)), marker=mk, ms=5,
                    mec="white", mew=0.8, label=LABEL[m], zorder=1.9)
            nser += 1
    a1.set_xticks([1, 2, 3]); a1.set_xlabel("Lead (dekads)"); a1.set_ylabel("MAE (NDTI)")
    a1.set_title("(a) Pooled test MAE by lead", loc="left", fontsize=8, color=ink)
    if nser:   # two-column legend above the data: extend the y range so it cannot overlap any line
        lo, hi = a1.get_ylim()
        a1.set_ylim(lo, hi + 0.42 * (hi - lo))
        a1.legend(frameon=False, fontsize=7, loc="upper left", ncol=2, columnspacing=1.0, handlelength=2.2)
    else:
        a1.legend(frameon=False, fontsize=7)
    a1.grid(axis="y", color=grid, lw=0.6); a1.set_axisbelow(True)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    # (b) mean block MAE difference vs persistence with 95% super-block bootstrap CI
    rows = [(m, S["tests"][f"{m}_vs_persistence"]["mae"]) for m in models
            if m != "persistence" and f"{m}_vs_persistence" in S["tests"]]
    rows.sort(key=lambda r: r[1]["mean_diff"])
    yy = np.arange(len(rows))[::-1]
    for y, (m, r) in zip(yy, rows):
        a2.plot(r["ci95"], [y, y], color=ink2, lw=1.5, solid_capstyle="round")
        a2.plot(r["mean_diff"], y, "o", color="#2a78d6" if r["ci95"][1] < 0 else ink2, ms=6, mec="white", mew=1)
        a2.text(1.0, y, f"{r['wins']}/{r['blocks']}", transform=a2.get_yaxis_transform(), va="center",
                ha="left", fontsize=7, color=ink2)
    a2.axvline(0, color=ink, lw=0.8)
    a2.set_yticks(yy); a2.set_yticklabels([LABEL[m] for m, _ in rows], fontsize=7)
    a2.set_xlabel("Block MAE minus persistence (95% CI)")
    a2.set_title("(b) Block-level difference vs persistence (wins/blocks)", loc="left", fontsize=8, color=ink)
    a2.grid(axis="x", color=grid, lw=0.6); a2.set_axisbelow(True)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig_final_results.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
