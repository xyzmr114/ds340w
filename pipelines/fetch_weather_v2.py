"""Daily weather for all 74 plan blocks (test + train + val), 2018-08-01 .. 2026-01-31.

Writes local_data/weather_v2/:
  era5_blk_<b>.json   Open-Meteo archive (ERA5) response at the block centre, same structure as
                      local_data/weather/blk_<b>.json: daily precipitation_sum (mm),
                      temperature_2m_mean (degC), soil_moisture_0_to_7cm_mean (m3/m3), tz Asia/Kolkata.
  chirps_blk_<b>.json CHIRPS 0.05 deg daily via SERVIR ClimateSERV, exactly as fetch_chirps.py:
                      precip_mm (centre pixel) and precip_mm_blockmean (0.25 deg bbox mean).

Resumable: blocks whose output file already exists and is complete are skipped. Files are written
atomically (tmp + os.replace).

  python fetch_weather_v2.py [era5|chirps|all] [--validate]
"""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipelines"))
from mh_grid import bbox  # noqa: E402
import fetch_chirps as FC  # noqa: E402  (reuse ClimateSERV client: poly, run, SRC)

D0, D1 = date(2018, 8, 1), date(2026, 1, 31)
DAYS = [(D0 + timedelta(i)).isoformat() for i in range((D1 - D0).days + 1)]
OUT = os.path.join(ROOT, "local_data", "weather_v2")
OLD = os.path.join(ROOT, "local_data", "weather")
OM = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["precipitation_sum", "temperature_2m_mean", "soil_moisture_0_to_7cm_mean"]
# Open-Meteo free tier weights a call by days/14 * vars/10 (~59 "calls" here; limit 600/min,
# 5000/h), so space requests ~8 s apart.
OM_GAP = 8.0


def plan_blocks():
    p = json.load(open(os.path.join(ROOT, "results_v2", "ndti_local", "plan.json")))
    return sorted(set(p["test"]) | set(p["train"]) | set(p["val"]))


def centre(b):
    x0, y0, x1, y1 = bbox(b)
    return (x0 + x1) / 2, (y0 + y1) / 2


def dump(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------- ERA5 (Open-Meteo archive)
def era5_ok(path):
    try:
        d = json.load(open(path))["daily"]
        return d["time"] == DAYS and all(len(d[v]) == len(DAYS) for v in VARS)
    except Exception:
        return False


def fetch_era5(b):
    path = os.path.join(OUT, f"era5_blk_{b}.json")
    if era5_ok(path):
        return b, "skip"
    cx, cy = centre(b)
    q = dict(latitude=cy, longitude=cx, start_date=D0.isoformat(), end_date=D1.isoformat(),
             daily=",".join(VARS), timezone="Asia/Kolkata")
    u = OM + "?" + urllib.parse.urlencode(q)
    for k in range(8):
        try:
            rec = json.loads(urllib.request.urlopen(u, timeout=180).read())
            break
        except urllib.error.HTTPError as e:
            if k == 7:
                raise
            wait = 65 if e.code == 429 else 5 * (k + 1)       # 429: wait out the minute window
            print(f"  era5 {b}: HTTP {e.code}, retry in {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            if k == 7:
                raise
            print(f"  era5 {b}: {e!r}, retry", flush=True)
            time.sleep(5 * (k + 1))
    rec["block"], rec["centre_lonlat"], rec["bbox"] = b, [cx, cy], list(bbox(b))
    rec["source"] = OM + " (default best_match = ERA5 reanalysis)"
    dump(path, rec)
    return b, "ok"


def run_era5(blocks):
    for i, b in enumerate(blocks):
        t = time.time()
        r = fetch_era5(b)
        print(f"era5 {i + 1}/{len(blocks)} blk {b}: {r[1]}", flush=True)
        if r[1] == "ok":
            time.sleep(max(0.0, OM_GAP - (time.time() - t)))


# ---------------------------------------------------------------- CHIRPS (ClimateSERV)
def chirps_ok(path):
    try:
        d = json.load(open(path))
        return d["time"] == DAYS and len(d["precip_mm"]) == len(d["precip_mm_blockmean"]) == len(DAYS)
    except Exception:
        return False


def fetch_chirps(b):
    path = os.path.join(OUT, f"chirps_blk_{b}.json")
    if chirps_ok(path):
        return b, "skip"
    x0, y0, x1, y1 = bbox(b)
    cx, cy = centre(b)
    e = 0.001
    for k in range(3):                       # whole-job retry (FC.get already retries HTTP)
        try:
            pt = FC.run(FC.poly(cx - e, cy - e, cx + e, cy + e), D0, D1)
            bm = FC.run(FC.poly(x0, y0, x1, y1), D0, D1)
            break
        except Exception as ex:
            if k == 2:
                print(f"chirps {b}: FAILED {ex!r}", flush=True)
                return b, "fail"
            time.sleep(20 * (k + 1))
    rec = {"source": FC.SRC, "source_note": "data.chc.ucsb.edu unreachable; CHIRPS via SERVIR ClimateSERV",
           "block": b, "centre_lonlat": [cx, cy], "bbox": [x0, y0, x1, y1],
           "time": DAYS, "precip_mm": [pt.get(d) for d in DAYS],
           "precip_mm_blockmean": [bm.get(d) for d in DAYS]}
    dump(path, rec)
    miss = sum(v is None for v in rec["precip_mm"]), sum(v is None for v in rec["precip_mm_blockmean"])
    print(f"chirps blk {b}: missing(centre, blockmean) = {miss}", flush=True)
    return b, "ok"


def run_chirps(blocks):
    with ThreadPoolExecutor(2) as ex:
        return list(ex.map(fetch_chirps, blocks))


# ---------------------------------------------------------------- validation
def validate(blocks):
    import numpy as np
    rep = {"n_blocks": len(blocks), "n_days": len(DAYS), "era5_files": 0, "chirps_files": 0,
           "era5_missing_values": {}, "chirps_missing_values": {}, "ranges": {}, "overlap_vs_v1": {}}
    rng = {v: [np.inf, -np.inf] for v in VARS + ["chirps_pt", "chirps_bm"]}
    for b in blocks:
        pe, pc = os.path.join(OUT, f"era5_blk_{b}.json"), os.path.join(OUT, f"chirps_blk_{b}.json")
        if era5_ok(pe):
            rep["era5_files"] += 1
            d = json.load(open(pe))["daily"]
            for v in VARS:
                a = np.array([np.nan if x is None else x for x in d[v]], float)
                if np.isnan(a).any():
                    rep["era5_missing_values"].setdefault(str(b), {})[v] = int(np.isnan(a).sum())
                rng[v] = [min(rng[v][0], np.nanmin(a)), max(rng[v][1], np.nanmax(a))]
        if chirps_ok(pc):
            rep["chirps_files"] += 1
            c = json.load(open(pc))
            for key, nm in (("precip_mm", "chirps_pt"), ("precip_mm_blockmean", "chirps_bm")):
                a = np.array([np.nan if x is None else x for x in c[key]], float)
                if np.isnan(a).any():
                    rep["chirps_missing_values"].setdefault(str(b), {})[key] = int(np.isnan(a).sum())
                rng[nm] = [min(rng[nm][0], np.nanmin(a)), max(rng[nm][1], np.nanmax(a))]
        # overlap with the 16 fast_test blocks fetched earlier (2024-08-01 .. 2026-01-31)
        oe, oc = os.path.join(OLD, f"blk_{b}.json"), os.path.join(OLD, f"chirps_blk_{b}.json")
        if os.path.exists(oe) and era5_ok(pe):
            o, n = json.load(open(oe))["daily"], json.load(open(pe))["daily"]
            ix = {t: i for i, t in enumerate(n["time"])}
            r = {}
            for v in VARS:
                diff = [abs((n[v][ix[t]] or 0) - (o[v][i] or 0)) for i, t in enumerate(o["time"])]
                r[v] = {"n": len(diff), "n_diff": int(sum(x > 1e-9 for x in diff)), "max_abs": max(diff)}
            rep["overlap_vs_v1"].setdefault(str(b), {})["era5"] = r
        if os.path.exists(oc) and chirps_ok(pc):
            o, n = json.load(open(oc)), json.load(open(pc))
            ix = {t: i for i, t in enumerate(n["time"])}
            r = {}
            for key in ("precip_mm", "precip_mm_blockmean"):
                diff = [abs((n[key][ix[t]] or 0) - (o[key][i] or 0)) for i, t in enumerate(o["time"])]
                r[key] = {"n": len(diff), "n_diff": int(sum(x > 1e-6 for x in diff)), "max_abs": max(diff)}
            rep["overlap_vs_v1"].setdefault(str(b), {})["chirps"] = r
    rep["ranges"] = {k: [float(a), float(z)] for k, (a, z) in rng.items()}
    return rep


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "all"
    blocks = plan_blocks()
    if "--blocks" in sys.argv:
        blocks = [int(x) for x in sys.argv[sys.argv.index("--blocks") + 1].split(",")]
    print(len(blocks), "blocks", DAYS[0], "..", DAYS[-1], len(DAYS), "days", flush=True)
    if what in ("era5", "all"):
        run_era5(blocks)
    if what in ("chirps", "all"):
        print(run_chirps(blocks), flush=True)
    if "--validate" in sys.argv:
        rep = validate(plan_blocks())
        dump(os.path.join(OUT, "validation.json"), rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "overlap_vs_v1"}, indent=1, default=str))
