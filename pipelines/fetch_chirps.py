"""Fetch CHIRPS daily rainfall for the 16 fast_test blocks.

Primary source data.chc.ucsb.edu (CHIRPS v3.0 / 2.0) was unreachable (TCP connect
timeout on 443/80), so this uses the SERVIR ClimateSERV API, which serves the UCSB
CHIRPS 0.05 deg daily grid as server-side polygon averages:
  precip_mm           : the single CHIRPS pixel containing the block centre
  precip_mm_blockmean : mean over the block's 0.25 deg bbox (~25 pixels)
"""
import json, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipelines"))
from mh_grid import bbox  # noqa: E402

API = "https://climateserv.servirglobal.net/api/"
SRC = (API + "submitDataRequest/?datatype=0(UCSB CHIRPS Rainfall, 0.05deg daily)"
       "&intervaltype=0&operationtype=5(avg)&geometry=<polygon>")
D0, D1 = date(2024, 8, 1), date(2026, 1, 31)
OUT = os.path.join(ROOT, "local_data", "weather")


def poly(x0, y0, x1, y1):
    return json.dumps({"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]})


def get(path, **q):
    u = API + path + "/?" + urllib.parse.urlencode(q)
    for k in range(5):
        try:
            return json.loads(urllib.request.urlopen(u, timeout=180).read())
        except Exception as e:  # retry transient errors
            if k == 4:
                raise
            time.sleep(3 * (k + 1))


def run(geom, d0, d1):
    jid = get("submitDataRequest", datatype=0, begintime=d0.strftime("%m/%d/%Y"),
              endtime=d1.strftime("%m/%d/%Y"), intervaltype=0, operationtype=5, geometry=geom)[0]
    for _ in range(600):
        p = get("getDataRequestProgress", id=jid)[0]
        if p >= 100 or p < 0:
            break
        time.sleep(2)
    if p < 0:
        raise RuntimeError(f"job {jid} failed")
    out = {}
    for r in get("getDataFromRequest", id=jid)["data"]:
        m, d, y = map(int, r["date"].split("/"))
        v = r["value"].get("avg")
        out[date(y, m, d).isoformat()] = None if v is None or v < -1 else float(v)
    return out


def fetch_block(b):
    x0, y0, x1, y1 = bbox(b)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    e = 0.001
    pt = run(poly(cx - e, cy - e, cx + e, cy + e), D0, D1)
    bm = run(poly(x0, y0, x1, y1), D0, D1)
    days = [(D0 + timedelta(i)).isoformat() for i in range((D1 - D0).days + 1)]
    rec = {"source": SRC, "source_note": "data.chc.ucsb.edu unreachable; CHIRPS via SERVIR ClimateSERV",
           "block": b, "centre_lonlat": [cx, cy], "bbox": [x0, y0, x1, y1],
           "time": days, "precip_mm": [pt.get(d) for d in days],
           "precip_mm_blockmean": [bm.get(d) for d in days]}
    with open(os.path.join(OUT, f"chirps_blk_{b}.json"), "w") as f:
        json.dump(rec, f)
    miss = sum(v is None for v in rec["precip_mm"]), sum(v is None for v in rec["precip_mm_blockmean"])
    print(b, "missing(centre, blockmean) =", miss, flush=True)
    return b, miss


if __name__ == "__main__":
    plan = json.load(open(os.path.join(ROOT, "results_v2", "ndti_local", "plan.json")))
    blocks = plan["fast_test"]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(fetch_block, blocks))
    print(res)
