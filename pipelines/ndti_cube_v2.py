"""Final-report cube builder (v2). Differences from ndti_local.py (Week 9):

  * one fixed grid per block for every season: the block's native UTM zone (43N west of 78 E,
    44N east), so per-pixel climatology and seasonal persistence line up across years
  * the last dekad runs to 1 February, so scenes on 31 January are kept
  * stores the four reflectance bands (B4, B8A, B11, B12; clear-observation medians, DN x1e4,
    0 = no clear observation) in addition to NDTI, NDVI and the clear count
  * stores the CRS and affine transform; writes atomically (tmp file then rename)

season Y = 11 Sep Y .. 31 Jan Y+1, 14 dekads; 12.8 km x 12.8 km window (640 x 640 px, 20 m) at the
block centre; Sentinel-2 L2A (Earth Search, harmonised, no offset), cloud cover < 70%, SCL classes
0,1,3,8-11 masked; WorldCover 2021 from the ESA open-data bucket.

Usage: python ndti_cube_v2.py --block B --season Y
Output: ../local_data/harvest20v2/blk_<B>_<Y>.npz and wc_<B>.npy
"""
import argparse
import os
import time

import dask
import numpy as np
import odc.stac
import pandas as pd
import pystac_client

import dryseason_blocks as D
from harvest_dekads import dekads
from mh_grid import bbox as block_bbox
from ndti_local import HALF, SCL_BAD, block_crs, read_worldcover
from s2io import search_s2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local_data", "harvest20v2")
BANDS = ["red", "nir08", "swir16", "swir22"]


def geobox(b, res=20.0):
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import BoundingBox
    from pyproj import Transformer
    crs = block_crs(b)
    w, s, e, n = block_bbox(b)
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform((w + e) / 2, (s + n) / 2)
    cx, cy = round(cx / 100.0) * 100.0, round(cy / 100.0) * 100.0
    return {"geobox": GeoBox.from_bbox(BoundingBox(cx - HALF, cy - HALF, cx + HALF, cy + HALF, crs=crs),
                                       resolution=res)}


def worldcover(b, gb):
    path = os.path.join(OUT, f"wc_{b:03d}.npy")
    if os.path.exists(path):
        return np.load(path)
    wc = read_worldcover(b, gb)
    tmp = path + ".tmp.npy"
    np.save(tmp, wc)
    os.replace(tmp, path)
    return wc


def cube(b, year):
    path = os.path.join(OUT, f"blk_{b:03d}_{year}.npz")
    if os.path.exists(path):
        return print("exists", path)
    t_start = time.time()
    gb = geobox(b)
    wc = worldcover(b, gb)
    items = search_s2(pystac_client.Client.open(D.EARTH_SEARCH), block_bbox(b),
                      f"{year}-09-11/{year + 1}-01-31", max_cloud=70)
    ds = odc.stac.load(items, bands=BANDS + ["scl"], groupby="solar_day",
                       chunks={"x": 1024, "y": 1024, "time": 1}, **gb)
    ok = (ds.scl > 0) & ~ds.scl.isin(SCL_BAD)
    f = lambda k: ds[k].where((ds[k] > 0) & ok).astype("float32")
    ndti = (f("swir16") - f("swir22")) / (f("swir16") + f("swir22"))
    ndvi = (f("nir08") - f("red")) / (f("nir08") + f("red"))
    starts = dekads(year)
    edges = list(starts[1:]) + [pd.Timestamp(year=year + 1, month=2, day=1)]
    times = pd.DatetimeIndex(ds.time.values)
    H, W = wc.shape
    out = {k: [] for k in ["ndti", "ndvi", "nclr"] + BANDS}
    for t0, t1 in zip(starts, edges):
        idx = np.where((times >= t0) & (times < t1))[0]
        if len(idx) == 0:
            out["ndti"].append(np.full((H, W), np.nan, np.float32))
            out["ndvi"].append(np.full((H, W), np.nan, np.float32))
            out["nclr"].append(np.zeros((H, W), np.uint8))
            for k in BANDS:
                out[k].append(np.zeros((H, W), np.uint16))
            continue
        sub = lambda a: a.isel(time=idx)
        res = dask.compute(sub(ndti).median("time"), sub(ndvi).median("time"), sub(ndti).notnull().sum("time"),
                           *[sub(f(k)).median("time") for k in BANDS])
        out["ndti"].append(res[0].values.astype(np.float32))
        out["ndvi"].append(res[1].values.astype(np.float32))
        out["nclr"].append(res[2].values.clip(0, 255).astype(np.uint8))
        for k, r in zip(BANDS, res[3:]):
            out[k].append(np.nan_to_num(r.values, nan=0).clip(0, 65535).astype(np.uint16))
        print(b, year, t0.date(), len(idx), "scenes", flush=True)
    g = gb["geobox"]
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, ndti=np.stack(out["ndti"]).astype(np.float16),
                        ndvi=np.stack(out["ndvi"]).astype(np.float16), nclr=np.stack(out["nclr"]),
                        **{k: np.stack(out[k]) for k in BANDS},
                        starts=np.array([str(s.date()) for s in starts]),
                        scenes=np.array(sorted(i.id for i in items)),
                        crs=np.array(str(g.crs)), transform=np.array(list(g.affine)[:6]))
    os.replace(tmp, path)
    print("wrote", path, "items", len(items), "secs", round(time.time() - t_start), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--season", type=int, required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cube(a.block, a.season)
