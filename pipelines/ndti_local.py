"""Local (no-Delta) dekadal NDTI cubes for residue-baseline scoring.

Same definitions as cube20.py / harvest_dekads.py, but run on a laptop for a
subset of blocks, and storing only what the persistence / climatology baselines need:

  season Y = Sep 11 Y .. Jan 31 Y+1, 14 dekads (harvest_dekads.dekads)
  per dekad, 20 m, EPSG:32644 grid snapped to 100 m, one 12.8 km window per block (window_geobox):
    ndti  median NDTI = (B11-B12)/(B11+B12) over clear observations
    ndvi  median NDVI = (B8A-B4)/(B8A+B4)   over clear observations
    nclr  number of clear observations
  static: WorldCover 2021 class (cropland = 40)

Sentinel-2 L2A from Earth Search `sentinel-2-l2a` (pixels already harmonised, no offset;
see s2io.py), SCL classes 0,1,3,8,9,10,11 masked, duplicate baselines dropped.

WorldCover is read from the ESA AWS open-data bucket (same v200 product as on Planetary Computer).

Usage:
  python ndti_local.py cropfrac --blocks 2,3,12        # WorldCover cropland share per block
  python ndti_local.py cube --block 61 --season 2024    # one block-season cube
Block ids are mh_grid ids (0.25 deg lattice from 72.5 E, 15.5 N). Every block uses the
EPSG:32644 grid of cube20.py; west of 78 E this is outside UTM zone 44, where the scale
error stays below 1%, so the 20 m pixels remain 20 +/- 0.2 m.
Output: ../local_data/harvest20/blk_<B>_<Y>.npz, ../local_data/harvest20/wc_<B>.npy
"""
import argparse
import json
import os
import time

import dask
import numpy as np
import odc.stac
import pandas as pd
import pystac_client

import dryseason_blocks as D
from harvest_dekads import dekads
from mh_grid import bbox as block_bbox   # all-Maharashtra block ids (mh_grid.py)
from s2io import search_s2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local_data", "harvest20")
SCL_BAD = [0, 1, 3, 8, 9, 10, 11]
HALF = 6400.0   # one 12.8 km x 12.8 km window (640 x 640 px at 20 m) centred in each block


def block_crs(b):
    """Native UTM zone of the block centre (43N west of 78 E, 44N east), so most scenes need no warp."""
    w, s, e, n = block_bbox(b)
    return "EPSG:32643" if (w + e) / 2 < 78.0 else "EPSG:32644"


def window_geobox(b, res=20.0):
    """Centre window of block b on its native UTM grid, edges snapped to 100 m."""
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import BoundingBox
    from pyproj import Transformer
    crs = os.environ.get("NDTI_CRS") or block_crs(b)
    w, s, e, n = block_bbox(b)
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform((w + e) / 2, (s + n) / 2)
    cx, cy = round(cx / 100.0) * 100.0, round(cy / 100.0) * 100.0
    return {"geobox": GeoBox.from_bbox(BoundingBox(cx - HALF, cy - HALF, cx + HALF, cy + HALF, crs=crs),
                                       resolution=res)}


WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map.tif")   # AWS open data, 3x3 deg tiles


def read_worldcover(b, gb):
    """WorldCover 2021 class on the geobox grid (nearest), mosaicked from the 3-degree tiles."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    g = gb["geobox"]
    out = np.zeros(g.shape, np.uint8)
    w, s, e, n = block_bbox(b)
    for lat in range(int(s // 3 * 3), int(n // 3 * 3) + 1, 3):
        for lon in range(int(w // 3 * 3), int(e // 3 * 3) + 1, 3):
            tmp = np.zeros(g.shape, np.uint8)
            with rasterio.open(WC_URL.format(lat=lat, lon=lon)) as src:
                reproject(rasterio.band(src, 1), tmp, dst_transform=g.affine, dst_crs=str(g.crs),
                          resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
            out = np.where(out == 0, tmp, out)
    return out


def worldcover(b, gb):
    path = os.path.join(OUT, f"wc_{b:03d}.npy")
    if os.path.exists(path):
        return np.load(path)
    wc = read_worldcover(b, gb)
    np.save(path, wc)
    return wc


def cropfrac(blocks):
    """Cropland share of each block's window, from WorldCover read at 100 m (overview read)."""
    out = {}
    for b in blocks:
        wc = read_worldcover(b, window_geobox(b, res=100.0))
        out[b] = round(float((wc == 40).mean()), 3)
        print(b, out[b], flush=True)
    return out


def cube(b, year):
    path = os.path.join(OUT, f"blk_{b:03d}_{year}.npz")
    if os.path.exists(path):
        return print("exists", path)
    t_start = time.time()
    bbox = block_bbox(b)
    gb = window_geobox(b)
    wc = worldcover(b, gb)
    items = search_s2(pystac_client.Client.open(D.EARTH_SEARCH), bbox,
                      f"{year}-09-11/{year + 1}-01-31", max_cloud=70)
    ds = odc.stac.load(items, bands=["red", "nir08", "swir16", "swir22", "scl"], groupby="solar_day",
                       chunks={"x": 1024, "y": 1024, "time": 1}, **gb)
    ok = (ds.scl > 0) & ~ds.scl.isin(SCL_BAD)
    f = lambda k: ds[k].where(ds[k] > 0).astype("float32")
    ndti = ((f("swir16") - f("swir22")) / (f("swir16") + f("swir22"))).where(ok)
    ndvi = ((f("nir08") - f("red")) / (f("nir08") + f("red"))).where(ok)
    starts = dekads(year)
    times = pd.DatetimeIndex(ds.time.values)
    H, W = wc.shape
    out = {"ndti": [], "ndvi": [], "nclr": []}
    for t0 in starts:
        idx = np.where((times >= t0) & (times < t0 + pd.Timedelta(days=10)))[0]
        if len(idx) == 0:
            out["ndti"].append(np.full((H, W), np.nan, np.float32))
            out["ndvi"].append(np.full((H, W), np.nan, np.float32))
            out["nclr"].append(np.zeros((H, W), np.uint8))
            continue
        a, v, n = dask.compute(ndti.isel(time=idx).median("time"), ndvi.isel(time=idx).median("time"),
                               ndti.isel(time=idx).notnull().sum("time"))
        out["ndti"].append(a.values.astype(np.float32))
        out["ndvi"].append(v.values.astype(np.float32))
        out["nclr"].append(n.values.clip(0, 255).astype(np.uint8))
        print(b, year, t0.date(), len(idx), "scenes", flush=True)
    np.savez_compressed(path, ndti=np.stack(out["ndti"]).astype(np.float16),
                        ndvi=np.stack(out["ndvi"]).astype(np.float16), nclr=np.stack(out["nclr"]),
                        starts=np.array([str(s.date()) for s in starts]),
                        scenes=np.array(sorted(i.id for i in items)))
    print("wrote", path, "items", len(items), "secs", round(time.time() - t_start), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cropfrac", "cube"])
    ap.add_argument("--blocks", default="")
    ap.add_argument("--block", type=int)
    ap.add_argument("--season", type=int)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "cropfrac":
        blocks = open(a.blocks[1:]).read() if a.blocks.startswith("@") else a.blocks
        res = cropfrac([int(x) for x in blocks.split(",")])
        json.dump(res, open(os.path.join(OUT, f"cropfrac{a.tag}.json"), "w"), indent=1)
    else:
        cube(a.block, a.season)


if __name__ == "__main__":
    main()
