"""Harvest-season datacube for residue forecasting (replaces the Gemini tile-mean set).

For one 0.25 deg block (same grid as dryseason_blocks.py) and one harvest season
(Sep 11 of year Y -> Jan 31 of Y+1), build 10-day (dekad) composites from
Sentinel-2 L2A at 20 m, then aggregate to 100 m:
  ndti   median NDTI = (B11-B12)/(B11+B12) of clear cropland obs   (residue / tillage)
  ndvi   median NDVI of clear cropland obs                          (crop still standing)
  clear  fraction of the 100 m cell's cropland pixels with >=1 clear obs in the dekad
Cropland = ESA WorldCover 2021 class 40. Non-crop pixels are excluded before aggregation.

Output: /work/hdd/bidp/hrathi1/cbg_biomass/v2/harvest/blk_<id>_<Y>.zarr
         dims (time=dekad start, y, x) on the 100 m EPSG:32644 grid.
"""
import argparse
import os

import dask
import numpy as np
import odc.stac
import pandas as pd
import pystac_client
import xarray as xr

from dryseason_blocks import (EARTH_SEARCH, blocks, geobox_kwargs, pc_catalog,
                              snapped_bbox)
from s2io import nd, search_s2

OUT_DIR = "/work/hdd/bidp/hrathi1/cbg_biomass/v2/harvest"
AGG = 5
SCL_BAD = [0, 1, 3, 8, 9, 10, 11]


def dekads(year):
    starts = []
    for m, y in [(9, year), (10, year), (11, year), (12, year), (1, year + 1)]:
        for d in (1, 11, 21):
            t = pd.Timestamp(year=y, month=m, day=d)
            if t >= pd.Timestamp(year=year, month=9, day=11):
                starts.append(t)
    return starts


def agg(a, how):
    t, h, w = a.shape
    h2, w2 = (h // AGG) * AGG, (w // AGG) * AGG
    r = a[:, :h2, :w2].reshape(t, h2 // AGG, AGG, w2 // AGG, AGG)
    with np.errstate(all="ignore"):
        return np.nanmedian(r, axis=(2, 4)) if how == "median" else np.nanmean(r, axis=(2, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--season", type=int, required=True, help="harvest year Y (Sep Y - Jan Y+1)")
    a = ap.parse_args()
    out = os.path.join(OUT_DIR, f"blk_{a.block:03d}_{a.season}.zarr")
    os.makedirs(OUT_DIR, exist_ok=True)

    bbox = blocks()[a.block]
    # Snap to 100 m (not just 20 m) so the 5x5 aggregates of neighbouring blocks share
    # one 100 m lattice and mosaic without offsets.
    x0, y0, x1, y1 = snapped_bbox(bbox)
    snap = lambda v, f: f(v / 100.0) * 100.0
    gb = geobox_kwargs((snap(x0, np.floor), snap(y0, np.floor), snap(x1, np.ceil), snap(y1, np.ceil)))
    es = pystac_client.Client.open(EARTH_SEARCH)
    items = search_s2(es, bbox, f"{a.season}-09-11/{a.season + 1}-01-31", max_cloud=80)
    ds = odc.stac.load(items, bands=["red", "nir", "swir16", "swir22", "scl"], groupby="solar_day",
                       chunks={"x": 1400, "y": 1400, "time": 1}, **gb)
    wc = odc.stac.load(list(pc_catalog().search(collections=["esa-worldcover"], bbox=bbox,
                                                datetime="2021-01-01/2021-12-31").items()),
                       bands=["map"], groupby="solar_day", **gb)["map"].isel(time=0).values
    crop = wc == 40

    f = lambda b: xr.where(ds[b] == 0, np.nan, ds[b].astype("float32") * 1e-4)
    ok = ~ds["scl"].isin(SCL_BAD) & (ds["scl"] > 0)
    ndti = nd(f("swir16"), f("swir22")).where(ok)
    ndvi = nd(f("nir"), f("red")).where(ok)

    starts = dekads(a.season)
    edges = starts + [starts[-1] + pd.Timedelta(days=10)]
    times = pd.DatetimeIndex(ds.time.values)
    stack = {"ndti": [], "ndvi": [], "clear": []}
    for t0, t1 in zip(edges[:-1], edges[1:]):
        idx = np.where((times >= t0) & (times < t1))[0]
        if len(idx) == 0:
            shape = crop.shape
            nan = np.full(shape, np.nan, np.float32)
            stack["ndti"].append(nan); stack["ndvi"].append(nan); stack["clear"].append(np.zeros(shape, np.float32))
            continue
        ti, vi = dask.compute(ndti.isel(time=idx).median("time"), ndvi.isel(time=idx).median("time"))
        ti, vi = ti.values, vi.values
        cl = np.isfinite(ti).astype(np.float32)
        ti[~crop] = np.nan; vi[~crop] = np.nan; cl[~crop] = np.nan
        stack["ndti"].append(ti); stack["ndvi"].append(vi); stack["clear"].append(cl)
        print(a.block, a.season, t0.date(), len(idx), "scenes", flush=True)

    x = ds.x.values[: (len(ds.x) // AGG) * AGG].reshape(-1, AGG).mean(1)
    y = ds.y.values[: (len(ds.y) // AGG) * AGG].reshape(-1, AGG).mean(1)
    cropfrac = agg(crop[None].astype(np.float32), "mean")[0]
    res = xr.Dataset(
        {"ndti": (("time", "y", "x"), agg(np.stack(stack["ndti"]), "median").astype(np.float32)),
         "ndvi": (("time", "y", "x"), agg(np.stack(stack["ndvi"]), "median").astype(np.float32)),
         "clear": (("time", "y", "x"), agg(np.stack(stack["clear"]), "mean").astype(np.float32)),
         "cropfrac": (("y", "x"), cropfrac.astype(np.float32))},
        coords={"time": starts, "y": y, "x": x},
        attrs={"crs": "EPSG:32644", "res": 100.0, "block": a.block, "season": a.season,
               "n_scenes": len(items)})
    res.to_zarr(out, mode="w")
    print("wrote", out, dict(res.sizes), flush=True)


if __name__ == "__main__":
    main()
