"""Per-block, per-year dry-season composites for the drought-persistence water map.

The study region is cut into 0.25 deg blocks (one Slurm array task each). For every
block and year it stores, on a fixed 20 m EPSG:32644 grid:
  ndvi_may   late-season (May) median NDVI of clear observations   (int16, x1000)
  mndwi_max  Mar-May max MNDWI                                      (int16, x1000)
  n_clear    clear observations Mar-May                             (uint8)
plus a one-off ESA WorldCover 2021 class layer (uint8).

Years 2018+ use Sentinel-2 L2A (Earth Search). 2016-2017 (the season after the 2015
drought) use Landsat 8/9 C2 L2 from Planetary Computer, since S2 L2A coverage of
Vidarbha only starts in 2018.

Output: /work/hdd/bidp/hrathi1/cbg_biomass/v2/dryseason/blk_<id>.zarr
"""
import argparse
import os

import numpy as np
import odc.stac
import pystac_client
import xarray as xr
from pyproj import Transformer

from s2io import nd, search_s2

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CRS = "EPSG:32644"
RES = 20.0
REGION = (77.25, 19.75, 80.25, 21.75)  # W, S, E, N  (Amravati -> Bhandara, Wardha -> Chhindwara edge)
BLOCK_DEG = 0.25
OUT_DIR = os.environ.get("CBG_DRY_DIR", "/work/hdd/bidp/hrathi1/cbg_biomass/v2/dryseason")


def blocks():
    w, s, e, n = REGION
    out = []
    for lat in np.arange(s, n, BLOCK_DEG):
        for lon in np.arange(w, e, BLOCK_DEG):
            out.append((round(lon, 4), round(lat, 4), round(lon + BLOCK_DEG, 4), round(lat + BLOCK_DEG, 4)))
    return out


def pc_catalog():
    import planetary_computer
    return pystac_client.Client.open(PC_URL, modifier=planetary_computer.sign_inplace)


def snapped_bbox(bbox):
    """Snap the block to the 20 m grid in EPSG:32644 so adjacent blocks tile exactly."""
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    xs, ys = zip(*[to_utm.transform(x, y) for x in bbox[::2] for y in bbox[1::2]])
    x0, x1 = np.floor(min(xs) / RES) * RES, np.ceil(max(xs) / RES) * RES
    y0, y1 = np.floor(min(ys) / RES) * RES, np.ceil(max(ys) / RES) * RES
    return x0, y0, x1, y1


def geobox_kwargs(xy):
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import BoundingBox
    return {"geobox": GeoBox.from_bbox(BoundingBox(*xy, crs=CRS), resolution=RES)}


def s2_year(cat, bbox, gb, year):
    items = search_s2(cat, bbox, f"{year}-03-01/{year}-05-31", max_cloud=60)
    if not items:
        return None
    ds = odc.stac.load(items, bands=["green", "red", "nir", "swir16", "scl"], groupby="solar_day",
                       chunks={"x": 1024, "y": 1024}, **gb)
    # Earth Search sentinel-2-l2a pixels are already harmonised (see s2io docstring).
    refl = {b: xr.where(ds[b] == 0, np.nan, ds[b].astype("float32") * 1e-4) for b in
            ["green", "red", "nir", "swir16"]}
    scl = xr.where(ds["scl"] == 0, np.nan, ds["scl"].astype("float32"))
    ok = scl.notnull() & ~scl.isin([0, 1, 3, 8, 9, 10, 11])
    return refl, ok, ds.time


def landsat_year(bbox, gb, year):
    cat = pc_catalog()
    items = list(cat.search(collections=["landsat-c2-l2"], bbox=bbox,
                            datetime=f"{year}-03-01/{year}-05-31",
                            query={"platform": {"in": ["landsat-8", "landsat-9"]},
                                   "eo:cloud_cover": {"lt": 60}}).items())
    if not items:
        return None
    ds = odc.stac.load(items, bands=["green", "red", "nir08", "swir16", "qa_pixel"],
                       groupby="solar_day", chunks={"x": 1024, "y": 1024}, **gb)
    sc = lambda b: xr.where(ds[b] == 0, np.nan, ds[b].astype("float32") * 2.75e-5 - 0.2)
    refl = {"green": sc("green"), "red": sc("red"), "nir": sc("nir08"), "swir16": sc("swir16")}
    qa = ds["qa_pixel"]
    # bits: 1 dilated cloud, 3 cloud, 4 shadow, 5 snow; 0 = fill
    bad = (qa == 0) | ((qa & 0b111010) > 0)
    return refl, ~bad, ds.time


def composite(refl, ok, time):
    g, r, n, s = (refl[b].where(ok) for b in ["green", "red", "nir", "swir16"])
    months = time.dt.month.values
    late = np.where(months == 5)[0]
    if len(late) == 0:
        late = np.arange(len(months))
    ndvi = nd(n, r)
    ndvi_may = ndvi.isel(time=late).median("time")
    # Rain-robust irrigation signal: 25th pct NDVI over Apr 15 - May 31. A pre-monsoon
    # shower greens rainfed land for days; an irrigated field stays green on most passes.
    # (Landsat check, block 56: May 2024 trees 0.60 vs 0.37 in 2019 -> 2024 was rain-green.)
    win = np.where((months == 5) | ((months == 4) & (time.dt.day.values >= 15)))[0]
    ndvi_p25 = ndvi.isel(time=win if len(win) else late).chunk({"time": -1}) \
        .quantile(0.25, dim="time", skipna=True)
    mndwi_max = nd(g, s).max("time")
    n_clear = (ok & n.notnull()).sum("time")
    i16 = lambda a: (a * 1000).round().fillna(-32768).astype("int16")
    return xr.Dataset({"ndvi_may": i16(ndvi_may), "ndvi_late_p25": i16(ndvi_p25.drop_vars("quantile")),
                       "mndwi_max": i16(mndwi_max),
                       "n_clear": n_clear.clip(max=255).astype("uint8")}).compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--years", default="2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026")
    a = ap.parse_args()

    bbox = blocks()[a.block]
    xy = snapped_bbox(bbox)
    gb = geobox_kwargs(xy)
    out = os.path.join(OUT_DIR, f"blk_{a.block:03d}.zarr")
    if os.path.exists(os.path.join(out, "zarr.json")) or os.path.exists(os.path.join(out, ".zmetadata")):
        print(a.block, "already done", flush=True)
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    es = pystac_client.Client.open(EARTH_SEARCH)

    layers = []
    for y in map(int, a.years.split(",")):
        got = landsat_year(bbox, gb, y) if y < 2018 else s2_year(es, bbox, gb, y)
        if got is None:
            print(a.block, y, "no scenes", flush=True)
            continue
        c = composite(*got).expand_dims(year=[y])
        c = c.drop_vars([v for v in c.coords if v not in ("year", "x", "y")])
        layers.append(c)
        print(a.block, y, "done", int((c.n_clear > 0).mean() * 100), "% observed", flush=True)

    wc = odc.stac.load(list(pc_catalog().search(collections=["esa-worldcover"], bbox=bbox,
                                                datetime="2021-01-01/2021-12-31").items()),
                       bands=["map"], groupby="solar_day", **gb)["map"].isel(time=0)
    ds = xr.concat(layers, dim="year")
    ds["worldcover"] = wc.drop_vars([v for v in wc.coords if v not in ("x", "y")]).astype("uint8")
    ds.attrs.update(crs=CRS, res=RES, block=a.block, bbox_lonlat=list(bbox))
    ds.to_zarr(out, mode="w")
    print("wrote", out, dict(ds.sizes), flush=True)


if __name__ == "__main__":
    main()
