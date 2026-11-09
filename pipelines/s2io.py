"""Sentinel-2 L2A loading from Earth Search, with verified radiometry.

Checked 2026-10-14 by reading raw COG pixels over tile 44QLJ:
  * `sentinel-2-l2a`: pixel DNs are ALREADY harmonised (no +1000), for every
    processing baseline, even though item metadata advertises offset -0.1.
    2024 (05.10) red median ~1190 vs 2019 (02.12 / 05.00) ~1560. Apply NO offset.
  * `sentinel-2-c1-l2a`: DNs DO include +1000 (2024 red median ~2185). Apply -0.1.
  * The same acquisition can appear twice in `sentinel-2-l2a` (original + 05.00
    reprocessing); duplicates are dropped, keeping the highest baseline.

`load_s2` returns float32 surface reflectance (NaN = nodata) plus the raw SCL band
(float, NaN = nodata).
"""
# Verified offsets per collection (see docstring); do not trust item metadata.
OFFSET = {"sentinel-2-l2a": 0.0, "sentinel-2-c1-l2a": -0.1}
import numpy as np
import odc.stac
import xarray as xr

SCL_BAD = [0, 1, 3, 8, 9, 10, 11]


def _key(item):
    p = item.properties
    tile = p.get("grid:code") or p.get("s2:mgrs_tile") or item.id.split("_")[1]
    return tile, item.datetime.strftime("%Y%m%dT%H%M")


def _baseline(item):
    try:
        return float(item.properties.get("s2:processing_baseline", "0"))
    except ValueError:
        return 0.0


def search_s2(catalog, bbox, datetime, max_cloud=None, collection="sentinel-2-l2a"):
    query = {"eo:cloud_cover": {"lt": max_cloud}} if max_cloud is not None else {}
    items = catalog.search(collections=[collection], bbox=bbox, datetime=datetime,
                           query=query).item_collection()
    best = {}
    for it in items:
        # Some old-baseline (00.01) items point at ESA's requester-pays JP2 bucket
        # (s3://sentinel-s2-l2a/...), which anonymous reads cannot open. Keep free COGs only.
        if "sentinel-cogs" not in it.assets.get("red", it.assets.get("B04")).href:
            continue
        k = _key(it)
        if k not in best or _baseline(it) > _baseline(best[k]):
            best[k] = it
    return list(best.values())


def load_s2(items, bands, bbox, res, crs, chunks=None, collection="sentinel-2-l2a"):
    """bands: reflectance band names (e.g. red, nir, swir16); SCL is always added."""
    if not items:
        return None
    offset = OFFSET[collection]
    ds = odc.stac.load(items, bands=list(bands) + ["scl"], bbox=bbox, resolution=res,
                       crs=crs, groupby="solar_day", chunks=chunks or {"x": 1024, "y": 1024})
    out = {}
    for b in bands:
        raw = ds[b]
        out[b] = xr.where(raw == 0, np.nan, raw.astype("float32") * 1e-4 + offset).clip(min=0)
    out["scl"] = xr.where(ds["scl"] == 0, np.nan, ds["scl"].astype("float32"))
    return xr.Dataset(out)


def clear_mask(ds):
    return ds["scl"].notnull() & ~ds["scl"].isin(SCL_BAD)


def nd(a, b):
    return (a - b) / (a + b + 1e-6)
