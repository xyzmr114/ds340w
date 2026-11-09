"""All-Maharashtra block grid and spatial split (replaces the Vidarbha-only 209-block split).

Lattice: 0.25 deg cells, 34 cols x 27 rows from (72.50 E, 15.50 N) -- the same lattice as grid.py.
A cell is a study block if at least 10% of it lies inside the state boundary
(data/maharashtra.geojson, Census 2011 districts).  Block id = row * 34 + col.

Split: cells are grouped into 2x2 super-blocks (~55 km); super-blocks are shuffled with a
fixed seed (340) and assigned 20% test, 10% validation, 70% train.  The split is decided
before any target is looked at.

Region: administrative division of the district containing the block centre
(nearest district when the centre falls outside the state).
"""
import json
import os

import numpy as np

W, S, NCOL, NROW, DEG = 72.5, 15.5, 34, 27, 0.25
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIVISION = {
    "Konkan": ["Mumbai", "Palghar", "Thane", "Raigad", "Ratnagiri", "Sindhudurg"],
    "Pune": ["Pune", "Satara", "Sangli", "Kolhapur", "Solapur"],
    "Nashik": ["Nashik", "Dhule", "Nandurbar", "Jalgaon", "Ahmednagar"],
    "Marathwada": ["Aurangabad", "Jalna", "Beed", "Latur", "Osmanabad", "Nanded", "Parbhani", "Hingoli"],
    "Vidarbha": ["Amravati", "Akola", "Washim", "Buldhana", "Yavatmal", "Nagpur", "Wardha", "Bhandara",
                 "Gondia", "Chandrapur", "Gadchiroli"],
}
DIV_OF = {d: k for k, v in DIVISION.items() for d in v}


def bbox(b):
    w = W + (b % NCOL) * DEG
    s = S + (b // NCOL) * DEG
    return (round(w, 4), round(s, 4), round(w + DEG, 4), round(s + DEG, 4))


def superblock(b):
    return (b // NCOL) // 2 * 100 + (b % NCOL) // 2


def _state():
    from shapely.geometry import shape
    gj = json.load(open(os.path.join(ROOT, "data", "maharashtra.geojson"), encoding="utf-8"))
    return [(f["properties"]["district"], shape(f["geometry"]).buffer(0)) for f in gj["features"]]


_CACHE = {}


def blocks():
    """[(id, state_fraction, division)] for every study block."""
    if "blocks" in _CACHE:
        return _CACHE["blocks"]
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    dists = _state()
    state = unary_union([g for _, g in dists])
    out = []
    for b in range(NCOL * NROW):
        w, s, e, n = bbox(b)
        fr = state.intersection(box(w, s, e, n)).area / DEG ** 2
        if fr < 0.10:
            continue
        c = Point((w + e) / 2, (s + n) / 2)
        name = min(dists, key=lambda d: d[1].distance(c))[0]
        out.append((b, round(fr, 3), DIV_OF[name]))
    _CACHE["blocks"] = out
    return out


def split_of(b, seed=340):
    if "role" not in _CACHE:
        sbs = sorted({superblock(i) for i, _, _ in blocks()})
        perm = np.random.default_rng(seed).permutation(sbs)
        n = len(perm)
        _CACHE["role"] = {s: ("test" if k < round(.2 * n) else "val" if k < round(.3 * n) else "train")
                          for k, s in enumerate(perm)}
    return _CACHE["role"][superblock(b)]


if __name__ == "__main__":
    from collections import Counter
    bl = blocks()
    print(len(bl), "blocks,", len({superblock(b) for b, _, _ in bl}), "super-blocks")
    print(Counter(split_of(b) for b, _, _ in bl))
    print(Counter((d, split_of(b)) for b, _, d in bl))
