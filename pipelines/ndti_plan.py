"""Choose the blocks scored locally (written before any NDTI target is looked at).

Eligible block: window cropland share >= 0.30 (WorldCover 2021, ndti_local.py cropfrac).
Within each split, blocks are drawn per administrative division in a fixed random order
(seed 340), with the number per division proportional to its eligible blocks (at least 2
where available).  Test blocks get seasons 2018-2022 (for climatology) + 2024, 2025;
train blocks 2018-2022; validation blocks 2023.

Usage: python ndti_plan.py --test 32 --train 32 --val 10
Writes ../results_v2/ndti_local/plan.json and ../local_data/harvest20/tasks.txt
"""
import argparse
import json
import os

import numpy as np

from mh_grid import blocks, split_of

HERE = os.path.dirname(os.path.abspath(__file__))
CF = os.path.join(HERE, "..", "local_data", "harvest20", "cropfrac.json")
OUT = os.path.join(HERE, "..", "results_v2", "ndti_local")
MIN_CROP = 0.30
SEASONS = {"test": [2018, 2019, 2020, 2021, 2022, 2024, 2025], "train": [2018, 2019, 2020, 2021, 2022],
           "val": [2023]}


def allocate(counts, k):
    total = sum(counts.values())
    alloc = {d: min(c, max(2, int(round(k * c / total)))) for d, c in counts.items() if c}
    floor = {d: min(2, counts[d]) for d in alloc}
    while sum(alloc.values()) > k:
        d = max((x for x in alloc if alloc[x] > floor[x]), key=lambda x: alloc[x] - counts[x] / total * k)
        alloc[d] -= 1
    while sum(alloc.values()) < k and any(alloc[d] < counts[d] for d in alloc):
        d = max((x for x in alloc if alloc[x] < counts[x]), key=lambda x: counts[x] / total * k - alloc[x])
        alloc[d] += 1
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=32)
    ap.add_argument("--train", type=int, default=32)
    ap.add_argument("--val", type=int, default=10)
    a = ap.parse_args()
    cf = {int(k): v for k, v in json.load(open(CF)).items()}
    rng = np.random.default_rng(340)
    plan = {"min_cropfrac": MIN_CROP, "division": {}, "cropfrac": {}, "eligible": {}}
    for split, k in (("test", a.test), ("train", a.train), ("val", a.val)):
        by_div = {}
        for b, _, d in blocks():
            if split_of(b) == split and cf.get(b, 0) >= MIN_CROP:
                by_div.setdefault(d, []).append(b)
        alloc = allocate({d: len(v) for d, v in by_div.items()}, k)
        chosen = []
        for d in sorted(by_div):
            order = rng.permutation(sorted(by_div[d]))
            chosen += [int(x) for x in order[:alloc.get(d, 0)]]
        plan[split] = sorted(chosen)
        plan["eligible"][split] = {d: len(v) for d, v in sorted(by_div.items())}
        for b in chosen:
            plan["division"][str(b)] = next(d for bb, _, d in blocks() if bb == b)
            plan["cropfrac"][str(b)] = cf[b]
    os.makedirs(OUT, exist_ok=True)
    json.dump(plan, open(os.path.join(OUT, "plan.json"), "w"), indent=1)
    with open(os.path.join(HERE, "..", "local_data", "harvest20", "tasks.txt"), "w") as fh:
        # test seasons first so scoring can start early; then train, then val
        for split in ("test", "train", "val"):
            for y in sorted(SEASONS[split], key=lambda y: (y not in (2024, 2025), y)):
                for b in plan[split]:
                    fh.write(f"{b} {y}\n")
    for s in ("test", "train", "val"):
        print(s, len(plan[s]), {d: sum(plan['division'][str(b)] == d for b in plan[s]) for d in plan['eligible'][s]},
              "eligible", plan["eligible"][s])


if __name__ == "__main__":
    main()
