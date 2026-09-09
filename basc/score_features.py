"""Score the connected-component detector against the AOI-mean baseline.

Everything measured on 2026-09-08 came from basc_series AOI *means*. These
features come from change.py: each pixel against its own seasonal history, the
surviving pixels grouped into connected components, then summarised per zone.
The comparison is the point -- same campuses, same rasters, same endmember
basis, different statistic.

Two zones:
  development   the whole parcel, as before
  building      inside the planned footprints (placebo footprints on negatives)

Component features are ALREADY anomalies against the pixel's own seasonal norm,
so the score is the value in the onset window, not a difference of differences.

  python score_features.py --pos results/featzone_pos16par \
                           --neg results/featzone_neg20
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from basc_discriminate import auc, boot_auc
from detect import CONFIRMED
from replay import ordn

FEATS = ["largest_component_m2", "total_changed_area_m2", "pct_zone_affected",
         "n_components", "veg_loss_area_m2", "new_soil_area_m2",
         "largest_component_rectangularity"]


def load(d):
    out = {}
    for fn in sorted(glob.glob(f"{d}/*.json")):
        j = json.load(open(fn))
        s = j["site"]
        by = {}
        for r in j["records"]:
            by.setdefault(r["zone"], {})[ordn(r["period"])] = r
        out[s["uid"]] = (s, ordn(s["start_period"]), by)
    return out


def onset_value(rows, o, feat, lo=-1, hi=2):
    v = [rows[k].get(feat) for k in range(o + lo, o + hi + 1)
         if k in rows and rows[k].get(feat) is not None]
    return float(np.mean(v)) if v else None


def first_confirmed(rows):
    for k in sorted(rows):
        if rows[k].get("detection_state") == CONFIRMED and rows[k].get("confirmed_period"):
            return ordn(rows[k]["confirmed_period"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--hit-k", type=int, default=6)
    a = ap.parse_args()
    P, N = load(a.pos), load(a.neg)
    print(f"positives {len(P)}   negatives {len(N)}\n")

    for zone in ("development", "building"):
        print(f"### zone: {zone}")
        print(f"{'feature':<36}{'n+':>4}{'n-':>4}{'AUC':>8}{'95% CI':>18}")
        print("-" * 70)
        for f in FEATS:
            vp = [onset_value(by.get(zone, {}), o, f) for _, o, by in P.values()]
            vn = [onset_value(by.get(zone, {}), o, f) for _, o, by in N.values()]
            vp = [v for v in vp if v is not None]
            vn = [v for v in vn if v is not None]
            if len(vp) < 3 or len(vn) < 3:
                print(f"{f:<36}{len(vp):>4}{len(vn):>4}   too few")
                continue
            A, lo, hi = boot_auc(vp, vn)
            print(f"{f:<36}{len(vp):>4}{len(vn):>4}{A:>8.3f}   [{lo:.3f},{hi:.3f}]")

        # scanning behaviour, straight from the causal states the features carry
        hits, early, miss = 0, 0, 0
        for _, o, by in P.values():
            c = first_confirmed(by.get(zone, {}))
            if c is None:
                miss += 1
            elif o - 2 <= c <= o + a.hit_k:
                hits += 1
            elif c < o - 2:
                early += 1
        fa = sum(1 for _, o, by in N.values() if first_confirmed(by.get(zone, {})) is not None)
        print(f"\n  scanning: detect {hits}/{len(P)}   early {early}   never {miss}"
              f"   negatives firing {fa}/{len(N)}")
        pa = sum(1 for _, o, by in P.values() if first_confirmed(by.get(zone, {})) is not None)
        prec = pa / (pa + fa) if pa + fa else float("nan")
        base = len(P) / (len(P) + len(N))
        print(f"  whether-precision {prec:.2f} vs base {base:.2f}  lift {prec/base:.2f}\n")


if __name__ == "__main__":
    main()
