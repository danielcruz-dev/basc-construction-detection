"""Score the connected-component detector against the AOI-mean baseline.

Everything measured on 2026-09-08 came from basc_series AOI *means*. These
features come from change.py: each pixel against its own seasonal history, the
surviving pixels grouped into connected components, then summarised per zone.
The comparison is the point -- same campuses, same rasters, same endmember
basis, different statistic.

Three zones:
  development   the whole parcel, as before
  building      inside the planned footprints (placebo footprints on negatives)
  context       the control ring: parcel+outer buffer minus the parcel, present
                only on grids fetched with --outer-buffer-m and injected by
                inject_zones.py. Nobody builds there; anything firing there is
                regional, not construction.

Component features are ALREADY anomalies against the pixel's own seasonal norm,
so the score is the value in the onset window, not a difference of differences.

The RING GATE. A per-pixel seasonal baseline cancels what a pixel usually does
at this time of year; it cannot cancel what this year does to every pixel -- a
drought, a late snowmelt, a ploughing season that shifted a fortnight. Those
fire the parcel and the ring together. The gate says a period counts as changed
only if the zone's affected fraction exceeds the ring's by --ring-ratio, i.e.
the change is CONCENTRATED on the site and not merely present in the region.
The gated series is then fed to the identical causal persistence rules, so
everything stays at-or-before t. The ratio is swept, not chosen: choosing it on
these 36 campuses would fit it on the test set.

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
from detect import CONFIRMED, run_causal
from replay import ordn, unordn

FEATS = ["largest_component_m2", "total_changed_area_m2", "pct_zone_affected",
         "n_components", "veg_loss_area_m2", "new_soil_area_m2",
         "largest_component_rectangularity"]
MMU_M2 = 8000.0            # same evidence bar basc_features uses for `changed`
CONTEXT = "context"


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


def ring_diff_rows(zone_rows, ctx_rows):
    """pct_zone_affected minus the ring's, period by period."""
    out = {}
    for k, r in zone_rows.items():
        c = ctx_rows.get(k)
        if c is None or r.get("pct_zone_affected") is None or c.get("pct_zone_affected") is None:
            continue
        out[k] = {"ring_diff_pct": r["pct_zone_affected"] - c["pct_zone_affected"]}
    return out


def first_confirmed(rows):
    for k in sorted(rows):
        if rows[k].get("detection_state") == CONFIRMED and rows[k].get("confirmed_period"):
            return ordn(rows[k]["confirmed_period"])
    return None


def gated_first_confirmed(zone_rows, ctx_rows, ratio):
    """Re-run the causal detector on a ring-gated `changed` series.

    changed(t) = zone changed area >= MMU  and  zone_pct(t) >= ratio * ring_pct(t)

    A ring with 0% affected gates nothing (any site change passes); a ring as
    active as the zone gates everything at ratio >= 1. Only periods present in
    BOTH zones are used, so the clear/unclear pattern is shared.
    """
    series = []
    for k in sorted(zone_rows):
        r, c = zone_rows[k], ctx_rows.get(k)
        if c is None:
            continue
        zp, cp = r.get("pct_zone_affected") or 0.0, c.get("pct_zone_affected") or 0.0
        changed = ((r.get("total_changed_area_m2") or 0.0) >= MMU_M2
                   and zp >= ratio * cp)
        series.append({"period": unordn(k), "clear": True, "changed": changed})
    for d in run_causal(series):
        if d.state == CONFIRMED and d.confirmed_period:
            return ordn(d.confirmed_period)
    return None


def scan_summary(P, N, first_fn, hit_k):
    hits, early, miss = 0, 0, 0
    for _, o, by in P.values():
        c = first_fn(o, by)
        if c is None:
            miss += 1
        elif o - 2 <= c <= o + hit_k:
            hits += 1
        elif c < o - 2:
            early += 1
    fa = sum(1 for _, o, by in N.values() if first_fn(o, by) is not None)
    pa = sum(1 for _, o, by in P.values() if first_fn(o, by) is not None)
    prec = pa / (pa + fa) if pa + fa else float("nan")
    base = len(P) / (len(P) + len(N))
    return dict(detect=hits, early=early, never=miss, fa=fa, fired=pa,
                prec=prec, base=base, lift=prec / base if base else float("nan"))


def print_scan(tag, s, nP, nN):
    print(f"  {tag:<22} detect {s['detect']:2d}/{nP}   early {s['early']:2d}"
          f"   never {s['never']:2d}   negatives firing {s['fa']:2d}/{nN}"
          f"   precision {s['prec']:.2f} vs base {s['base']:.2f}  lift {s['lift']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--hit-k", type=int, default=6)
    ap.add_argument("--ring-ratios", default="1,1.5,2,3,5",
                    help="ring-gate ratios to sweep; only used when a context "
                         "zone is present")
    a = ap.parse_args()
    P, N = load(a.pos), load(a.neg)
    print(f"positives {len(P)}   negatives {len(N)}\n")
    have_ring = any(CONTEXT in by for _, _, by in list(P.values()) + list(N.values()))
    zones = ("development", "building") + ((CONTEXT,) if have_ring else ())
    ratios = [float(x) for x in a.ring_ratios.split(",") if x]

    for zone in zones:
        print(f"### zone: {zone}")
        print(f"{'feature':<36}{'n+':>4}{'n-':>4}{'AUC':>8}{'95% CI':>18}")
        print("-" * 70)
        feats = list(FEATS)
        if have_ring and zone != CONTEXT:
            feats.append("ring_diff_pct")
        for f in feats:
            if f == "ring_diff_pct":
                vp = [onset_value(ring_diff_rows(by.get(zone, {}), by.get(CONTEXT, {})), o, f)
                      for _, o, by in P.values()]
                vn = [onset_value(ring_diff_rows(by.get(zone, {}), by.get(CONTEXT, {})), o, f)
                      for _, o, by in N.values()]
            else:
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
        print()
        raw = scan_summary(P, N, lambda o, by: first_confirmed(by.get(zone, {})), a.hit_k)
        print_scan("scanning: ungated", raw, len(P), len(N))
        if have_ring and zone != CONTEXT:
            for r in ratios:
                g = scan_summary(
                    P, N,
                    lambda o, by, r=r: gated_first_confirmed(by.get(zone, {}),
                                                             by.get(CONTEXT, {}), r),
                    a.hit_k)
                print_scan(f"ring-gated ratio {r:g}", g, len(P), len(N))
        print()


if __name__ == "__main__":
    main()
