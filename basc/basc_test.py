"""
Does the fraction signal actually move at the verified construction start?

Design: within-site, season-matched, with a placebo arm.

  onset    = the 4 fortnights [start-1 .. start+2]
  control  = the SAME calendar fortnights one and two years EARLIER

Each site is therefore its own control, matched on season. That removes the two
confounds this project keeps tripping over -- site geography and the annual
vegetation cycle -- without needing a separate negative-parcel sweep. A bare
field in February is compared against the same field in February.

The placebo arm reruns the identical statistic against a start date shifted by
a whole number of years, so the onset window lands on the same season but the
wrong year. If the real statistic is not clearly larger than the placebo, the
"signal" is an artefact of the window arithmetic rather than of construction --
which is exactly the failure mode that inflated the v1/v4 hit rates.

  python basc_test.py --series results/series_sp16
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from replay import ordn

PERIODS_PER_YEAR = 24
METRICS = ["soil", "high_albedo", "vegetation", "low_albedo",
           "ndvi", "ndbi", "swir"]


def load(series_dir, min_clear):
    out = {}
    for fn in sorted(glob.glob(f"{series_dir}/*.json")):
        if fn.endswith("_endmembers.json"):
            continue
        j = json.load(open(fn))
        s = j["site"]
        rows = {}
        for r in j["series"]:
            if r.get("n_valid", 0) < 20:
                continue
            # clear_frac normalises by AOI shape; fall back for older runs
            c = r.get("clear_frac", r.get("valid_frac", 0))
            if c < min_clear:
                continue
            rows[ordn(r["period"])] = r
        out[os.path.basename(fn)[:-5]] = (s, ordn(s["start_period"]), rows)
    return out


def window(rows, o, met, lo=-1, hi=2):
    v = [rows[k][met] for k in range(o + lo, o + hi + 1)
         if k in rows and rows[k].get(met) is not None]
    return float(np.mean(v)) if v else None


def delta(rows, o, met, years=(1, 2)):
    """Onset minus the same season in earlier years. None if unmatched."""
    a = window(rows, o, met)
    if a is None:
        return None
    ctrl = [window(rows, o - PERIODS_PER_YEAR * y, met) for y in years]
    ctrl = [c for c in ctrl if c is not None]
    if not ctrl:
        return None
    return a - float(np.mean(ctrl))


def boot(x, n=20000, seed=0):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--min-clear", type=float, default=0.6)
    ap.add_argument("--placebo-years", type=int, default=1,
                    help="shift the start by this many YEARS for the placebo arm, "
                         "so the window keeps its season but loses its event")
    a = ap.parse_args()

    D = load(a.series, a.min_clear)
    print(f"sites loaded: {len(D)}   (clear_frac >= {a.min_clear})\n")

    print(f"{'metric':<13}{'n':>4}{'REAL mean':>11}{'95% CI':>20}"
          f"{'PLACEBO mean':>14}{'95% CI':>20}   verdict")
    print("-" * 96)
    for met in METRICS:
        real, plac = [], []
        for k, (s, o, rows) in D.items():
            real.append(delta(rows, o, met))
            plac.append(delta(rows, o - PERIODS_PER_YEAR * a.placebo_years, met))
        rm, rlo, rhi, rn = boot(real)
        pm, plo, phi, pn = boot(plac, seed=1)
        if np.isnan(rm):
            print(f"{met:<13}{rn:>4}   insufficient matched windows")
            continue
        sig = (rlo > 0 or rhi < 0)
        beats = sig and (abs(rm) > abs(pm)) and (plo <= 0 <= phi)
        verdict = "REAL" if beats else ("sig, but placebo too" if sig else "no signal")
        print(f"{met:<13}{rn:>4}{rm:>+11.4f}  [{rlo:+.4f},{rhi:+.4f}]"
              f"{pm:>+14.4f}  [{plo:+.4f},{phi:+.4f}]   {verdict}")

    print("\nper-site onset-minus-same-season deltas:")
    print(f"  {'site':<34}{'start':<11}{'d_soil':>9}{'d_high':>9}{'d_ndvi':>9}")
    for k, (s, o, rows) in sorted(D.items()):
        ds, dh, dn = (delta(rows, o, m) for m in ("soil", "high_albedo", "ndvi"))
        f = lambda v: f"{v:+9.3f}" if v is not None else "        —"
        print(f"  {s['name'][:34]:<34}{s['start_period']:<11}{f(ds)}{f(dh)}{f(dn)}")


if __name__ == "__main__":
    main()
