"""Can the fraction signal SEPARATE starting campuses from non-starting ones?

basc_test.py answers a strictly weaker question. It is within-site: it asks
whether a site changes more at its own start than at a fake date a year earlier.
A detector that fires on every campus every fortnight passes that test. Ranking
a review queue needs the between-class question, and that needs negatives.

Design
------
Negatives are campuses whose recorded pipeline stage is pre-construction
(Announcement / Land Bank / Delayed), that carry no verified start, and that
have a polygon. Sixteen of the twenty are matched 1:1 to an sp16 positive on
log parcel area and share that positive's start period, so the primary analysis
is PAIRED on both area and season -- the two confounds that dominate here. The
remaining four close the area distribution (KS 0.100) and enter only the
unpaired arms.

A negative has no start, so its window is anchored on a PSEUDO-start. That is an
anchor, not a label: it says where to centre the season-matched window, nothing
more. grid.json records start_is_pseudo so the two can never be confused.

Both classes must be read on ONE endmember basis and ONE AOI kind, or the
comparison measures the basis and the AOI instead of construction -- the exact
artefact recorded as mistake 1 in the 2026-09-07 log, where a tighter AOI
appeared to sharpen Digital's signal only because it had its own basis.

  python basc_discriminate.py --pos results/series_pos16par \
                              --neg results/series_neg20 \
                              --pairs results/pairs_neg20.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from replay import ordn

PERIODS_PER_YEAR = 24
METRICS = ["soil", "high_albedo", "vegetation", "low_albedo", "ndvi", "ndbi", "swir"]

# Direction a construction start is EXPECTED to move each metric, fixed from the
# paper and from the within-site result in section 3 of the 2026-09-07 log. It is
# hard-coded on purpose: taking the sign from the positives in this same sample
# would fit the direction on the test set and inflate every AUC toward 1.
EXPECTED = {"soil": +1, "high_albedo": +1, "vegetation": -1, "low_albedo": +1,
            "ndvi": -1, "ndbi": +1, "swir": +1}


def load(series_dir, min_clear):
    """uid -> (site record, start ordinal, {period ordinal: row})."""
    out = {}
    for fn in sorted(glob.glob(f"{series_dir}/*.json")):
        if os.path.basename(fn).startswith("_"):
            continue
        j = json.load(open(fn))
        s = j["site"]
        rows = {}
        for r in j["series"]:
            if r.get("n_valid", 0) < 20:
                continue
            c = r.get("clear_frac", r.get("valid_frac", 0))
            if c < min_clear:
                continue
            rows[ordn(r["period"])] = r
        out[s.get("uid", os.path.basename(fn)[:-5])] = (s, ordn(s["start_period"]), rows)
    return out


def window(rows, o, met, lo=-1, hi=2):
    v = [rows[k][met] for k in range(o + lo, o + hi + 1)
         if k in rows and rows[k].get(met) is not None]
    return float(np.mean(v)) if v else None


def delta(rows, o, met, years=(1, 2)):
    """Onset minus the same calendar fortnights in earlier years."""
    a = window(rows, o, met)
    if a is None:
        return None
    ctrl = [window(rows, o - PERIODS_PER_YEAR * y, met) for y in years]
    ctrl = [c for c in ctrl if c is not None]
    if not ctrl:
        return None
    return a - float(np.mean(ctrl))


def auc(pos, neg):
    """P(a positive outranks a negative), ties counted as half."""
    pos = [v for v in pos if v is not None]
    neg = [v for v in neg if v is not None]
    if not pos or not neg:
        return None
    w = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def boot_auc(pos, neg, n=10000, seed=0):
    pos = [v for v in pos if v is not None]
    neg = [v for v in neg if v is not None]
    if len(pos) < 3 or len(neg) < 3:
        return None, None, None
    rng = np.random.default_rng(seed)
    P, N = np.asarray(pos), np.asarray(neg)
    out = []
    for _ in range(n):
        out.append(auc(rng.choice(P, len(P), replace=True).tolist(),
                       rng.choice(N, len(N), replace=True).tolist()))
    return auc(pos, neg), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_paired(d, n=20000, seed=0):
    """Paired bootstrap over MATCHED PAIRS, resampling pairs not sites.

    Resampling the two classes independently would discard the matching and
    overstate the interval; the pair is the unit of replication here.
    """
    d = np.asarray([v for v in d if v is not None], float)
    if len(d) < 3:
        return np.nan, np.nan, np.nan, len(d)
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(d)


def lift_at_k(scored, k):
    """Precision in the top k over the base rate -- production's queue bar.

    scored: list of (score, is_positive), higher score = more construction-like.
    """
    scored = [s for s in scored if s[0] is not None]
    if not scored:
        return None, None, 0
    scored.sort(key=lambda t: -t[0])
    k = min(k, len(scored))
    prec = sum(1 for _, y in scored[:k] if y) / k
    base = sum(1 for _, y in scored if y) / len(scored)
    return prec, (prec / base if base else None), k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True, help="series dir, verified starts")
    ap.add_argument("--neg", required=True, help="series dir, pseudo-started negatives")
    ap.add_argument("--pairs", default="", help="JSON {neg_uid: pos_uid} for the paired arm")
    ap.add_argument("--min-clear", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=10)
    a = ap.parse_args()

    P, N = load(a.pos, a.min_clear), load(a.neg, a.min_clear)

    # Refuse to run on a mixed-up pair of directories. Every precision, AUC and
    # lift number below is meaningless if a pseudo-started campus is counted as a
    # verified start, and the failure is silent -- the arithmetic still completes.
    bad_p = [s["name"] for s, _, _ in P.values() if s.get("start_is_pseudo")]
    if bad_p:
        raise SystemExit(f"--pos contains {len(bad_p)} PSEUDO-started campuses "
                         f"(e.g. {bad_p[:3]}). Those are negatives, not positives.")
    unflagged = [s["name"] for s, _, _ in N.values() if not s.get("start_is_pseudo")]
    if unflagged:
        raise SystemExit(f"--neg contains {len(unflagged)} campuses NOT flagged "
                         f"start_is_pseudo (e.g. {unflagged[:3]}). Refusing: a verified "
                         f"start in the negative arm destroys the base rate.")
    overlap = set(P) & set(N)
    if overlap:
        raise SystemExit(f"{len(overlap)} campuses appear in BOTH arms: "
                         f"{[u[:8] for u in list(overlap)[:5]]}")

    print(f"positives {len(P)}   negatives {len(N)}   (clear_frac >= {a.min_clear})")
    pairs = json.load(open(a.pairs)) if a.pairs else {}
    pairs = {n: p for n, p in pairs.items() if n in N and p in P}
    print(f"matched pairs usable: {len(pairs)}\n")

    print("UNPAIRED -- can the metric rank a start above a non-start?")
    print(f"{'metric':<13}{'dir':>4}{'n+':>4}{'n-':>4}{'pos mean':>11}{'neg mean':>11}"
          f"{'AUC':>8}{'95% CI':>18}")
    print("-" * 82)
    auc_by_met = {}
    for met in METRICS:
        sgn = EXPECTED[met]
        dp = [delta(r, o, met) for _, o, r in P.values()]
        dn = [delta(r, o, met) for _, o, r in N.values()]
        # orient so that "more construction-like" is always larger
        op = [sgn * v for v in dp if v is not None]
        on = [sgn * v for v in dn if v is not None]
        A, lo, hi = boot_auc(op, on)
        auc_by_met[met] = A
        mp = np.mean([v for v in dp if v is not None]) if any(v is not None for v in dp) else np.nan
        mn = np.mean([v for v in dn if v is not None]) if any(v is not None for v in dn) else np.nan
        ci = f"[{lo:.3f},{hi:.3f}]" if A is not None else "—"
        As = f"{A:.3f}" if A is not None else "—"
        print(f"{met:<13}{sgn:>+4}{len(op):>4}{len(on):>4}{mp:>+11.4f}{mn:>+11.4f}{As:>8}{ci:>18}")

    if pairs:
        print("\nPAIRED -- matched on log parcel area and start period")
        print(f"{'metric':<13}{'n':>4}{'pos-neg':>11}{'95% CI':>22}   verdict")
        print("-" * 66)
        for met in METRICS:
            sgn = EXPECTED[met]
            d = []
            for nuid, puid in pairs.items():
                _, po, pr = P[puid]
                _, no, nr = N[nuid]
                a1, a2 = delta(pr, po, met), delta(nr, no, met)
                d.append(sgn * (a1 - a2) if (a1 is not None and a2 is not None) else None)
            m, lo, hi, n = boot_paired(d)
            if np.isnan(m):
                print(f"{met:<13}{n:>4}   too few matched pairs")
                continue
            v = "SEPARATES" if lo > 0 else ("wrong sign" if hi < 0 else "no separation")
            print(f"{met:<13}{n:>4}{m:>+11.4f}  [{lo:+.4f},{hi:+.4f}]   {v}")

    print(f"\nQUEUE -- precision and lift in the top {a.topk}")
    print(f"{'metric':<13}{'prec@k':>9}{'lift':>8}{'k':>5}")
    print("-" * 35)
    base = len(P) / (len(P) + len(N))
    for met in METRICS:
        sgn = EXPECTED[met]
        scored = [(sgn * v if v is not None else None, True)
                  for v in (delta(r, o, met) for _, o, r in P.values())]
        scored += [(sgn * v if v is not None else None, False)
                   for v in (delta(r, o, met) for _, o, r in N.values())]
        prec, lift, k = lift_at_k(scored, a.topk)
        if prec is None:
            print(f"{met:<13}      —")
            continue
        print(f"{met:<13}{prec:>9.3f}{lift:>8.2f}{k:>5}")
    print(f"\nbase rate {base:.3f}  ({len(P)} positives of {len(P)+len(N)})")
    print("Production's review-queue bar recorded on 2026-09-04 is lift > 1.1x.")


if __name__ == "__main__":
    main()
