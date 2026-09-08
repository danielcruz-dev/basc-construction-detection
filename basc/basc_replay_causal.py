"""Chronological replay: scan for the start instead of being handed it.

basc_discriminate.py centres its window on a KNOWN start. That measures whether
a starting campus looks different at its start -- not whether the start can be
found. A screen over the announcement pipeline has to make a call at every
fortnight, and the false alarms compound across every window it scans.

The change statistic here is the same seasonal comparison, made CAUSAL:

    delta(t) = mean NDVI over [t-3 .. t]  -  mean over the SAME four calendar
               fortnights one and two years earlier

Every term is at or before t. The section 3 window (lo=-1, hi=+2) reaches two
fortnights past the start and cannot be computed at t, which is exactly the
defect detect.py's docstring describes for the old persistence measure.

`changed` is then delta(t) > tau, and detect.run_causal turns the sequence of
per-period flags into provisional / confirmed calls with the same persistence
rules the detector already uses. tau is SWEPT, not chosen: picking one on these
36 campuses and reporting its score would be fitting on the test set.

Scoring:
  negatives  nothing was ever built, so ANY confirmed call is a false alarm.
  positives  a confirmation is a hit only if it lands in [start-2, start+hit_k];
             a confirmation before that is a false alarm, not an early detection.

  python basc_replay_causal.py --pos results/series_pos16par \
                               --neg results/series_neg20
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from detect import CONFIRMED, run_causal
from replay import ordn, unordn

PERIODS_PER_YEAR = 24
TRAIL = 3          # trailing window is [t-TRAIL .. t], four fortnights


def load(series_dir, min_clear):
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
            if r.get("clear_frac", r.get("valid_frac", 0)) < min_clear:
                continue
            if r.get("ndvi") is None:
                continue
            rows[ordn(r["period"])] = r
        periods = [ordn(r["period"]) for r in j["series"]]
        out[s.get("uid", os.path.basename(fn)[:-5])] = (s, ordn(s["start_period"]),
                                                        rows, periods)
    return out


def causal_delta(rows, t, met="ndvi"):
    """Trailing-window seasonal delta at t, using only observations <= t."""
    def win(end):
        v = [rows[k][met] for k in range(end - TRAIL, end + 1) if k in rows]
        return float(np.mean(v)) if v else None
    a = win(t)
    if a is None:
        return None
    ctrl = [win(t - PERIODS_PER_YEAR * y) for y in (1, 2)]
    ctrl = [c for c in ctrl if c is not None]
    if not ctrl:
        return None
    return a - float(np.mean(ctrl))


def observations(rows, periods, tau, sign=-1):
    """Per-period causal flags. sign=-1 because a start pushes NDVI DOWN."""
    obs = []
    for t in periods:
        d = causal_delta(rows, t)
        if t not in rows or d is None:
            obs.append({"period": unordn(t), "clear": False, "changed": False})
        else:
            obs.append({"period": unordn(t), "clear": True,
                        "changed": bool(sign * d > tau)})
    return obs


def first_confirmed(dets):
    for d in dets:
        if d.state == CONFIRMED and d.confirmed_period:
            return ordn(d.confirmed_period)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--min-clear", type=float, default=0.6)
    ap.add_argument("--hit-k", type=int, default=6,
                    help="fortnights after the start within which a confirmation "
                         "still counts as detecting THAT start")
    ap.add_argument("--confirm-n", type=int, default=2)
    a = ap.parse_args()

    P = load(a.pos, a.min_clear)
    N = load(a.neg, a.min_clear)
    print(f"positives {len(P)}   negatives {len(N)}   confirm_n={a.confirm_n}   "
          f"hit window [start-2, start+{a.hit_k}]\n")

    cfg = {"confirm_n": a.confirm_n}
    print(f"{'tau':>7}{'detect':>8}{'median':>8}{'FA neg':>8}{'FA pos':>8}"
          f"{'per-win':>9}{'campus':>8}")
    print(f"{'':>7}{'rate':>8}{'lag':>8}{'campus':>8}{'early':>8}"
          f"{'FA rate':>9}{'prec':>8}")
    print("-" * 56)
    for tau in (0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18):
        hits, lags, early = 0, [], 0
        for uid, (s, o, rows, periods) in P.items():
            dets = run_causal(observations(rows, periods, tau), cfg)
            c = first_confirmed(dets)
            if c is None:
                continue
            if o - 2 <= c <= o + a.hit_k:
                hits += 1
                lags.append(c - o)
            elif c < o - 2:
                early += 1
        fa_neg, win_flags, win_clear = 0, 0, 0
        for uid, (s, o, rows, periods) in N.items():
            obs = observations(rows, periods, tau)
            win_flags += sum(1 for ob in obs if ob["clear"] and ob["changed"])
            win_clear += sum(1 for ob in obs if ob["clear"])
            if first_confirmed(run_causal(obs, cfg)) is not None:
                fa_neg += 1
        det = hits / len(P)
        fa_rate = fa_neg / len(N)
        perwin = win_flags / win_clear if win_clear else float("nan")
        prec = hits / (hits + fa_neg + early) if (hits + fa_neg + early) else float("nan")
        med = f"{np.median(lags):+.0f}" if lags else "—"
        print(f"{tau:>7.2f}{det:>8.2f}{med:>8}{fa_rate:>8.2f}"
              f"{early/len(P):>8.2f}{perwin:>9.3f}{prec:>8.2f}")

    print("\ndetect rate = positives confirmed inside the hit window")
    print("median lag  = fortnights from true start to confirmation (+ is late)")
    print("FA neg campus = share of NON-STARTING campuses with any confirmed call")
    print("FA pos early  = positives confirmed more than 2 fortnights BEFORE the start")
    print("per-win FA rate = share of clear negative fortnights flagged changed")
    print("campus prec = hits / (hits + negative FAs + early FAs), at base rate "
          f"{len(P)/(len(P)+len(N)):.2f}")


if __name__ == "__main__":
    main()
