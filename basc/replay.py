"""
Offline replay of every candidate detector against the verified start dates.

The point of this file is that no detector should be chosen by argument. Each
variant below is run over the SAME cached index series, on the SAME campuses,
and scored against the construction start dates verified by eye in
`campus_coords_ui.html` -- not against `EVENT_DATE`, 52.8% of which fall on the
1st or 15th of a month and therefore cannot resolve a fortnight.

Variants:
  v1        the deployed threshold rule (acd-run-detection, pre-v3)
  v3        v1 + robust trailing z + YoY guard, exactly as written today
  prod      detect-construction-start: weighted z composite + persistence
  v4        the proposal -- see det_v4 for what changed and why
  v5        scored, thresholded; built from measured feature separability

VERDICT as of 2026-09-03, on 189 campuses with complete history, thresholds
picked on a train half and reported on a held-out test half
(`curve.py`, `significance.py`, paired bootstrap over campuses):

  v3 IS A REGRESSION. Against v1 on the full population it catches 8.4 fewer
  starts per 100 campuses [-14.5, -2.2] AND raises false alarms by 2.7 points
  [+1.1, +4.3]. Both intervals exclude zero. It is strictly dominated, and it
  is what is deployed. It was shipped on a hit-rate edge measured at n=23.

  v4 IS THE BEST CANDIDATE. Against v1 on held-out test: +9.7 hits per 100
  [+3.2, +17.2], and it calls the start 0.34 half-months earlier [-0.66,
  -0.07] -- both significant -- for +4.1 points of false alarm [+2.4, +6.1].
  A real trade, not a free win: shipping it is a cost-ratio decision.

  v5 IS NOT AN IMPROVEMENT. Indistinguishable from both v1 and v4 on hit rate,
  false alarm AND timeliness -- every interval straddles zero. Its features
  separate better in isolation (see the AUC table below) and that did not
  survive into decision quality. It is kept because it is the only variant
  with a threshold, so it is the only one that can be compared at matched
  cost; it is not kept because it won anything.

Every detector sees one campus's level series and returns the FIRST period it
would have fired, so the score is a detection DATE error in half-months, which
is the quantity the whole exercise is about.

Coverage honesty: the stats cache was built around the RECORDED dates, so a
campus whose date moved may not have a contiguous run under its corrected date.
Each detector reports the n it could actually be computed on; they are not
silently compared across different populations.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# ── period arithmetic (same half-month grid as production) ────────────────────

def period_of(datestr: str) -> str:
    y, m, d = map(int, datestr.split("-")[:3])
    return f"{y:04d}-{m:02d}-{'H1' if d <= 15 else 'H2'}"


def ordn(period: str) -> int:
    y, m, h = period.split("-")
    return int(y) * 24 + (int(m) - 1) * 2 + (0 if h == "H1" else 1)


def unordn(o: int) -> str:
    y, rem = divmod(o, 24)
    m, h = divmod(rem, 2)
    return f"{y:04d}-{m + 1:02d}-{'H1' if h == 0 else 'H2'}"


# ── inputs ────────────────────────────────────────────────────────────────────

def load_series() -> dict[str, dict[int, dict]]:
    """uid -> {period ordinal: {'ndbi','swir','ndvi', 'dist':…}} from the cache."""
    with open(os.path.join(HERE, "stats_cache.json")) as fh:
        cache = json.load(fh)
    out: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for key, val in cache.items():
        src, rest = key.split(":", 1)
        uid, start, _end = rest.split("|")
        if src != "parcel" or not val:
            continue
        out[uid][ordn(period_of(start))] = val
    return out


# Latest date with imagery on the ground. A "verified" start after this cannot
# have been checked by eye against a scene that does not exist yet.
DATA_CUTOFF = dt.date(2026, 9, 3)
OBSERVABLE_MARGIN = 2      # the onset band runs to +2; it must be observable


def load_truth(strict: bool = True, report: bool = False) -> dict[str, int]:
    """uid -> corrected start period ordinal.

    Three classes of unusable label are dropped, not two:

      impossible years   0001-01-01, 0202-02-26, 2222-02-02 -- placeholder rows.
      FUTURE dates       a start after DATA_CUTOFF. 2026-10-01 sits a month past
                         today and passed the old year filter untouched, so it
                         entered every evaluation as a real campus whose entire
                         onset band is unobservable.
      UNOBSERVABLE tail  a start so recent that its onset band (+2 periods, six
                         weeks) has not happened yet. Scoring these counts
                         "no imagery" as "detector said no", which silently
                         penalises every detector on the newest campuses.

    strict=False restores the old permissive behaviour for comparison only.
    """
    with open(os.path.join(RESULTS, "acd_ui_corrections.json")) as fh:
        dates = json.load(fh)["dates"]
    cutoff_period = ordn(period_of(DATA_CUTOFF.isoformat()))
    truth: dict[str, int] = {}
    dropped = {"unparseable": 0, "impossible_year": 0, "future": 0,
               "unobservable": 0}
    for uid, val in dates.items():
        try:
            d = dt.date.fromisoformat(str(val).strip())
        except Exception:
            dropped["unparseable"] += 1
            continue
        if not (2015 <= d.year <= 2027):
            dropped["impossible_year"] += 1
            continue
        o = ordn(period_of(d.isoformat()))
        if strict:
            if d > DATA_CUTOFF:
                dropped["future"] += 1
                continue
            if o + OBSERVABLE_MARGIN > cutoff_period:
                dropped["unobservable"] += 1
                continue
        truth[uid] = o
    if report:
        print(f"load_truth: kept {len(truth)}; dropped " +
              ", ".join(f"{k}={v}" for k, v in dropped.items() if v))
    return truth


# ── shared statistics ─────────────────────────────────────────────────────────

def robust_baseline(vals: list[float]) -> tuple[float, float]:
    """Median and MAD-scaled sigma, matching the v3 edge function."""
    a = [v for v in vals if v is not None and math.isfinite(v)]
    if len(a) < 3:
        return float("nan"), float("nan")
    s = sorted(a)
    median = s[len(s) // 2]
    devs = sorted(abs(v - median) for v in a)
    return median, 1.4826 * devs[len(devs) // 2]


def zscore(x: float, mean: float, sd: float) -> float:
    return (x - mean) / sd if sd and math.isfinite(sd) and sd > 1e-9 else float("nan")


def level(series: dict[int, dict], o: int, idx: str):
    v = series.get(o)
    return None if v is None else v.get(idx)


def delta(series: dict[int, dict], o: int, idx: str):
    cur, prv = level(series, o, idx), level(series, o - 1, idx)
    return None if cur is None or prv is None else cur - prv


# ── detectors ─────────────────────────────────────────────────────────────────
# Each returns (fired: bool, computable: bool). `computable` separates "the rule
# said no" from "this campus had no data to run the rule on" -- collapsing those
# two into a single False is what makes a detector look good on the sites it
# silently skipped.

def det_v1(series, o, _ctx):
    nd, sw = delta(series, o, "ndbi"), delta(series, o, "swir")
    if nd is None or sw is None:
        return False, False
    return (nd >= 0.04 or (nd >= 0.02 and sw >= 0.03)), True


def det_v3(series, o, _ctx):
    nd, sw = delta(series, o, "ndbi"), delta(series, o, "swir")
    if nd is None or sw is None:
        return False, False
    legacy = nd >= 0.04 or (nd >= 0.02 and sw >= 0.03)

    trail = [level(series, o - k, "ndbi") for k in range(1, 7)]
    median, sigma = robust_baseline([v for v in trail if v is not None])
    z = (level(series, o, "ndbi") - median) / sigma if (
        math.isfinite(sigma) and sigma > 1e-6) else float("nan")
    robust = math.isfinite(z) and z >= 2.5 and nd > 0

    yoy_prev = level(series, o - 24, "ndbi")
    yoy = None if yoy_prev is None else level(series, o, "ndbi") - yoy_prev
    guard = yoy is None or yoy >= 0.02          # null passes -- the v3 asymmetry

    return ((legacy or robust) and guard), True


def det_prod(series, o, ctx):
    """detect-construction-start: weighted z composite + greenfield split.

    BSI is absent from the cache (the lab's evalscript emits three outputs, not
    four), so its 0.25 weight is redistributed across NDVI and NDBI rather than
    scored as zero -- otherwise the 2.0 threshold would be unreachable for a
    reason that has nothing to do with the rule.
    """
    base = [(level(series, o - k, "ndvi"), level(series, o - k, "ndbi"))
            for k in range(1, 13)]
    nv = [b[0] for b in base if b[0] is not None]
    nb = [b[1] for b in base if b[1] is not None]
    if len(nv) < 3 or len(nb) < 3:
        return False, False
    cur_nv, cur_nb = level(series, o, "ndvi"), level(series, o, "ndbi")
    if cur_nv is None or cur_nb is None:
        return False, False

    zn = zscore(cur_nv, st.mean(nv), st.pstdev(nv))
    zb = zscore(cur_nb, st.mean(nb), st.pstdev(nb))
    if not (math.isfinite(zn) and math.isfinite(zb)):
        return False, False

    score = (0.40 * -zn + 0.35 * zb) / 0.75
    green = max(nv) >= 0.4
    ind = (zn < -2.5 and zb > 2.5) if green else (zb > 2.5)
    fired = score > 2.0 or ind

    # Persistence: the deployed rule needs 3 flags within 30 days of each other.
    ctx.setdefault("flags", [])
    if fired:
        ctx["flags"].append(o)
    recent = [f for f in ctx["flags"] if o - f <= 4]
    return (len(recent) >= 3), True


def det_v4(series, o, _ctx):
    """The proposal. Changes from v3, each traceable to a measured defect:

    * NDVI drop joins the decision. v1/v3 measure it and never read it, yet on
      a greenfield site vegetation loss PRECEDES the concrete NDBI sees.
    * The trailing window skips the period being differenced against, so an
      onset that began last fortnight does not lift its own baseline and
      suppress its own z.
    * sigma gets a floor. v3's `sigma > 1e-6` lets a stable parcel's MAD of
      ~0.002 turn a 0.008 NDBI move into z = 4, well under v1's own 0.02 floor.
    * The YoY guard only ever VETOES a marginal flag, and a missing YoY window
      no longer silently grants a pass -- under v3 a cloudy year-ago scene made
      a site easier to flag than a clear one.
    """
    nd, sw = delta(series, o, "ndbi"), delta(series, o, "swir")
    nv_d = delta(series, o, "ndvi")
    if nd is None or sw is None:
        return False, False

    legacy = nd >= 0.04 or (nd >= 0.02 and sw >= 0.03)

    # Gap of one period: trailing window is o-7..o-2, never o-1.
    trail = [level(series, o - k, "ndbi") for k in range(2, 8)]
    median, sigma = robust_baseline([v for v in trail if v is not None])
    sigma = max(sigma, 0.005) if math.isfinite(sigma) else float("nan")
    cur = level(series, o, "ndbi")
    z = (cur - median) / sigma if (cur is not None and math.isfinite(sigma)) else float("nan")
    robust = math.isfinite(z) and z >= 2.5 and nd >= 0.01

    # Clearing signal: NDVI falling while NDBI rises is construction onset on
    # green land, and it is the earliest optical evidence available.
    clearing = nv_d is not None and nv_d <= -0.05 and nd >= 0.015

    strong = nd >= 0.04
    yoy_prev = level(series, o - 24, "ndbi")
    yoy = None if yoy_prev is None else cur - yoy_prev
    # Veto only marginal evidence, and only on evidence: a missing YoY window
    # is not permission to fire.
    if not strong and yoy is not None and yoy < 0.0:
        return False, True

    return (legacy or robust or clearing), True


# ── v5: a scored detector rather than a stack of ORs ─────────────────────────
# Built from measured separability, not from argument. Over 179 campuses, the
# paired AUC lift of each candidate feature against the quiet band was:
#
#     ndvi over 3 periods   0.151      ndbi over 3 periods   0.050
#     swir over 3 periods   0.138      ndbi 1-period delta   0.023
#     ndbi z vs trailing    0.114      swir 1-period delta   0.072
#     ndvi 1-period delta   0.092
#
# Two things follow, and both contradict how v1/v3/v4 are built:
#
#  1. EVERY existing detector differences ONE fortnight. A three-period (six
#     week) difference separates onset from quiet about twice as well on both
#     indices that carry signal. Construction onset is a ramp, not a step, and
#     a 1-period delta spends most of its variance on cloud and view angle.
#
#  2. NDBI -- the index v1 and v3 are built around -- is the WEAKEST of the
#     three as a raw delta (0.023). It only becomes useful normalised by its
#     own trailing MAD (0.114). NDVI, which v1/v3 fetch and never read, is the
#     strongest single feature available.
#
# Percentiles were tested too (p75/p90/spread, on the theory that construction
# starts in one corner of a large parcel and moves the tail before the mean).
# They did not beat the mean on any index and are deliberately not used.
#
# Cloudy periods are treated as NOT MEASURABLE rather than as "no change",
# which is the same distinction `computable` draws everywhere else in this file.

VF_MIN = 0.60            # 8.7% of periods; below this the mean is cloud, not ground
V5_SCALE = (0.0683, 0.0384, 1.0867)   # pooled quiet-band MAD sigma per feature
V5_WEIGHT = (-1.0, 1.0, 0.35)         # ndvi down, swir up, ndbi z as support
V5_THRESHOLD = 2.2


def v5_features(series, o):
    """(ndvi_d3, swir_d3, ndbi_z) or None if this period is not measurable."""
    cur = series.get(o)
    if cur is None:
        return None
    vf = cur.get("valid_frac")
    if vf is not None and vf < VF_MIN:
        return None
    nv, nv3 = level(series, o, "ndvi"), level(series, o - 3, "ndvi")
    sw, sw3 = level(series, o, "swir"), level(series, o - 3, "swir")
    if nv is None or nv3 is None or sw is None or sw3 is None:
        return None
    # Trailing window skips o-1 so an onset already under way cannot lift its
    # own baseline and suppress its own z -- the v4 fix, kept.
    trail = [level(series, o - k, "ndbi") for k in range(2, 8)]
    median, sigma = robust_baseline([v for v in trail if v is not None])
    nb = level(series, o, "ndbi")
    z = ((nb - median) / max(sigma, 0.005)
         if nb is not None and math.isfinite(sigma) else 0.0)
    return (nv - nv3, sw - sw3, z)


def v5_score(series, o):
    f = v5_features(series, o)
    if f is None:
        return None
    return sum(V5_WEIGHT[i] * f[i] / V5_SCALE[i] for i in range(3))


def make_v5(threshold=V5_THRESHOLD):
    """A v5 at a given threshold. The knob is the point: v1/v3/v4 are single
    operating points with no way to trade hit rate against false alarms, so
    'which is better' is unanswerable for them without changing their code."""
    def det_v5(series, o, _ctx):
        sc = v5_score(series, o)
        if sc is None:
            return False, False
        return sc >= threshold, True
    return det_v5


DETECTORS = {"v1": det_v1, "v3": det_v3, "prod": det_prod, "v4": det_v4,
             "v5": make_v5()}


# ── scoring ───────────────────────────────────────────────────────────────────

def first_flag(fn, series: dict[int, dict], target: int,
               lo: int = -8, hi: int = 8):
    """First period in [target+lo, target+hi] where `fn` fires.

    Returns (offset or None, n_computable). Sweeping a window centred on the
    VERIFIED date is the whole reason this is worth rerunning: the original
    backtest centred its window on the recorded date instead.
    """
    ctx: dict = {}
    hit, n_ok = None, 0
    for o in range(target + lo, target + hi + 1):
        fired, ok = fn(series, o, ctx)
        n_ok += ok
        if fired and hit is None:
            hit = o - target
    return hit, n_ok


def main() -> None:
    series = load_series()
    truth = load_truth()
    print(f"campuses with a verified date: {len(truth)}")

    rows: dict[str, list] = {k: [] for k in DETECTORS}
    for uid, target in truth.items():
        s = series.get(uid)
        if not s:
            continue
        for name, fn in DETECTORS.items():
            off, n_ok = first_flag(fn, s, target)
            if n_ok >= 3:                     # rule was genuinely exercised
                rows[name].append((uid, off))

    print(f"\n{'detector':<8} {'n':>4} {'fired':>6} {'|err|<=1':>9} "
          f"{'median err':>11} {'early':>7} {'late':>7}")
    print("-" * 60)
    for name in DETECTORS:
        r = rows[name]
        fired = [o for _, o in r if o is not None]
        if not r:
            continue
        within = sum(1 for o in fired if abs(o) <= 1)
        med = st.median(fired) if fired else float("nan")
        early = sum(1 for o in fired if o < 0)
        late = sum(1 for o in fired if o > 0)
        print(f"{name:<8} {len(r):>4} {len(fired):>6} "
              f"{within:>4} ({within / len(r) * 100:4.1f}%) "
              f"{med:>10.1f} {early:>7} {late:>7}")

    print("\nmedian err is in half-months; 0 = the detector fired in the exact "
          "fortnight you verified.")

    with open(os.path.join(RESULTS, "replay_offsets.json"), "w") as fh:
        json.dump({k: {u: o for u, o in v} for k, v in rows.items()}, fh)
    print(f"wrote {os.path.join(RESULTS, 'replay_offsets.json')}")


if __name__ == "__main__":
    main()
