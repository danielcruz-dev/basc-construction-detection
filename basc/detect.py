"""
Causal onset detection: what could have been known at time t, and only that.

The previous persistence measure looked FORWARD -- it asked how many of the
next observations still showed the change. That is fine for a retrospective
map and disqualifying for a review queue, because it cannot be computed at the
moment a reviewer needs it. It also returns None for the most recent period,
which is precisely the period anyone cares about.

Here every quantity at time t is computed from observations at or before t.
Persistence is backward-looking: of the clear observations already in hand,
how many consecutively showed the change.

THREE STATES, and the distinction is about evidence, not about certainty:

    insufficient_evidence   too few clear looks to say anything. Distinct from
                            "no change" -- a snowed-in site is not a quiet site.
    provisional_start       change is present now, but has not yet been seen
                            often enough to rule out a transient (harvest,
                            flood, a missed cloud).
    confirmed_start         the change has held across `confirm_n` consecutive
                            clear observations, all of them already past.

A provisional call is a real output, not a placeholder. It is what the queue
should act on; confirmation arrives later and upgrades the record in place,
which is the normal shape of monitoring rather than a defect.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

INSUFFICIENT = "insufficient_evidence"
PROVISIONAL = "provisional_start"
CONFIRMED = "confirmed_start"
NO_CHANGE = "no_change"

DEFAULTS = dict(
    confirm_n=2,            # consecutive clear looks that must show the change
    min_clear_history=3,    # clear looks needed before any call at all
    allow_gap=1,            # clear looks that may miss and still count as held
)


@dataclass
class Detection:
    period: str
    state: str
    changed_now: bool
    n_clear_seen: int = 0
    run_len: int = 0
    provisional_period: str | None = None
    confirmed_period: str | None = None
    evidence: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def run_causal(observations, cfg=None) -> list[Detection]:
    """Walk a site's clear observations forward, emitting one call per period.

    `observations` is an ordered list of dicts with at least:
        period    label
        clear     bool -- was this period observed at all
        changed   bool -- did the evidence bar pass, at this period alone

    Only observations up to and including index i inform the call at index i.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    out: list[Detection] = []
    n_clear = 0
    run = 0
    misses = 0
    provisional = None
    confirmed = None

    for ob in observations:
        period = ob["period"]
        if not ob.get("clear"):
            # An unobserved period cannot change the evidence. The run is held,
            # not reset: cloud is not counter-evidence.
            d = Detection(period=period, state=INSUFFICIENT, changed_now=False,
                          n_clear_seen=n_clear, run_len=run,
                          provisional_period=provisional,
                          confirmed_period=confirmed)
            d.notes.append("no clear observation this period")
            out.append(d)
            continue

        n_clear += 1
        changed = bool(ob.get("changed"))
        if changed:
            run += 1
            misses = 0
        else:
            misses += 1
            if misses > cfg["allow_gap"]:
                run = 0
                # a broken run invalidates an unconfirmed provisional call
                if confirmed is None:
                    provisional = None

        if n_clear < cfg["min_clear_history"]:
            state = INSUFFICIENT
        elif run == 0:
            state = NO_CHANGE
        else:
            if provisional is None:
                provisional = period
            if run >= cfg["confirm_n"]:
                if confirmed is None:
                    confirmed = provisional      # onset dated to FIRST sighting
                state = CONFIRMED
            else:
                state = PROVISIONAL

        d = Detection(period=period, state=state, changed_now=changed,
                      n_clear_seen=n_clear, run_len=run,
                      provisional_period=provisional,
                      confirmed_period=confirmed,
                      evidence=dict(ob.get("evidence") or {}))
        if state == PROVISIONAL:
            d.notes.append(f"changed at {run}/{cfg['confirm_n']} consecutive clear "
                           f"looks — actionable now, confirmable later")
        if state == INSUFFICIENT and n_clear < cfg["min_clear_history"]:
            d.notes.append(f"only {n_clear} clear look(s); need "
                           f"{cfg['min_clear_history']}")
        out.append(d)
    return out


def causal_persistence(observations, i, cfg=None) -> float | None:
    """Backward persistence at index i: share of the last `confirm_n+1` clear
    looks (all at or before i) that showed the change. None if too few."""
    cfg = {**DEFAULTS, **(cfg or {})}
    win = cfg["confirm_n"] + 1
    seen = [ob for ob in observations[:i + 1] if ob.get("clear")]
    if len(seen) < 2:
        return None
    tail = seen[-win:]
    return round(sum(1 for ob in tail if ob.get("changed")) / len(tail), 3)


def latest_call(dets: list[Detection]) -> Detection | None:
    """The call for the most recent period -- never None just because the
    future has not happened yet."""
    return dets[-1] if dets else None
