"""
Construction-start ground truth as an INTERVAL, not a date.

A start date is not observed; it is inferred from two observations that bracket
it. The honest object is therefore:

    last_unchanged   the latest clear observation still showing no change
    first_changed    the earliest clear observation showing change
    the start lies somewhere in (last_unchanged, first_changed]

Recording a single date discards the width of that bracket, and the width is
not small: this dataset is 69% clear overall, and in January-February -- which
carry 15.8% and 9.6% of all reported starts -- it is 48% and 35%. A February
start can easily be bracketed by observations two months apart, and scoring a
detector to the fortnight against a date that was never that precise
manufactures both false earliness and false lateness.

The REPORTED date is kept separately and never overwritten. It is evidence
about the start, with its own source and confidence, not the start itself.

Sources, weakest to strongest:
    event_date        the raw inventory field. 52.8% of its values fall on the
                      1st or 15th of a month -- data-entry placeholders that
                      cannot resolve a fortnight.
    reviewer          hand-corrected against imagery. Better, but subject to
                      the visibility bias above: a reviewer can only date a
                      start to when it BECAME VISIBLE.
    imagery_bracket   derived here from clear observations.
    permit / filing   an external documentary date, when one exists.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict, field

SOURCES = ("event_date", "reviewer", "imagery_bracket", "permit", "unknown")


@dataclass
class StartInterval:
    campus_uid: str
    last_unchanged_period: str | None = None
    first_changed_period: str | None = None
    last_unchanged_date: str | None = None
    first_changed_date: str | None = None
    # what somebody SAID the start was -- kept, never merged into the interval
    reported_date: str | None = None
    reported_period: str | None = None
    reported_source: str = "unknown"
    reported_confidence: float | None = None
    # provenance of the interval itself
    interval_source: str = "imagery_bracket"
    n_clear_before: int = 0
    n_clear_after: int = 0
    notes: list = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def is_bounded(self) -> bool:
        return self.last_unchanged_period is not None and \
               self.first_changed_period is not None

    @property
    def width_days(self) -> int | None:
        a = _d(self.last_unchanged_date)
        b = _d(self.first_changed_date)
        return (b - a).days if a and b else None

    def contains_period(self, period_ord: int, ordn) -> bool:
        """Would a detection at this period be consistent with the truth?"""
        if not self.is_bounded:
            return False
        return ordn(self.last_unchanged_period) < period_ord <= \
               ordn(self.first_changed_period)

    def agrees_with_reported(self, ordn, tol_periods: int = 0) -> bool | None:
        """Does the reported date fall inside the imagery bracket?

        None when either side is missing. A False here is informative on its
        own: it means the label and the imagery disagree about that site, which
        is a data-quality finding rather than a detector failure.
        """
        if not self.is_bounded or not self.reported_period:
            return None
        lo = ordn(self.last_unchanged_period) - tol_periods
        hi = ordn(self.first_changed_period) + tol_periods
        return lo < ordn(self.reported_period) <= hi

    def as_dict(self):
        d = asdict(self)
        d["width_days"] = self.width_days
        d["is_bounded"] = self.is_bounded
        return d


def _d(s):
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def bracket_from_detections(campus_uid, dets, period_bounds,
                            reported_date=None, reported_period=None,
                            reported_source="reviewer", reported_confidence=None):
    """Build an interval from a causal detection run.

    last_unchanged is the final clear observation before the first sighting that
    later stuck; first_changed is that sighting. Both come from the detection
    record, so an interval is only ever as precise as the imagery allowed.
    """
    from detect import CONFIRMED, PROVISIONAL

    first_changed = None
    for d in dets:
        if d.state in (PROVISIONAL, CONFIRMED):
            first_changed = d.provisional_period or d.period
            break

    last_unchanged = None
    if first_changed is not None:
        for d in dets:
            if d.period == first_changed:
                break
            if d.changed_now is False and d.state != "insufficient_evidence":
                last_unchanged = d.period

    si = StartInterval(
        campus_uid=campus_uid,
        last_unchanged_period=last_unchanged,
        first_changed_period=first_changed,
        last_unchanged_date=period_bounds(last_unchanged)[0] if last_unchanged else None,
        first_changed_date=period_bounds(first_changed)[1] if first_changed else None,
        reported_date=reported_date,
        reported_period=reported_period,
        reported_source=reported_source,
        reported_confidence=reported_confidence,
        n_clear_before=sum(1 for d in dets
                           if d.changed_now is False and d.state != "insufficient_evidence"),
        n_clear_after=sum(1 for d in dets if d.changed_now),
    )
    if first_changed is None:
        si.notes.append("no change ever detected: interval is open at the right")
    if last_unchanged is None and first_changed is not None:
        si.notes.append("no confirmed-unchanged observation before the first change: "
                        "interval is open at the left, the start may predate the record")
    w = si.width_days
    if w is not None and w > 45:
        si.notes.append(f"bracket is {w} days wide — cloud-limited; do not score "
                        f"this site to the fortnight")
    return si
