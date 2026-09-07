"""
Site-plan document metadata, and when a plan may legitimately be used.

`plan-meta.json` carries a single `sheet_date`, which conflates several
different things that matter differently for leakage:

    document_date       the date drawn on the sheet
    revision_date       a later revision of the same drawing
    public_date         when it entered a public record (filing, agenda packet)
    plan_type           masterplan | phase | as_built | unknown
    usable_from         the date from which a detector could have had it

usable_from is the only one the evaluator consults. It is the LATEST of the
dates we actually know, because a detector cannot use a document before it
exists, and cannot use a filing before it is filed. Where only a document date
is known, that is used and the record says so -- a drawing usually predates its
publication, so this is the optimistic end and is flagged.

TWO POLICIES, and the difference is reported rather than hidden:

  strict   (default, and the primary evaluation) usable_from must be <= the
           observation date. This is the number that gets quoted.
  relaxed  allows a plan whose document_date precedes the observation even if
           it surfaced publicly later, on the argument that these are drawings
           of PROPOSED buildings and the footprints existed as intent before
           they were filed. This is a SENSITIVITY ANALYSIS only. Anything run
           under it is labelled and reported separately; it never replaces the
           strict figure.

An `as_built` plan is never usable under either policy: it records what was
built, so using it to find construction is circular.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict, field

POLICIES = ("strict", "relaxed")
PLAN_TYPES = ("masterplan", "phase", "as_built", "unknown")


def parse_date(v) -> dt.date | None:
    if not v:
        return None
    if isinstance(v, dt.date):
        return v
    try:
        return dt.date.fromisoformat(str(v).strip()[:10])
    except Exception:
        return None


@dataclass
class PlanDoc:
    key: str
    campus_uid: str
    document_date: dt.date | None = None
    revision_date: dt.date | None = None
    public_date: dt.date | None = None
    plan_type: str = "unknown"
    residual_m: float | None = None
    jackknife_p90_m: float | None = None
    n_rings: int = 0
    rings: list = field(default_factory=list)
    date_provenance: str = ""

    # ------------------------------------------------------------------
    @property
    def usable_from(self) -> dt.date | None:
        """Latest known date -- the earliest a detector could have held it."""
        known = [d for d in (self.document_date, self.revision_date,
                             self.public_date) if d is not None]
        return max(known) if known else None

    @property
    def document_from(self) -> dt.date | None:
        """The optimistic date used by the relaxed sensitivity policy."""
        known = [d for d in (self.document_date, self.revision_date) if d is not None]
        return min(known) if known else self.usable_from

    def availability(self, observation_date, policy: str = "strict"):
        """(available: bool, reason: str) under `policy`."""
        if self.plan_type == "as_built":
            return False, "as-built plan: records what was built, circular"
        if self.n_rings < 1:
            return False, "no building rings"
        gate = self.usable_from if policy == "strict" else self.document_from
        if gate is None:
            return False, "no usable date on the sheet"
        if observation_date is not None and gate > observation_date:
            return False, (f"{policy}: usable_from {gate} postdates observation "
                           f"{observation_date}")
        return True, ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("rings", None)
        for k in ("document_date", "revision_date", "public_date"):
            d[k] = d[k].isoformat() if d[k] else None
        d["usable_from"] = self.usable_from.isoformat() if self.usable_from else None
        d["document_from"] = self.document_from.isoformat() if self.document_from else None
        return d


def from_layer(layer: dict) -> PlanDoc:
    """Normalise one plan-meta.json layer into a PlanDoc.

    plan-meta.json currently supplies only `sheet_date`, so document_date is
    populated from it and the absence of a distinct public_date is recorded in
    date_provenance. When the intake pipeline starts carrying filing dates,
    they populate public_date and usable_from tightens automatically -- no
    caller changes.
    """
    fit = layer.get("fit") or {}
    rings = [f["ring"] for f in (layer.get("features") or [])
             if f.get("ring") and len(f["ring"]) >= 4]

    doc = parse_date(layer.get("document_date") or layer.get("sheet_date"))
    rev = parse_date(layer.get("revision_date"))
    pub = parse_date(layer.get("public_date") or layer.get("filing_date"))

    prov = []
    if layer.get("document_date") or layer.get("sheet_date"):
        prov.append("document_date from sheet_date" if not layer.get("document_date")
                    else "document_date explicit")
    if pub is None:
        prov.append("no public/filing date known -- usable_from is optimistic")

    ptype = (layer.get("plan_type") or "").strip().lower()
    if ptype not in PLAN_TYPES:
        name = f"{layer.get('name','')} {layer.get('sheet','')}".lower()
        ptype = ("as_built" if "as-built" in name or "as built" in name
                 else "phase" if "phase" in name
                 else "masterplan" if "master" in name or "overall" in name
                 else "unknown")

    return PlanDoc(
        key=layer.get("key", "?"), campus_uid=layer.get("campus_uid", "?"),
        document_date=doc, revision_date=rev, public_date=pub,
        plan_type=ptype, residual_m=fit.get("residual_m"),
        jackknife_p90_m=fit.get("jackknife_p90_m"),
        n_rings=len(rings), rings=rings, date_provenance="; ".join(prov),
    )


def load_docs(plan_layers) -> list[PlanDoc]:
    return [from_layer(l) for l in plan_layers]
