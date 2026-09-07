"""
Construction-stage model, 0-15%, with abstention.

    0%    no construction
    5%    site clearing
   10%    grading and levelling
   15%    foundation work
   "10-15" stage GROUP, when the evidence says "worked ground" but the
          resolution cannot separate a graded pad from poured foundations

Everything past foundations is out of scope: those are vertical changes a 10 m
nadir optical sensor reads poorly, and there is no labelled data here for them.

FOUNDATIONS ARE NOT CALLED FROM BRIGHTNESS. High albedo alone is the weakest
possible evidence for a foundation: Wu & Murray's own high-albedo endmember
absorbs concrete, sand AND cloud, and this repo measured that fraction
anti-correlating with valid-pixel count at -0.37 to -0.42, i.e. behaving partly
as a cloud detector. So 15% requires a CONFIGURABLE COMBINATION of independent
evidence, and falls back to the 10-15 group when it cannot be separated:

    persistence          the change held across consecutive PAST clear looks
    connected area       a single component above the minimum mapping unit
    geometry             rectangularity/compactness consistent with a pad
    footprint proximity  near a planned building, where a plan exists
    prior vegetation loss  the site was cleared before it was bright
    visual evidence      an optional human/model confirmation flag

STATUS: every threshold is an unvalidated starting value, chosen to be legible
and to sit above the 8,000 m2 minimum mapping unit. None is fitted -- there is
no labelled stage data in this repo, and fitting to five sites would produce a
number that describes those five sites. No accuracy claim attaches to any of it
until chronological replay over positive AND negative campuses.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

MMU_M2 = 8000.0
FOUNDATION_MMU_M2 = 2000.0

DEFAULTS = dict(
    mmu_m2=MMU_M2,
    foundation_mmu_m2=FOUNDATION_MMU_M2,
    min_persistence=0.5,
    grading_rectangularity=0.55,
    grading_soil_ratio=0.6,
    max_dist_to_building_m=300.0,
    # --- foundation evidence policy -----------------------------------------
    foundation_min_evidence=4,       # of the six below
    foundation_min_persistence=0.6,
    foundation_min_rectangularity=0.6,
    foundation_require_prior_veg_loss=False,
    # abstain rather than guess when foundation evidence is partial
    abstain_min_evidence=2,
)

STAGES = {0: "no construction", 5: "site clearing",
          10: "grading and levelling", 15: "foundation work"}
GROUP_10_15 = "10-15"


@dataclass
class StageCall:
    stage_pct: int | None
    stage_group: str | None
    label: str
    confidence: float
    evidence: dict
    reasons: list

    def as_dict(self):
        return asdict(self)


def _foundation_evidence(f, cfg):
    """Which independent signals support 'foundation', not just 'bright'."""
    high = f.get("new_high_albedo_area_m2") or 0.0
    soil = f.get("new_soil_area_m2") or 0.0
    big = f.get("largest_component_m2") or 0.0
    rect = f.get("largest_component_rectangularity")
    pers = f.get("persistence")
    dbld = f.get("dist_to_building_m")
    prior_veg = f.get("prior_veg_loss_m2")
    visual = f.get("visual_confirmed")

    ev = {}
    ev["bright_area"] = high >= cfg["foundation_mmu_m2"]
    ev["connected_area"] = big >= cfg["mmu_m2"]
    ev["persistence"] = (pers is not None and pers >= cfg["foundation_min_persistence"])
    ev["geometry"] = (rect is not None and rect >= cfg["foundation_min_rectangularity"])
    ev["footprint_proximity"] = (dbld is not None and dbld <= cfg["max_dist_to_building_m"])
    ev["prior_veg_loss"] = (prior_veg is not None and prior_veg >= cfg["mmu_m2"])
    ev["visual"] = bool(visual)
    ev["worked_ground"] = soil >= cfg["mmu_m2"]
    return ev


def classify(zf, cfg=None, aoi_source: str | None = None) -> StageCall:
    cfg = {**DEFAULTS, **(cfg or {})}
    f = zf if isinstance(zf, dict) else zf.as_dict()
    reasons: list[str] = []

    changed = f.get("total_changed_area_m2") or 0.0
    veg = f.get("veg_loss_area_m2") or 0.0
    soil = f.get("new_soil_area_m2") or 0.0
    high = f.get("new_high_albedo_area_m2") or 0.0
    big = f.get("largest_component_m2") or 0.0
    rect = f.get("largest_component_rectangularity")
    pers = f.get("persistence")

    ev = {"changed_m2": round(changed, 1), "veg_loss_m2": round(veg, 1),
          "new_soil_m2": round(soil, 1), "new_high_albedo_m2": round(high, 1),
          "largest_component_m2": round(big, 1), "rectangularity": rect,
          "persistence": pers,
          "dist_to_building_m": f.get("dist_to_building_m"),
          "prior_veg_loss_m2": f.get("prior_veg_loss_m2"),
          "visual_confirmed": f.get("visual_confirmed")}

    if big < cfg["mmu_m2"] and veg < cfg["mmu_m2"]:
        reasons.append(f"largest component {big:,.0f} m2 and vegetation loss "
                       f"{veg:,.0f} m2 both below the {cfg['mmu_m2']:,.0f} m2 minimum")
        return StageCall(0, None, STAGES[0], 0.6, ev, reasons)

    if pers is not None and pers < cfg["min_persistence"]:
        reasons.append(f"change did not persist ({pers:.2f} < "
                       f"{cfg['min_persistence']:.2f}) — transient, likely "
                       f"agricultural or moisture")
        return StageCall(0, None, STAGES[0], 0.55, ev, reasons)

    # ---- 15%: multi-evidence, never brightness alone ------------------------
    fe = _foundation_evidence(f, cfg)
    ev["foundation_evidence"] = fe
    n_ev = sum(1 for k, v in fe.items()
               if v and k in ("bright_area", "connected_area", "persistence",
                              "geometry", "footprint_proximity",
                              "prior_veg_loss", "visual"))
    ev["foundation_evidence_count"] = n_ev

    if fe["bright_area"] and fe["worked_ground"]:
        need_veg = (not cfg["foundation_require_prior_veg_loss"]) or fe["prior_veg_loss"]
        if n_ev >= cfg["foundation_min_evidence"] and need_veg:
            reasons.append(f"foundation: {n_ev} independent signals "
                           f"({', '.join(k for k, v in fe.items() if v)})")
            return StageCall(15, None, STAGES[15],
                             _conf(0.6, pers, rect, aoi_source, n_ev), ev, reasons)
        if n_ev >= cfg["abstain_min_evidence"]:
            reasons.append(
                f"bright material over worked ground, but only {n_ev} independent "
                f"signals (need {cfg['foundation_min_evidence']}) — cannot separate "
                f"a graded pad from poured foundations at 10 m; abstaining")
            return StageCall(None, GROUP_10_15, "grading or foundation (unresolved)",
                             _conf(0.45, pers, rect, aoi_source, n_ev), ev, reasons)

    # ---- 10%: soil dominates and the shape has tightened --------------------
    soil_ratio = soil / changed if changed > 0 else 0.0
    if (soil >= cfg["mmu_m2"] and soil_ratio >= cfg["grading_soil_ratio"]
            and (rect is None or rect >= cfg["grading_rectangularity"])):
        if fe["bright_area"] and n_ev >= cfg["abstain_min_evidence"]:
            reasons.append("graded pad with some bright material — 10 vs 15 "
                           "unresolved at this resolution")
            return StageCall(None, GROUP_10_15, "grading or foundation (unresolved)",
                             _conf(0.45, pers, rect, aoi_source, n_ev), ev, reasons)
        reasons.append(f"soil {soil:,.0f} m2 is {soil_ratio*100:.0f}% of the changed "
                       f"area" + (f", rectangularity {rect:.2f}" if rect else ""))
        return StageCall(10, None, STAGES[10],
                         _conf(0.6, pers, rect, aoi_source, n_ev), ev, reasons)

    # ---- 5%: vegetation gone, ground not yet worked into a pad --------------
    if veg >= cfg["mmu_m2"] or soil >= cfg["mmu_m2"]:
        reasons.append(f"vegetation loss {veg:,.0f} m2, soil {soil:,.0f} m2, "
                       f"no dominant graded pad yet")
        return StageCall(5, None, STAGES[5],
                         _conf(0.5, pers, rect, aoi_source, n_ev), ev, reasons)

    reasons.append("change present but below every stage's evidence bar")
    return StageCall(0, None, STAGES[0], 0.5, ev, reasons)


def _conf(base, pers, rect, aoi_source, n_ev=0):
    c = base
    if pers is not None:
        c += 0.15 * (pers - 0.5) * 2
    if rect is not None:
        c += 0.10 * (rect - 0.5) * 2
    c += 0.03 * max(0, n_ev - 2)
    c += {"siteplan": 0.10, "parcel": 0.0, "point_box": -0.10}.get(aoi_source, 0.0)
    return round(max(0.05, min(0.95, c)), 3)
