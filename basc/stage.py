"""
Construction-stage model, 0-15%.

    0%   no construction
    5%   site clearing
   10%   grading and levelling
   15%   foundation work

Only the first four stages are modelled. Everything past foundations (steel,
envelope, fit-out, energisation) is deliberately absent: those are vertical
changes that a 10 m nadir optical sensor reads poorly, and there is no labelled
data here to fit them.

WHAT THE STAGES LOOK LIKE SPECTRALLY

  clearing    vegetation removed. Vegetation fraction falls against the site's
              own seasonal norm; soil rises somewhat; footprint is often ragged
              because clearing follows terrain and property lines.
  grading     soil now dominates and the shape TIGHTENS -- a graded pad is
              rectangular in a way a harvested field is not. Area typically
              grows and stabilises.
  foundation  bright material appears (concrete, aggregate, formwork) inside a
              graded area: high albedo rises over a smaller, blockier area.

The ordering is a ratchet in evidence, not in time: a site can be observed at
grading without ever having been observed clearing, because of cloud. So the
rules test for the HIGHEST stage whose evidence is present, rather than
requiring the sequence to have been seen.

STATUS: every threshold below is an unvalidated starting value. They were
chosen to be legible and to sit above the 8,000 m2 minimum mapping unit, NOT
fitted to labelled stage data -- none exists in this repo. Nothing here has
been tested against positive and negative campuses in chronological replay, so
no accuracy claim attaches to it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

MMU_M2 = 8000.0          # the paper's "heavy construction" floor
FOUNDATION_MMU_M2 = 2000.0   # concrete shows before the whole pad is poured

DEFAULTS = dict(
    mmu_m2=MMU_M2,
    foundation_mmu_m2=FOUNDATION_MMU_M2,
    min_persistence=0.5,        # must survive half the following clear looks
    grading_rectangularity=0.55,
    grading_soil_ratio=0.6,     # soil should dominate the changed area
    max_dist_to_building_m=300.0,
)

STAGES = {0: "no construction", 5: "site clearing",
          10: "grading and levelling", 15: "foundation work"}


@dataclass
class StageCall:
    stage_pct: int
    label: str
    confidence: float
    evidence: dict
    reasons: list

    def as_dict(self):
        return asdict(self)


def classify(zf, cfg=None, aoi_source: str | None = None) -> StageCall:
    """Assign a stage from one zone's features.

    `zf` is a change.ZoneFeatures (or its dict). Highest satisfied stage wins.
    """
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
    dbld = f.get("dist_to_building_m")

    ev = {"changed_m2": round(changed, 1), "veg_loss_m2": round(veg, 1),
          "new_soil_m2": round(soil, 1), "new_high_albedo_m2": round(high, 1),
          "largest_component_m2": round(big, 1), "rectangularity": rect,
          "persistence": pers, "dist_to_building_m": dbld}

    # --- gate: is anything happening at all, at a size worth reviewing? -----
    if big < cfg["mmu_m2"] and veg < cfg["mmu_m2"]:
        reasons.append(
            f"largest component {big:,.0f} m2 and vegetation loss {veg:,.0f} m2 "
            f"are both below the {cfg['mmu_m2']:,.0f} m2 minimum")
        return StageCall(0, STAGES[0], 0.6, ev, reasons)

    # persistence is what separates construction from a harvest or a wet field
    if pers is not None and pers < cfg["min_persistence"]:
        reasons.append(f"change did not persist (persistence {pers:.2f} < "
                       f"{cfg['min_persistence']:.2f}) — transient, likely "
                       f"agricultural or moisture")
        return StageCall(0, STAGES[0], 0.55, ev, reasons)

    # --- 15%: bright material inside a graded area --------------------------
    near = (dbld is None) or (dbld <= cfg["max_dist_to_building_m"])
    if high >= cfg["foundation_mmu_m2"] and soil >= cfg["mmu_m2"] and near:
        reasons.append(f"high-albedo {high:,.0f} m2 over soil {soil:,.0f} m2"
                       + (f", {dbld:.0f} m from a planned footprint" if dbld is not None else ""))
        return StageCall(15, STAGES[15], _conf(0.65, pers, rect, aoi_source), ev, reasons)

    # --- 10%: soil dominates and the shape has tightened --------------------
    soil_ratio = soil / changed if changed > 0 else 0.0
    if (soil >= cfg["mmu_m2"] and soil_ratio >= cfg["grading_soil_ratio"]
            and (rect is None or rect >= cfg["grading_rectangularity"])):
        reasons.append(f"soil {soil:,.0f} m2 is {soil_ratio*100:.0f}% of the "
                       f"changed area" + (f", rectangularity {rect:.2f}" if rect else ""))
        return StageCall(10, STAGES[10], _conf(0.6, pers, rect, aoi_source), ev, reasons)

    # --- 5%: vegetation gone, ground not yet worked into a pad --------------
    if veg >= cfg["mmu_m2"] or soil >= cfg["mmu_m2"]:
        reasons.append(f"vegetation loss {veg:,.0f} m2, soil {soil:,.0f} m2, "
                       f"no dominant graded pad yet")
        return StageCall(5, STAGES[5], _conf(0.5, pers, rect, aoi_source), ev, reasons)

    reasons.append("change present but below every stage's evidence bar")
    return StageCall(0, STAGES[0], 0.5, ev, reasons)


def _conf(base, pers, rect, aoi_source):
    c = base
    if pers is not None:
        c += 0.15 * (pers - 0.5) * 2          # +-0.15 across the range
    if rect is not None:
        c += 0.10 * (rect - 0.5) * 2
    # a call inside a drawn footprint is better localised than one in a 100 ha box
    c += {"siteplan": 0.10, "parcel": 0.0, "point_box": -0.10}.get(aoi_source, 0.0)
    return round(max(0.05, min(0.95, c)), 3)
