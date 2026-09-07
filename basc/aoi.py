"""
Unified AOI resolver.

One function decides what patch of ground a site is measured over, in a fixed
priority order, and records WHY. Everything downstream carries that provenance,
because the AOI choice dominates every number the model produces: on a 320 ha
parcel an 8,000 m2 event is 0.25% of the mean and invisible, while on a 16 ha
box it is 5% and obvious.

    1. site plan   -- union of PLANNED building footprints, if a georeferenced
                      plan existed at the observation date and validates
    2. parcel      -- the validated Regrid parcel at the confirmed point
    3. point box   -- ground squares around the best available coordinate

Each level falls back to the next with a recorded reason. Resolution is
per-OBSERVATION-DATE, not per-site: a plan filed in March 2025 is available to
score June 2025 and must not be used to score June 2024.

THREE ANALYSIS ZONES are always returned, so features can be computed at
different scales from one fetch:

    building     the planned footprints themselves (site plan only)
    development  footprints + buffer: clearing, grading, foundations, roads
    context      the wider search area, for discovery and for context features

`geometry` is the union envelope -- the extent that must be FETCHED so every
zone can be computed from a single raster.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

from shapely.geometry import Point, mapping

from geo_utils import (as_geom, buffer_m, geodesic_area_m2, geodesic_distance_m,
                       ground_square, safe_union)
from siteplan_meta import load_docs

# ---------------------------------------------------------------------------
# Validation thresholds. These are POLICY, not physics -- they are the points
# at which we would rather fall back than trust the geometry. None is derived
# from a validated study; they are starting values, deliberately visible.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    siteplan_buffer_m=150.0,     # clearing/grading/roads reach well past the pad
    context_buffer_m=400.0,      # discovery ring around the development zone
    point_sizes_m=(400.0, 1000.0),
    max_residual_m=60.0,         # HARD: beyond ~6 pixels the plan is not on the
                                 # ground at all. Not a quality preference.
    warn_residual_m=15.0,
    # Parcel size is a QUALITY signal, never a rejection. A 900 ha parcel is
    # still the only search boundary that site has; it is tiled, not discarded.
    warn_parcel_ha=150.0,
    tile_parcel_ha=100.0,        # above this, search in tiles rather than whole
    tile_px=64,
    tile_overlap_px=16,
    warn_point_offset_m=1000.0,
    max_point_offset_m=5000.0,   # HARD: beyond this the point and polygon are
                                 # not the same place and one of them is wrong
    min_footprints=1,
    policy="strict",             # strict | relaxed (relaxed is a SENSITIVITY
                                 # analysis and is reported separately)
)

# Only these three justify refusing a level outright. Everything else degrades
# the quality score and raises a warning.
HARD_REJECTIONS = ("invalid geometry", "unusable georeferencing", "temporal leakage")

COORD_QUALITY = ("confirmed", "approximate", "inferred")


def _q(site) -> str:
    if getattr(site, "coord_confirmed", False):
        return "confirmed"
    if getattr(site, "lat", None) is not None and getattr(site, "lon", None) is not None:
        return "approximate"
    return "inferred"


def _parse_date(v) -> dt.date | None:
    if not v:
        return None
    if isinstance(v, dt.date):
        return v
    try:
        return dt.date.fromisoformat(str(v).strip()[:10])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. site plan
# ---------------------------------------------------------------------------

def _siteplan_candidate(plan_layers, campus_uid, observation_date, cfg, warnings):
    """Usable plan documents for this campus at this date, or None + reason.

    Rejection is per DOCUMENT and only ever for the three hard reasons:
    temporal leakage (the sheet did not exist yet, or is an as-built), unusable
    georeferencing, or no geometry. A merely mediocre fit is kept and warned
    about -- discarding it would silently demote the site to a parcel mean,
    which is worse than a plan that is 20 m off.
    """
    docs = [d for d in load_docs(plan_layers) if d.campus_uid == campus_uid]
    if not docs:
        return None, "no site plan for campus"

    policy = cfg.get("policy", "strict")
    usable, rejected = [], []
    for d in docs:
        ok, why = d.availability(observation_date, policy)
        if not ok:
            rejected.append((d.key, why, "temporal leakage"))
            continue
        if d.residual_m is not None and d.residual_m > cfg["max_residual_m"]:
            rejected.append((d.key, f"residual {d.residual_m:.1f} m",
                             "unusable georeferencing"))
            continue
        if d.residual_m is not None and d.residual_m > cfg["warn_residual_m"]:
            warnings.append(f"siteplan {d.key}: georeferencing residual "
                            f"{d.residual_m:.1f} m (> {cfg['warn_residual_m']:.0f} m)")
        if d.date_provenance:
            warnings.append(f"siteplan {d.key}: {d.date_provenance}")
        usable.append(d)

    for key, why, kind in rejected:
        warnings.append(f"siteplan {key} rejected ({kind}): {why}")
    if not usable:
        return None, (rejected[0][1] if rejected else "no usable sheet")
    return usable, None


def _from_siteplan(site, plan_layers, observation_date, cfg, warnings):
    uid = getattr(site, "unit_uid", None)
    usable, why = _siteplan_candidate(plan_layers, uid, observation_date, cfg, warnings)
    if usable is None:
        return None, why

    from shapely.geometry import Polygon
    polys, keys, gates, docs = [], [], [], []
    for d in usable:
        polys += [Polygon(r) for r in d.rings]
        keys.append(d.key)
        if d.usable_from:
            gates.append(d.usable_from)
        docs.append(d.as_dict())

    building = safe_union(polys)
    if building is None or building.is_empty:
        return None, "footprint union empty"          # invalid geometry
    if geodesic_area_m2(building) <= 0:
        return None, "footprint area zero"

    lat = getattr(site, "lat", None) or building.centroid.y
    development = buffer_m(building, cfg["siteplan_buffer_m"], lat)
    context = buffer_m(development, cfg["context_buffer_m"], lat)

    off = None
    if getattr(site, "lat", None) is not None:
        off = geodesic_distance_m(site.lon, site.lat,
                                  building.centroid.x, building.centroid.y)
        if off > cfg["max_point_offset_m"]:
            return None, (f"site plan centroid {off:.0f} m from the point "
                          f"— not the same place")
        if off > cfg["warn_point_offset_m"]:
            warnings.append(f"site plan centroid {off:.0f} m from the recorded point")

    return dict(
        geometry=context,
        source="siteplan",
        analysis_zones={"building": building, "development": development,
                        "context": context},
        buffer_m=cfg["siteplan_buffer_m"],
        siteplan_date=max(gates).isoformat() if gates else None,
        siteplan_keys=keys,
        siteplan_docs=docs,
        siteplan_policy=cfg.get("policy", "strict"),
        n_footprints=len(polys),
        point_offset_m=off,
    ), None


# ---------------------------------------------------------------------------
# 2. parcel
# ---------------------------------------------------------------------------

def _from_parcel(site, cfg, warnings):
    """The parcel is a SEARCH BOUNDARY and is never rejected for being big.

    Size is a quality signal, not a disqualification: a 900 ha parcel is still
    the only boundary that site has, and discarding it demotes the site to a
    box on a coordinate that -- measured in this repo -- misses the actual
    construction entirely on large campuses. Instead the parcel is kept whole
    and flagged for TILED search, so localized components survive rather than
    being averaged away.

    Hard rejection is limited to geometry that is unusable: empty, unrepairable,
    zero-area, or so far from the project point that one of the two is wrong.
    """
    src = getattr(site, "aoi_source", "") or ""
    if not src.startswith("parcel"):
        return None, f"no parcel geometry (aoi_source={src or 'none'})"
    geom = as_geom(getattr(site, "geometry", None))
    if geom is None or geom.is_empty:
        return None, "parcel geometry empty"                    # invalid geometry
    if not geom.is_valid:
        geom = geom.buffer(0)
        warnings.append("parcel geometry was invalid; repaired with buffer(0)")
        if geom.is_empty:
            return None, "parcel geometry invalid and unrepairable"

    area = geodesic_area_m2(geom)
    if area <= 0:
        return None, "parcel area zero"                         # invalid geometry
    ha = area / 1e4

    needs_tiling = ha > cfg["tile_parcel_ha"]
    if ha > cfg["warn_parcel_ha"]:
        warnings.append(
            f"parcel is large ({ha:,.0f} ha): an 8,000 m2 event is "
            f"{8000/area*100:.3f}% of it — use the tiled component search, "
            f"never the parcel mean")

    n_parts = len(geom.geoms) if geom.geom_type == "MultiPolygon" else 1
    if n_parts > 1:
        warnings.append(f"parcel is fragmented ({n_parts} parts)")

    lat = getattr(site, "lat", None) or geom.centroid.y
    off = None
    if getattr(site, "lat", None) is not None:
        pt = Point(site.lon, site.lat)
        if not geom.contains(pt):
            off = geodesic_distance_m(site.lon, site.lat,
                                      geom.centroid.x, geom.centroid.y)
            if off > cfg["max_point_offset_m"]:
                return None, (f"point {off:.0f} m outside the parcel — "
                              f"not the same place")
            warnings.append(f"project point falls OUTSIDE the parcel "
                            f"({off:.0f} m from its centroid)")

    context = buffer_m(geom, cfg["context_buffer_m"], lat)
    return dict(
        geometry=context,
        source="parcel",
        analysis_zones={"building": None, "development": geom, "context": context},
        buffer_m=0.0,
        parcel_id=getattr(site, "unit_uid", None),
        n_parcel_parts=n_parts,
        point_offset_m=off,
        needs_tiling=needs_tiling,
        tile_px=cfg["tile_px"],
        tile_overlap_px=cfg["tile_overlap_px"],
    ), None


# ---------------------------------------------------------------------------
# 3. point box
# ---------------------------------------------------------------------------

def _from_point(site, cfg, warnings):
    lat, lon = getattr(site, "lat", None), getattr(site, "lon", None)
    if lat is None or lon is None:
        return None, "no coordinate"
    sizes = sorted(cfg["point_sizes_m"])
    small, large = sizes[0], sizes[-1]
    dev = ground_square(lat, lon, small)
    ctx = ground_square(lat, lon, large)
    q = _q(site)
    if q != "confirmed":
        warnings.append(f"coordinate is {q}: a {small:.0f} m box may miss the pad")
    warnings.append(
        f"point AOI only — {small:.0f} m box for sensitivity "
        f"(8,000 m2 = {8000/(small*small)*100:.1f}% of it), "
        f"{large:.0f} m for discovery")
    return dict(
        geometry=ctx,
        source="point_box",
        analysis_zones={"building": None, "development": dev, "context": ctx},
        buffer_m=0.0,
        point_sizes_m=[small, large],
    ), None


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def resolve_aoi(site, observation_date=None, plan_layers=None, config=None) -> dict[str, Any]:
    """Resolve one site to an AOI, with provenance.

    observation_date gates the site plan. Pass the date being SCORED, never
    "today": scoring 2024 with a 2025 sheet is temporal leakage, and in
    historical replay it is the difference between a real backtest and one that
    knows the answer.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)
    observation_date = _parse_date(observation_date)

    warnings: list[str] = []
    reasons: list[str] = []
    out = None

    if plan_layers:
        out, why = _from_siteplan(site, plan_layers, observation_date, cfg, warnings)
        if out is None:
            reasons.append(f"siteplan: {why}")
    else:
        reasons.append("siteplan: no plan index supplied")

    if out is None:
        out, why = _from_parcel(site, cfg, warnings)
        if out is None:
            reasons.append(f"parcel: {why}")

    if out is None:
        out, why = _from_point(site, cfg, warnings)
        if out is None:
            reasons.append(f"point: {why}")
            raise ValueError(f"cannot resolve an AOI for "
                             f"{getattr(site,'unit_uid','?')}: {'; '.join(reasons)}")

    zones = out["analysis_zones"]
    rec: dict[str, Any] = {
        "geometry": out["geometry"],
        "source": out["source"],
        "analysis_zones": zones,
        "area_m2": geodesic_area_m2(out["geometry"]),
        "zone_area_m2": {k: (geodesic_area_m2(v) if v is not None else None)
                         for k, v in zones.items()},
        "buffer_m": out.get("buffer_m", 0.0),
        "coordinate_quality": _q(site),
        "siteplan_date": out.get("siteplan_date"),
        "parcel_id": out.get("parcel_id"),
        "confidence": None,
        "warnings": warnings,
        "fallback_reason": "; ".join(reasons) if reasons else None,
    }
    for k in ("siteplan_keys", "siteplan_docs", "siteplan_policy", "n_footprints",
              "n_parcel_parts", "point_offset_m", "point_sizes_m",
              "needs_tiling", "tile_px", "tile_overlap_px"):
        if k in out:
            rec[k] = out[k]
    rec["quality"] = _quality(rec)
    rec["confidence"] = rec["quality"]["score"]
    return rec


def _quality(rec) -> dict:
    """How much the GEOMETRY can be trusted -- not how likely construction is.

    Ordered by how directly the AOI localises the event: a drawn footprint beats
    a legal boundary beats a dot on a map. Penalties are for the things measured
    in this repo as actually going wrong.
    """
    base = {"siteplan": 0.9, "parcel": 0.65, "point_box": 0.4}[rec["source"]]
    deductions = []
    if rec["coordinate_quality"] == "approximate":
        base -= 0.05; deductions.append(("coordinate approximate", -0.05))
    elif rec["coordinate_quality"] == "inferred":
        base -= 0.15; deductions.append(("coordinate inferred", -0.15))
    for w in rec["warnings"]:
        if "OUTSIDE the parcel" in w or "from the recorded point" in w:
            base -= 0.15; deductions.append(("point/polygon disagreement", -0.15))
        elif "fragmented" in w:
            base -= 0.05; deductions.append(("fragmented parcel", -0.05))
        elif "is large" in w:
            # a size penalty, NOT a rejection: the parcel is still searched,
            # just by tiles rather than by its mean
            base -= 0.10; deductions.append(("large parcel — tiled search", -0.10))
        elif "residual" in w:
            base -= 0.10; deductions.append(("weak georeferencing", -0.10))
        elif "optimistic" in w:
            base -= 0.05; deductions.append(("plan availability date optimistic", -0.05))
    return {"score": round(max(0.05, min(1.0, base)), 3),
            "base": {"siteplan": 0.9, "parcel": 0.65, "point_box": 0.4}[rec["source"]],
            "deductions": deductions}


def to_json(rec) -> dict:
    """Serialisable copy: geometries -> GeoJSON. Provenance is preserved."""
    d = dict(rec)
    d["geometry"] = mapping(rec["geometry"])
    d["analysis_zones"] = {k: (mapping(v) if v is not None else None)
                           for k, v in rec["analysis_zones"].items()}
    return d


def load_plan_layers(path) -> list[dict]:
    return json.load(open(path))["layers"]


def provenance(rec) -> dict:
    """The subset that must ride along with every feature and prediction."""
    return {
        "aoi_source": rec["source"],
        "aoi_area_m2": round(rec["area_m2"], 1),
        "aoi_buffer_m": rec["buffer_m"],
        "aoi_confidence": rec["confidence"],
        "aoi_quality": rec.get("quality"),
        "aoi_needs_tiling": rec.get("needs_tiling", False),
        "siteplan_policy": rec.get("siteplan_policy"),
        "coordinate_quality": rec["coordinate_quality"],
        "siteplan_date": rec.get("siteplan_date"),
        "parcel_id": rec.get("parcel_id"),
        "aoi_fallback_reason": rec.get("fallback_reason"),
        "aoi_warnings": list(rec.get("warnings") or []),
    }
