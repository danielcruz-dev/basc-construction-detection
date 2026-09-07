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

# ---------------------------------------------------------------------------
# Validation thresholds. These are POLICY, not physics -- they are the points
# at which we would rather fall back than trust the geometry. None is derived
# from a validated study; they are starting values, deliberately visible.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    siteplan_buffer_m=150.0,     # clearing/grading/roads reach well past the pad
    context_buffer_m=400.0,      # discovery ring around the development zone
    point_sizes_m=(400.0, 1000.0),
    max_residual_m=25.0,         # georeferencing fit; beyond this the plan is
                                 # not reliably on the ground at 10 m pixels
    max_parcel_ha=800.0,         # above this a parcel mean is meaningless
    warn_parcel_ha=150.0,
    max_point_offset_m=2000.0,   # point should sit in/near its parcel
    min_footprints=1,
)

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
    """Best usable plan layer set for this campus at this date, or None + reason.

    Rejection is per SHEET, not per campus: a campus may hold a valid 2024
    masterplan and a 2025 revision, and scoring mid-2024 must use only the
    first. This is the temporal-leakage guard and it is the whole reason
    resolution takes an observation date.
    """
    mine = [l for l in plan_layers if l.get("campus_uid") == campus_uid]
    if not mine:
        return None, "no site plan for campus"

    usable, rejected = [], []
    for l in mine:
        d = _parse_date(l.get("sheet_date"))
        if d is None:
            rejected.append((l.get("key"), "no sheet date"))
            continue
        if observation_date is not None and d > observation_date:
            rejected.append((l.get("key"), f"sheet {d} postdates observation"))
            continue
        fit = l.get("fit") or {}
        res = fit.get("residual_m")
        if res is not None and res > cfg["max_residual_m"]:
            rejected.append((l.get("key"), f"residual {res:.1f} m"))
            continue
        rings = [f["ring"] for f in (l.get("features") or [])
                 if f.get("ring") and len(f["ring"]) >= 4]
        if len(rings) < cfg["min_footprints"]:
            rejected.append((l.get("key"), "no building rings"))
            continue
        usable.append((l, rings, d))

    for key, why in rejected:
        warnings.append(f"siteplan sheet {key} rejected: {why}")
    if not usable:
        return None, (rejected[0][1] if rejected else "no usable sheet")
    return usable, None


def _from_siteplan(site, plan_layers, observation_date, cfg, warnings):
    uid = getattr(site, "unit_uid", None)
    usable, why = _siteplan_candidate(plan_layers, uid, observation_date, cfg, warnings)
    if usable is None:
        return None, why

    from shapely.geometry import Polygon
    polys, keys, dates = [], [], []
    for l, rings, d in usable:
        polys += [Polygon(r) for r in rings]
        keys.append(l.get("key"))
        dates.append(d)

    building = safe_union(polys)
    if building is None or building.is_empty:
        return None, "footprint union empty"

    lat = getattr(site, "lat", None) or building.centroid.y
    development = buffer_m(building, cfg["siteplan_buffer_m"], lat)
    context = buffer_m(development, cfg["context_buffer_m"], lat)

    n_fp = len(polys)
    if geodesic_area_m2(building) <= 0:
        return None, "footprint area zero"

    # sanity: does the plan sit anywhere near the recorded point?
    off = None
    if getattr(site, "lat", None) is not None:
        off = geodesic_distance_m(site.lon, site.lat,
                                  building.centroid.x, building.centroid.y)
        if off > cfg["max_point_offset_m"]:
            warnings.append(
                f"site plan centroid {off:.0f} m from the recorded point "
                f"(> {cfg['max_point_offset_m']:.0f} m)")

    return dict(
        geometry=context,
        source="siteplan",
        analysis_zones={"building": building, "development": development,
                        "context": context},
        buffer_m=cfg["siteplan_buffer_m"],
        siteplan_date=max(dates).isoformat(),
        siteplan_keys=keys,
        n_footprints=n_fp,
        point_offset_m=off,
    ), None


# ---------------------------------------------------------------------------
# 2. parcel
# ---------------------------------------------------------------------------

def _from_parcel(site, cfg, warnings):
    """The parcel is kept as a SEARCH BOUNDARY, never collapsed to one mean.

    Downstream work computes pixel-level change, connected components and
    tiles INSIDE it -- that is the point of returning it whole. A large parcel
    is flagged, not rejected, because rejecting it would throw away the only
    boundary we have.
    """
    src = getattr(site, "aoi_source", "") or ""
    if not src.startswith("parcel"):
        return None, f"no parcel geometry (aoi_source={src or 'none'})"
    geom = as_geom(getattr(site, "geometry", None))
    if geom is None or geom.is_empty:
        return None, "parcel geometry empty"
    if not geom.is_valid:
        geom = geom.buffer(0)
        warnings.append("parcel geometry was invalid; repaired with buffer(0)")
        if geom.is_empty:
            return None, "parcel geometry invalid and unrepairable"

    area = geodesic_area_m2(geom)
    if area <= 0:
        return None, "parcel area zero"
    ha = area / 1e4

    if ha > cfg["max_parcel_ha"]:
        return None, f"parcel {ha:.0f} ha exceeds max {cfg['max_parcel_ha']:.0f} ha"
    if ha > cfg["warn_parcel_ha"]:
        warnings.append(f"parcel is large ({ha:.0f} ha): an 8,000 m2 event is "
                        f"{8000/area*100:.2f}% of it — rely on the pixel-level "
                        f"change map, not the parcel mean")

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
            warnings.append(f"project point falls OUTSIDE the parcel "
                            f"({off:.0f} m from its centroid)")
            if off > cfg["max_point_offset_m"]:
                return None, f"point {off:.0f} m outside parcel"

    context = buffer_m(geom, cfg["context_buffer_m"], lat)
    return dict(
        geometry=context,
        source="parcel",
        analysis_zones={"building": None, "development": geom, "context": context},
        buffer_m=0.0,
        parcel_id=getattr(site, "unit_uid", None),
        n_parcel_parts=n_parts,
        point_offset_m=off,
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
    for k in ("siteplan_keys", "n_footprints", "n_parcel_parts",
              "point_offset_m", "point_sizes_m"):
        if k in out:
            rec[k] = out[k]
    rec["confidence"] = _confidence(rec)
    return rec


def _confidence(rec) -> float:
    """How much the GEOMETRY can be trusted -- not how likely construction is.

    Ordered by how directly the AOI localises the event: a drawn footprint beats
    a legal boundary beats a dot on a map. Penalties are for the things measured
    in this repo as actually going wrong.
    """
    base = {"siteplan": 0.9, "parcel": 0.65, "point_box": 0.4}[rec["source"]]
    if rec["coordinate_quality"] == "approximate":
        base -= 0.05
    elif rec["coordinate_quality"] == "inferred":
        base -= 0.15
    for w in rec["warnings"]:
        if "OUTSIDE the parcel" in w or "from the recorded point" in w:
            base -= 0.15
        elif "fragmented" in w:
            base -= 0.05
        elif "is large" in w:
            base -= 0.10
    return round(max(0.05, min(1.0, base)), 3)


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
        "coordinate_quality": rec["coordinate_quality"],
        "siteplan_date": rec.get("siteplan_date"),
        "parcel_id": rec.get("parcel_id"),
        "aoi_fallback_reason": rec.get("fallback_reason"),
        "aoi_warnings": list(rec.get("warnings") or []),
    }
