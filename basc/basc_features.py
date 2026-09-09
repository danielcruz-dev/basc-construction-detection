"""
Per-zone, per-observation features and stage calls.

Ties the pieces together: rasters -> LSMA fractions per pixel -> seasonal
baseline -> change masks -> connected components -> features for each analysis
zone -> a 0-15% stage call. Every record carries the AOI provenance, so a
prediction can always be traced back to which footprint it was measured over
and why that footprint was chosen.

    python basc_features.py --root results/sp16 --series results/series_sp16 \
        --endmembers results/series_sp16/_endmembers.json --out results/features_sp16

Reads the AOI zones stored in each site's grid.json by basc_fetch.py --resolve.
Where those are absent (runs made before the resolver) it falls back to the
whole raster as a single "development" zone and says so in the record.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import change
import detect
import stage as stage_mod
import tiling
from truth import bracket_from_detections
from basc_lsma import EM_NAMES, load_stack, unmix
from geo_utils import as_geom, pixel_area_m2, rasterize
from replay import ordn, unordn as unordn_

FRACTION_INDEX = {n: i for i, n in enumerate(EM_NAMES)}


def zone_masks(g):
    """(name -> bool mask) for the zones stored in grid.json, plus a fallback."""
    bbox, W, H = g["bbox"], g["width"], g["height"]
    aoi = g.get("aoi") or {}
    zones = aoi.get("analysis_zones") or {}
    out = {}
    for name in ("building", "development", "context"):
        geom = as_geom(zones.get(name))
        if geom is not None and not geom.is_empty:
            out[name] = rasterize(geom, bbox, W, H)
    if not out:
        geom = as_geom(g.get("geometry"))
        m = rasterize(geom, bbox, W, H) if geom is not None else np.ones((H, W), bool)
        out["development"] = m
    return out


def fraction_cube(site_dir, g, M):
    """period ordinal -> (H,W,4) fractions, NaN where not clear."""
    H, W = g["height"], g["width"]
    cube = {}
    for p in g["periods"]:
        fn = os.path.join(site_dir, f"{p}.tif")
        if not os.path.exists(fn):
            continue
        try:
            r, m = load_stack(fn)
        except Exception:
            continue
        ok = m & (r > 0).all(-1) & (r < 1.2).all(-1)
        if ok.sum() < 20:
            continue
        F = np.full((H, W, 4), np.nan)
        F[ok] = unmix(r[ok], M)[0]
        cube[ordn(p)] = F
    return cube


def run_site(site_dir, M, cfg, persist_ahead=None):
    """persist_ahead is retained only for backwards compatibility and is
    ignored: persistence is now strictly causal (backward-looking), so a
    prediction at t never consults an observation after t."""
    g = json.load(open(os.path.join(site_dir, "grid.json")))
    bbox, W, H = g["bbox"], g["width"], g["height"]
    px_area = pixel_area_m2(bbox, W, H)
    prov = g.get("aoi_provenance") or {"aoi_source": "unknown"}
    aoi = g.get("aoi") or {}
    building = as_geom((aoi.get("analysis_zones") or {}).get("building"))
    from shapely.geometry import Point
    point = Point(g["lon"], g["lat"]) if g.get("lon") is not None else None

    cube = fraction_cube(site_dir, g, M)
    if not cube:
        return None
    zmasks = zone_masks(g)
    ords = sorted(cube)

    by = {n: {o: cube[o][..., FRACTION_INDEX[n]] for o in ords} for n in EM_NAMES}

    # combined change mask per period, reused for persistence
    combined = {}
    layers = {}
    for o in ords:
        L = {}
        for name, direction, key in (("soil", "up", "new_soil"),
                                     ("high_albedo", "up", "new_high"),
                                     ("vegetation", "down", "veg_loss")):
            med, sig, n = change.seasonal_baseline(by[name], o, cfg)
            L[key] = change.change_mask(by[name][o], med, sig, n, cfg, direction)
        layers[o] = L
        combined[o] = L["new_soil"] | L["new_high"]

    # --- causal detection, PER ZONE -----------------------------------------
    # Detection used to run only over `development`, so every building-zone
    # record carried detection_state=None and any scanning number measured on
    # that zone was a plumbing artefact rather than a result -- 0 detections and
    # 0 false alarms, which reads like a perfect detector and is an empty one.
    series_by, dets_by, det_by_zone, prior_veg_by = {}, {}, {}, {}
    for zname, zm in zmasks.items():
        prior_veg = {}
        veg_run = 0.0
        series_z = []
        for o in ords:
            ch = combined[o] & (zm if zm is not None else True)
            vl = layers[o]["veg_loss"] & (zm if zm is not None else True)
            veg_run = max(veg_run, float(vl.sum()) * px_area)
            prior_veg[o] = veg_run      # only ever accumulates from the PAST
            series_z.append({"period": unordn_(o), "clear": True,
                             "changed": bool(ch.sum() * px_area
                                             >= cfg.get("mmu_m2", 8000.0))})
        series_by[zname] = series_z
        dets_by[zname] = detect.run_causal(series_z)
        det_by_zone[zname] = {d.period: d for d in dets_by[zname]}
        prior_veg_by[zname] = prior_veg

    dev_mask = zmasks.get("development")
    # the campus-level record still keys off development
    series = series_by.get("development") or next(iter(series_by.values()), [])
    dets = dets_by.get("development") or next(iter(dets_by.values()), [])

    tiled = None
    if prov.get("aoi_needs_tiling") and dev_mask is not None:
        tiled = {}
        for o in ords:
            lab, cr = tiling.tiled_components(
                combined[o] & dev_mask,
                tile_px=prov.get("tile_px") or 64,
                overlap_px=prov.get("tile_overlap_px") or 16,
                min_px=cfg["min_component_px"])
            tiled[o] = tiling.rank_disturbances(lab, cr, px_area,
                                                building_geom=building, point=point,
                                                bbox=bbox, width=W, height=H,
                                                top_k=5)

    recs = []
    for i, o in enumerate(ords):
        # BACKWARD persistence only: observations at or before this period
        for zname, zm in zmasks.items():
            pers = detect.causal_persistence(series_by[zname], i)
            f = change.zone_features(zm, layers[o], px_area, bbox, W, H,
                                     building_geom=building, point=point,
                                     cfg=cfg, persistence_masks=None)
            f.persistence = pers
            f.zone = zname
            d = f.as_dict()
            d["prior_veg_loss_m2"] = prior_veg_by[zname].get(o)
            d["visual_confirmed"] = None
            dd = det_by_zone[zname].get(unordn_(o))
            if dd is not None:
                d["detection_state"] = dd.state
                d["provisional_period"] = dd.provisional_period
                d["confirmed_period"] = dd.confirmed_period
                d["n_clear_seen"] = dd.n_clear_seen
            if zname == "development":
                if tiled is not None:
                    d["tiled_top_disturbances"] = tiled.get(o, [])
                sc = stage_mod.classify(d, aoi_source=prov.get("aoi_source"))
                d["stage_pct"] = sc.stage_pct
                d["stage_group"] = sc.stage_group
                d["stage_label"] = sc.label
                d["stage_confidence"] = sc.confidence
                d["stage_reasons"] = sc.reasons
            d["period"] = None
            recs.append({"period_ord": o, "zone": zname, **d, **prov})
    for r in recs:
        r["period"] = unordn_(r["period_ord"])
        r.pop("period_ord", None)

    from acd_core import period_bounds
    interval = bracket_from_detections(
        g.get("uid", "?"), dets, period_bounds,
        reported_date=None, reported_period=g.get("start_period"),
        reported_source="reviewer")
    return {"site": {k: g.get(k) for k in ("uid", "name", "state", "start_period")},
            "aoi_provenance": prov, "px_area_m2": px_area,
            "n_periods": len(ords),
            "start_interval": interval.as_dict(),
            "detections": [d.as_dict() for d in dets],
            "records": recs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--endmembers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-sigma", type=float, default=change.DEFAULTS["k_sigma"])
    ap.add_argument("--min-component-px", type=int,
                    default=change.DEFAULTS["min_component_px"])
    a = ap.parse_args()

    cfg = dict(change.DEFAULTS)
    cfg["k_sigma"] = a.k_sigma
    cfg["min_component_px"] = a.min_component_px

    src = json.load(open(a.endmembers))["endmembers"]
    M = np.vstack([src[n] for n in EM_NAMES])
    os.makedirs(a.out, exist_ok=True)

    for d in sorted(glob.glob(os.path.join(a.root, "*"))):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "grid.json")):
            continue
        res = run_site(d, M, cfg)
        if res is None:
            print(f"  {os.path.basename(d)[:44]:<44} no clear periods")
            continue
        json.dump(res, open(os.path.join(a.out, os.path.basename(d) + ".json"), "w"),
                  indent=1)
        dev = [r for r in res["records"] if r["zone"] == "development"]
        staged = [r for r in dev if (r.get("stage_pct") or 0) > 0]
        abstain = [r for r in dev if r.get("stage_group") == "10-15"]
        si = res["start_interval"]
        print(f"  {os.path.basename(d)[:40]:<40} AOI={res['aoi_provenance'].get('aoi_source','?'):<9}"
              f" {res['n_periods']:3d}p  stage>0 {len(staged):3d}  abstain {len(abstain):3d}"
              f"  interval {si.get('last_unchanged_period')}..{si.get('first_changed_period')}")


if __name__ == "__main__":
    main()
