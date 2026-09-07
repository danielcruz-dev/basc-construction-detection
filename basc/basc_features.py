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
import stage as stage_mod
from basc_lsma import EM_NAMES, load_stack, unmix
from geo_utils import as_geom, pixel_area_m2, rasterize
from replay import ordn

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


def run_site(site_dir, M, cfg, persist_ahead=4):
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

    recs = []
    for i, o in enumerate(ords):
        later = [combined[x] for x in ords[i + 1:i + 1 + persist_ahead]]
        for zname, zm in zmasks.items():
            f = change.zone_features(zm, layers[o], px_area, bbox, W, H,
                                     building_geom=building, point=point,
                                     cfg=cfg, persistence_masks=later or None)
            f.zone = zname
            d = f.as_dict()
            if zname == "development":
                sc = stage_mod.classify(d, aoi_source=prov.get("aoi_source"))
                d["stage_pct"] = sc.stage_pct
                d["stage_label"] = sc.label
                d["stage_confidence"] = sc.confidence
                d["stage_reasons"] = sc.reasons
            d["period"] = None
            recs.append({"period_ord": o, "zone": zname, **d, **prov})
    # attach human-readable periods
    from replay import unordn
    for r in recs:
        r["period"] = unordn(r["period_ord"])
        r.pop("period_ord", None)
    return {"site": {k: g.get(k) for k in ("uid", "name", "state", "start_period")},
            "aoi_provenance": prov, "px_area_m2": px_area,
            "n_periods": len(ords), "records": recs}


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
        staged = [r for r in dev if r.get("stage_pct", 0) > 0]
        print(f"  {os.path.basename(d)[:44]:<44} AOI={res['aoi_provenance'].get('aoi_source','?'):<9}"
              f" {res['n_periods']:3d} periods, {len(staged):3d} with stage>0")


if __name__ == "__main__":
    main()
