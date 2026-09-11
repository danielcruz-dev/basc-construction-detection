"""Add planned-footprint analysis zones to existing grid.json files.

Lever 2: measure change INSIDE the buildings a site plan says are coming, rather
than averaged over the whole parcel. An 8,000 m2 pad is ~1.5% of a 60 ha parcel
and a large fraction of a footprint, which is the entire signal-to-noise argument.

This does NOT refetch. Only `aoi.analysis_zones` is written; bbox, width, height
and geometry are left exactly as they are, because they describe rasters already
on disk and changing them would silently decouple the grid from its .tif files.

Negatives have no site plan, so they get a PLACEBO footprint: the geometry of
their matched positive's footprint union, translated onto the negative's parcel
centroid and clipped to the parcel. Same size and shape, somewhere nothing was
ever built -- so anything it fires on is a false alarm. Without it the footprint
detector has no negative class at all and no false-alarm rate can be computed.

  python inject_zones.py --roots results/pos16par,results/neg20 \
                         --pairs results/pairs_neg20.json

Lever 3, the control ring. A grid fetched with --outer-buffer-m carries the
parcel in `aoi_geometry` and the buffered fetch extent in `geometry`. Their
difference is a ring of land nobody is building on, seen by the same sensor on
the same date through the same atmosphere. Whatever fires there is not
construction: a regional dry-down, a ploughing season, a snowmelt, a baseline
year that was unusually green. It is written as the `context` zone so the
detector can be gated on it. For such grids `development` is ALWAYS written as
the parcel, footprint or not -- otherwise the whole-raster fallback in
basc_features.zone_masks would silently make the ring part of the parcel.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from shapely.affinity import translate
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from geo_utils import buffer_m, geodesic_area_m2, safe_union

PLAN_META = "/home/azul/Aterio/acd-restored/public/poc/site-plans/plan-meta.json"


def footprints_by_campus(path):
    out = {}
    for l in json.load(open(path))["layers"]:
        uid = l.get("campus_uid")
        if not uid:
            continue
        polys = [shape({"type": "Polygon", "coordinates": [f["ring"]]})
                 for f in (l.get("features") or [])
                 if f.get("ring") and len(f["ring"]) >= 4]
        if polys:
            out.setdefault(uid, []).extend(polys)
    return {k: safe_union(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True)
    ap.add_argument("--pairs", default="results/pairs_neg20.json")
    ap.add_argument("--buffer-m", type=float, default=50.0,
                    help="outward buffer on the footprint union. A pad is graded "
                         "wider than the slab, and the plan's georeferencing is not "
                         "exact; 0 would measure only inside the drawn line.")
    a = ap.parse_args()

    fps = footprints_by_campus(PLAN_META)
    pairs = json.load(open(a.pairs)) if os.path.exists(a.pairs) else {}
    print(f"campuses with footprints: {len(fps)}   pairs: {len(pairs)}")

    grids = {}
    for root in a.roots.split(","):
        for gp in sorted(glob.glob(os.path.join(root, "*", "grid.json"))):
            grids[json.load(open(gp))["uid"]] = gp

    done = {"real": 0, "placebo": 0, "none": 0}
    for uid, gp in grids.items():
        g = json.load(open(gp))
        parcel = shape(g["aoi_geometry"] if g.get("aoi_geometry") else g["geometry"])
        ring = None
        if g.get("aoi_geometry") and (g.get("outer_buffer_m") or 0) > 0:
            ring = shape(g["geometry"]).difference(parcel)
        fp, kind = fps.get(uid), "real"
        if fp is None and uid in pairs and pairs[uid] in fps:
            # placebo: same footprint, moved onto this parcel, clipped to it
            src = fps[pairs[uid]]
            dx = parcel.centroid.x - src.centroid.x
            dy = parcel.centroid.y - src.centroid.y
            fp = translate(src, xoff=dx, yoff=dy).intersection(parcel)
            kind = "placebo"
        aoi = g.get("aoi") or {}
        zones = dict(aoi.get("analysis_zones") or {})
        zones["development"] = mapping(parcel)
        if fp is None or fp.is_empty:
            done["none"] += 1
            kind = "none"
            zones.pop("building", None)
        else:
            zone = buffer_m(fp, a.buffer_m, g["lat"])
            zones["building"] = mapping(zone)
            done[kind] += 1
            if ring is not None:
                # the 50 m footprint buffer spills past a small parcel; the
                # control must not contain the thing it controls for
                ring = ring.difference(zone)
        if ring is not None and not ring.is_empty:
            zones["context"] = mapping(ring)
        aoi["analysis_zones"] = zones
        aoi["footprint_kind"] = kind
        g["aoi"] = aoi
        json.dump(g, open(gp, "w"))
        ring_txt = (f"  ring {geodesic_area_m2(ring)/1e4:7.1f} ha"
                    if ring is not None and not ring.is_empty else "")
        if kind == "none":
            print(f"  {g['name'][:38]:<38} {kind:<8} no footprint; parcel "
                  f"{geodesic_area_m2(parcel)/1e4:7.1f} ha{ring_txt}")
            continue
        print(f"  {g['name'][:38]:<38} {kind:<8} "
              f"footprint {geodesic_area_m2(zone)/1e4:7.1f} ha of "
              f"{geodesic_area_m2(parcel)/1e4:7.1f} ha parcel  "
              f"({100*geodesic_area_m2(zone)/geodesic_area_m2(parcel):4.1f}%){ring_txt}")
    print(f"\nreal {done['real']}   placebo {done['placebo']}   none {done['none']}")


if __name__ == "__main__":
    main()
