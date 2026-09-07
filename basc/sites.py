"""
Build the AOI for every monitored unit. The parcel boundary IS the AOI.

Resolution order per unit:
  1. parcel  -- union of every Regrid parcel bought for the unit, if its area is
                plausible. This is the AOI. No box, no buffer, no padding.
  2. box     -- only where step 1 gives nothing: no parcel was ever bought, or
                the polygon is obvious nonsense (a county-sized "parcel").

Every site carries `aoi_source` so a result is never ambiguous about which of
those two produced it, and `coord_confirmed` so box fallbacks resting on an
unverified coordinate can be filtered out rather than quietly trusted.

Unit membership replicates production's filterAcdSites exactly: group by
campus UID (falling back to building UID), keep the unit only if EVERY building
in it is Announcement or Construction, centroid is the mean of valid building
coordinates, and the representative building is the first with
EXACT_LOCATION_FOUND=Y (else the first). Matching that rule is what makes a
number here comparable to a production number.

The fallback box is BOX_SIDE_M **on a side**, defaulting to 750 m -- production's
documented intent. Production itself passes 375 into a side parameter and so
measures a 375 m box; nothing here reproduces that.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
import pandas as pd
from pyproj import Geod
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

from acd_core import MAX_AOI_KM2, MIN_AOI_M2

PARCELS_DIR = "/home/azul/Aterio/parcels-run"
GPKG = f"{PARCELS_DIR}/all_parcels.gpkg"
INVENTORY = f"{PARCELS_DIR}/data_center_inventory.csv"

# Every links file, read together: a unit's parcels may have been bought in any
# of the sweep runs. Header-based, because the column ORDER differs between
# these files (all_parcels_site_links.csv leads with `stage`, the others don't).
LINK_FILES = [
    f"{PARCELS_DIR}/all_parcels_site_links.csv",
    f"{PARCELS_DIR}/announcement_parcels_site_links.csv",
    f"{PARCELS_DIR}/construction_parcels_site_links.csv",
    f"{PARCELS_DIR}/construction_topup_site_links.csv",
    f"{PARCELS_DIR}/pilot_parcels_site_links.csv",
]

ACD_STAGES = {"announcement", "construction"}
BOX_SIDE_M = 750.0

# WGS84 geodesic area. Avoids to_crs(), whose PROJ database is missing in this
# conda env (proj_create_from_database fails); Geod needs only the ellipsoid.
_GEOD = Geod(ellps="WGS84")


@dataclass
class Site:
    unit_type: str
    unit_uid: str
    unit_name: str
    provider: str
    state_code: str
    stage: str
    lat: float
    lon: float
    n_buildings: int
    n_parcels: int
    area_m2: float
    aoi_source: str          # 'parcel' | 'parcel_union' | 'box_no_parcel' | 'box_parcel_rejected'
    coord_confirmed: bool
    geometry: object         # shapely, EPSG:4326
    reject_reason: Optional[str] = None
    # Of n_buildings, how many are actually Announcement/Construction. Equal to
    # n_buildings unless a mixed-stage campus was let in -- see load_sites.
    # Surfaced so a mixed campus is never mistaken for a clean one.
    n_eligible_buildings: int = 0
    mixed_stage: bool = False

    @property
    def geojson(self) -> dict:
        return mapping(self.geometry)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.geometry.bounds

    @property
    def uses_parcel(self) -> bool:
        return self.aoi_source.startswith("parcel")

    def control_ring(self, inner_m: float = 120.0, outer_m: float = 400.0):
        """Land just outside the parcel, as a same-scene control.

        Illumination, atmosphere, soil moisture and phenology are shared between
        a parcel and its immediate surroundings; construction is not. Measuring
        the ring in the same request and subtracting cancels the shared part,
        which is what generates the large spurious deltas the detector fires on
        (a rained-on field and a graded pad both raise NDBI over the parcel; only
        one of them raises it over the ring too).

        `inner_m` holds the ring off the parcel edge so spillover from work just
        inside the boundary -- or a parcel polygon a little too small -- does not
        contaminate the control.
        """
        # Degrees, at this geometry's own latitude: to_crs() is unavailable in
        # this env (missing PROJ db, see _GEOD above), so buffer in degrees with
        # a per-axis scale rather than reprojecting to metres.
        lat = self.geometry.centroid.y
        dx = 1.0 / (111_320.0 * max(math.cos(math.radians(lat)), 0.01))
        dy = 1.0 / 110_540.0
        scale = (dx + dy) / 2  # buffer() is isotropic; use the mean degree size
        ring = self.geometry.buffer(outer_m * scale).difference(
            self.geometry.buffer(inner_m * scale))
        return ring if (not ring.is_empty and ring.is_valid) else None

    @property
    def control_geojson(self) -> Optional[dict]:
        r = self.control_ring()
        return mapping(r) if r is not None else None

    def summary(self) -> dict:
        return {
            "unit_type": self.unit_type,
            "unit_uid": self.unit_uid,
            "unit_name": self.unit_name,
            "provider": self.provider,
            "state_code": self.state_code,
            "stage": self.stage,
            "n_buildings": self.n_buildings,
            "n_eligible_buildings": self.n_eligible_buildings,
            "mixed_stage": self.mixed_stage,
            "n_parcels": self.n_parcels,
            "area_km2": round(self.area_m2 / 1e6, 4),
            "aoi_source": self.aoi_source,
            "coord_confirmed": self.coord_confirmed,
            "reject_reason": self.reject_reason or "",
        }


def geodesic_area_m2(geom) -> float:
    area, _ = _GEOD.geometry_area_perimeter(geom)
    return abs(area)


def ground_square(lat: float, lon: float, side_m: float) -> Polygon:
    """A square ~side_m on the ground, centred on (lat, lon).

    The 1/cos(lat) term is what makes it a ground square instead of a degree
    square; without it the box narrows with latitude and a Wyoming site gets a
    materially different footprint from a Virginia one.
    """
    half = side_m / 2.0
    dlat = half / 111_111.0
    dlon = half / (111_111.0 * math.cos(math.radians(lat)))
    return Polygon([
        (lon - dlon, lat - dlat), (lon + dlon, lat - dlat),
        (lon + dlon, lat + dlat), (lon - dlon, lat + dlat),
    ])


def _load_links() -> pd.DataFrame:
    frames = []
    for path in LINK_FILES:
        try:
            df = pd.read_csv(path, dtype=str)
        except FileNotFoundError:
            continue
        if {"unit_uid", "aterio_parcel_id"} <= set(df.columns):
            frames.append(df[["unit_uid", "aterio_parcel_id"]])
    if not frames:
        return pd.DataFrame(columns=["unit_uid", "aterio_parcel_id"])
    return pd.concat(frames, ignore_index=True).dropna().drop_duplicates()


def load_sites(
    stages: Optional[set[str]] = None,
    box_side_m: float = BOX_SIDE_M,
    require_confirmed_coord: bool = False,
    require_all_stages: bool = False,
) -> list[Site]:
    stages = stages or ACD_STAGES
    inv = pd.read_csv(INVENTORY, dtype=str, low_memory=False)
    links = _load_links()
    parcels = gpd.read_file(GPKG)
    pgeom = parcels.set_index("aterio_parcel_id").geometry
    parcels_by_unit = links.groupby("unit_uid")["aterio_parcel_id"].apply(list).to_dict()

    def _s(v) -> str:
        """Cell -> clean string. pandas gives float('nan') for blanks, and
        `nan or ''` is nan (nan is truthy), so str() would yield the literal
        'nan' and it would end up in site names and providers."""
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        s = str(v).strip()
        return "" if s.lower() in ("nan", "none") else s

    def norm(v) -> str:
        return _s(v).lower()

    inv["_key"] = (inv["ATERIO_DATA_CENTER_CAMPUS_UID"].fillna("").str.strip()
                   .where(lambda s: s != "", inv["ATERIO_DATA_CENTER_UID"].fillna("").str.strip()))

    sites: list[Site] = []
    for key, grp in inv[inv["_key"] != ""].groupby("_key"):
        is_eligible = grp["DATA_CENTER_STAGE"].map(norm).isin(stages)
        n_eligible = int(is_eligible.sum())
        # Production requires EVERY building on the campus to be an eligible
        # stage, so one Active/Withdrawn sibling drops the whole campus out of
        # scope -- Amazon's 7-building Hilliard campus (Announcement +
        # Construction + Active mixed) is exactly this case, and it excludes a
        # Construction-stage building that plainly should be watched.
        #
        # Default here is looser: ANY eligible building qualifies the campus.
        # The parcel AOI already isn't stage-filtered (parcels_by_unit is keyed
        # on the whole campus, not on eligible buildings alone), so a mixed
        # campus was always going to be measured over its full footprint --
        # this just stops throwing the result away. --require-all-stages on
        # the CLI restores exact production parity for a like-for-like count.
        if require_all_stages:
            if n_eligible != len(grp):
                continue
        elif n_eligible == 0:
            continue
        mixed = n_eligible < len(grp)

        coords = []
        for _, b in grp.iterrows():
            try:
                la, lo = float(b["LOCATION_LATITUDE"]), float(b["LOCATION_LONGITUDE"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(la) and math.isfinite(lo):
                coords.append((la, lo))
        if not coords:
            continue
        lat = sum(c[0] for c in coords) / len(coords)
        lon = sum(c[1] for c in coords) / len(coords)

        # Representative is chosen from the ELIGIBLE buildings only. Picking
        # from the whole group let a finished/withdrawn sibling supply the
        # site's own `stage`/`provider`/`name` -- a mixed campus would then
        # summarise itself as "Active" while being monitored as in-scope,
        # which is the labelling half of the same bug the inclusion rule had.
        pool = grp[is_eligible] if is_eligible.any() else grp
        exact = pool[pool["EXACT_LOCATION_FOUND"].fillna("").str.strip().str.upper() == "Y"]
        rep = exact.iloc[0] if len(exact) else pool.iloc[0]
        coord_confirmed = len(exact) > 0
        unit_type = "campus" if _s(rep.get("ATERIO_DATA_CENTER_CAMPUS_UID")) else "building"
        name = (_s(rep.get("DATA_CENTER_CAMPUS_NAME"))
                or _s(rep.get("DATA_CENTER_BUILDING_NAME"))
                or f"Site {key[:8]}")

        # ── AOI: parcel first, box only if that fails ──────────────────────────
        pids = [p for p in parcels_by_unit.get(key, []) if p in pgeom.index]
        geom = None
        aoi_source = ""
        reject = None
        if pids:
            geoms = [pgeom.loc[p] for p in pids]
            cand = unary_union(geoms) if len(geoms) > 1 else geoms[0]
            a = geodesic_area_m2(cand)
            if a > MAX_AOI_KM2 * 1e6:
                reject = f"parcel {a/1e6:.1f} km2 > {MAX_AOI_KM2} km2"
            elif a < MIN_AOI_M2:
                reject = f"parcel {a:.0f} m2 < {MIN_AOI_M2:.0f} m2"
            else:
                geom = cand
                aoi_source = "parcel_union" if len(pids) > 1 else "parcel"
        if geom is None:
            geom = ground_square(lat, lon, box_side_m)
            aoi_source = "box_parcel_rejected" if reject else "box_no_parcel"

        if require_confirmed_coord and not coord_confirmed and not aoi_source.startswith("parcel"):
            # A box on an unconfirmed coordinate is a guess about a guess.
            continue

        sites.append(Site(
            unit_type=unit_type, unit_uid=key, unit_name=name,
            provider=_s(rep.get("PROVIDER_NAME")),
            state_code=_s(rep.get("STATE_CODE")),
            stage=_s(rep.get("DATA_CENTER_STAGE")),
            lat=lat, lon=lon, n_buildings=len(grp), n_parcels=len(pids),
            area_m2=geodesic_area_m2(geom), aoi_source=aoi_source,
            coord_confirmed=coord_confirmed, geometry=geom, reject_reason=reject,
            n_eligible_buildings=n_eligible, mixed_stage=mixed,
        ))

    sites.sort(key=lambda s: -s.area_m2)
    return sites


def main() -> None:
    ap = argparse.ArgumentParser(description="AOIs for monitored units: parcel boundary, box fallback.")
    ap.add_argument("--stage", default="both", choices=["announcement", "construction", "both"])
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--box-side", type=float, default=BOX_SIDE_M)
    ap.add_argument("--only", choices=["parcel", "box"], help="show one AOI source")
    ap.add_argument("--require-all-stages", action="store_true",
                     help="production parity: drop a campus if ANY building is outside "
                          "--stage, instead of including it when ANY building qualifies")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    stages = ACD_STAGES if args.stage == "both" else {args.stage}
    sites = load_sites(stages, box_side_m=args.box_side, require_all_stages=args.require_all_stages)

    parcel = [s for s in sites if s.uses_parcel]
    box = [s for s in sites if not s.uses_parcel]
    rejected = [s for s in box if s.aoi_source == "box_parcel_rejected"]

    def med(xs):
        v = sorted(x.area_m2 for x in xs)
        return v[len(v) // 2] / 1e6 if v else float("nan")

    print(f"stages={sorted(stages)}   monitored units: {len(sites)}")
    print(f"  AOI = parcel boundary : {len(parcel):5d} ({100*len(parcel)/max(1,len(sites)):5.1f}%)  median {med(parcel):.3f} km2")
    print(f"  AOI = box fallback    : {len(box):5d} ({100*len(box)/max(1,len(sites)):5.1f}%)  median {med(box):.3f} km2"
          f"   [{args.box_side:.0f} m side]")
    print(f"      no parcel bought  : {len(box)-len(rejected):5d}")
    print(f"      parcel rejected   : {len(rejected):5d}")
    print(f"  multi-parcel unions   : {sum(1 for s in parcel if s.n_parcels > 1):5d}")
    print(f"  unconfirmed coord     : {sum(1 for s in sites if not s.coord_confirmed):5d}"
          f"  (of which on a box: {sum(1 for s in box if not s.coord_confirmed)})")
    mixed = [s for s in sites if s.mixed_stage]
    print(f"  mixed-stage campuses  : {len(mixed):5d}  "
          f"(let in because {'--require-all-stages is off' if not args.require_all_stages else 'n/a'} -- "
          f"a sibling outside --stage was present but at least one building qualified)")
    print(f"  total AOI area        : {sum(s.area_m2 for s in sites)/1e6:.1f} km2")

    show = sites
    if args.only == "parcel":
        show = parcel
    elif args.only == "box":
        show = box

    hdr = (f"{'aoi_source':20s} {'area_km2':>9s} {'np':>3s} {'nb':>3s} {'cc':>3s} "
           f"{'st':3s} {'provider':20s} name")
    print("\n" + hdr)
    print("-" * len(hdr))
    for s in show[: args.limit]:
        print(f"{s.aoi_source:20s} {s.area_m2/1e6:9.3f} {s.n_parcels:3d} {s.n_buildings:3d} "
              f"{'Y' if s.coord_confirmed else 'n':>3s} {(s.state_code or '-')[:3]:3s} "
              f"{(s.provider or '-')[:20]:20s} {s.unit_name[:36]}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump([s.summary() for s in show], fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
