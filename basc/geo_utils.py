"""
Geometry primitives: metre-accurate buffering, rasterisation, geodesic measures.

Everything here works in EPSG:4326 lon/lat but never treats a degree as a
distance. Two rules the rest of the code depends on:

  * buffering happens in METRES. A degree of longitude is 111.32*cos(lat) km
    against 110.54 km for latitude, so `geom.buffer(metres / 111000)` inflates
    a footprint by ~28% more north-south than east-west at 39N, and by ~100%
    more at 60N. Every buffer goes through buffer_m().

  * cloud fraction is measured against RASTERISED AOI PIXELS, never against
    polygon-area / bbox-area. The two differ whenever the AOI is small relative
    to the pixel grid or its edges cut pixels, and the pixel count is what the
    imagery actually delivers.

pyproj.Geod is used for measurement (it needs no PROJ database). CRS
transforms are deliberately avoided: the geo env's PROJ database is broken, so
to_crs()/Transformer would fail or silently mis-project.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from pyproj import Geod
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

_GEOD = Geod(ellps="WGS84")

M_PER_DEG_LAT = 110_540.0      # local, WGS84, mid-latitudes
M_PER_DEG_LON_EQ = 111_320.0


def m_per_deg_lon(lat: float) -> float:
    return M_PER_DEG_LON_EQ * max(math.cos(math.radians(lat)), 1e-6)


def buffer_m(geom, metres: float, lat: float | None = None):
    """Buffer a lon/lat geometry by `metres`, isotropically on the ground.

    Local equirectangular method: scale longitude by cos(lat) so one unit means
    the same distance on both axes, buffer in that frame, scale back. Accurate
    to well under a percent for the AOI sizes here (< ~10 km); a projected CRS
    would be equivalent but needs a working PROJ database.
    """
    if metres == 0:
        return geom
    if lat is None:
        lat = geom.centroid.y
    k = max(math.cos(math.radians(lat)), 1e-6)
    g = scale(geom, xfact=k, yfact=1.0, origin=(0, 0))
    g = g.buffer(metres / M_PER_DEG_LAT)
    return scale(g, xfact=1.0 / k, yfact=1.0, origin=(0, 0))


def geodesic_area_m2(geom) -> float:
    """True surface area. Sums polygons and subtracts holes."""
    if geom is None or geom.is_empty:
        return 0.0
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    total = 0.0
    for p in polys:
        a, _ = _GEOD.geometry_area_perimeter(p)
        total += abs(a)
    return float(total)


def geodesic_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    _, _, d = _GEOD.inv(lon1, lat1, lon2, lat2)
    return float(d)


def pixel_centres(bbox, width: int, height: int):
    """Lon/lat of every pixel centre, row 0 at the TOP (north), as (H,W) arrays."""
    x0, y0, x1, y1 = bbox
    xs = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
    ys = y1 - (np.arange(height) + 0.5) * (y1 - y0) / height
    return np.meshgrid(xs, ys)


def rasterize(geom, bbox, width: int, height: int) -> np.ndarray:
    """Boolean (height, width) mask of pixel centres inside `geom`.

    matplotlib's path test is used rather than rasterio (not installed) or a
    shapely point loop (far too slow at these grid sizes). Holes are handled by
    testing exteriors and subtracting interiors.
    """
    from matplotlib.path import Path

    lon, lat = pixel_centres(bbox, width, height)
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    inside = np.zeros(len(pts), bool)
    if geom is None or geom.is_empty:
        return inside.reshape(height, width)
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.is_empty:
            continue
        inside |= Path(np.asarray(p.exterior.coords)).contains_points(pts)
        for ring in p.interiors:
            inside &= ~Path(np.asarray(ring.coords)).contains_points(pts)
    return inside.reshape(height, width)


def pixel_area_m2(bbox, width: int, height: int) -> float:
    x0, y0, x1, y1 = bbox
    lat = (y0 + y1) / 2.0
    return ((x1 - x0) / width * m_per_deg_lon(lat)) * ((y1 - y0) / height * M_PER_DEG_LAT)


def ground_square(lat: float, lon: float, side_m: float) -> Polygon:
    """A square `side_m` on a side, centred on (lat, lon). Metres, not degrees."""
    half = side_m / 2.0
    dlat = half / M_PER_DEG_LAT
    dlon = half / m_per_deg_lon(lat)
    return Polygon([(lon - dlon, lat - dlat), (lon + dlon, lat - dlat),
                    (lon + dlon, lat + dlat), (lon - dlon, lat + dlat)])


def safe_union(geoms: Iterable) -> Polygon | MultiPolygon | None:
    """Union that survives self-intersecting rings (buffer(0) repair)."""
    good = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if not g.is_valid:
            g = g.buffer(0)
            if g.is_empty:
                continue
        good.append(g)
    if not good:
        return None
    u = unary_union(good)
    return None if u.is_empty else u


def as_geom(obj):
    """GeoJSON mapping or shapely -> shapely, or None."""
    if obj is None:
        return None
    if hasattr(obj, "geom_type"):
        return obj
    try:
        return shape(obj)
    except Exception:
        return None
