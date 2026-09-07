"""
Overlapping tiles over a large AOI, with components merged across tile seams.

A 900 ha parcel is not a reason to give up on a site -- it is the only search
boundary we have. Rejecting it throws away the boundary; averaging over it
buries an 8,000 m2 event at 0.09% of the mean. So the parcel is kept whole as a
BOUNDARY and searched in tiles.

Tiles OVERLAP by design. A disturbance sitting on a seam would otherwise be
split into two sub-threshold fragments and dropped by the minimum-component
filter -- the exact failure the filter exists to avoid causing. Overlap means
such a component is seen whole by at least one tile, and the union-find merge
below then reconciles the duplicates.

Merging is by shared pixels in the GLOBAL grid, not by tile adjacency
bookkeeping: two local labels that cover any of the same global pixel are the
same disturbance. That is robust to any tiling geometry, including the ragged
partial tiles at an AOI edge.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage

_Q8 = np.ones((3, 3), int)


@dataclass
class Tile:
    ix: int
    iy: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def slices(self):
        return (slice(self.y0, self.y1), slice(self.x0, self.x1))


def make_tiles(width: int, height: int, tile_px: int, overlap_px: int) -> list[Tile]:
    """Cover the grid with tiles of `tile_px`, stepping by tile-overlap."""
    step = max(1, tile_px - overlap_px)
    out, iy = [], 0
    y = 0
    while True:
        y1 = min(y + tile_px, height)
        ix, x = 0, 0
        while True:
            x1 = min(x + tile_px, width)
            out.append(Tile(ix, iy, x, y, x1, y1))
            if x1 >= width:
                break
            x += step
            ix += 1
        if y1 >= height:
            break
        y += step
        iy += 1
    return out


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def tiled_components(mask: np.ndarray, tile_px: int = 64, overlap_px: int = 16,
                     min_px: int = 6) -> tuple[np.ndarray, list[dict]]:
    """Label `mask` tile by tile, then merge across seams.

    Returns a global label image and one record per merged component. The
    minimum-size filter is applied AFTER merging, so a component that is
    sub-threshold inside every individual tile but large once reassembled is
    correctly kept -- filtering per tile would have discarded it.
    """
    H, W = mask.shape
    tiles = make_tiles(W, H, tile_px, overlap_px)
    owner = np.zeros((H, W), np.int64)          # global pixel -> local key id
    uf = _UF()
    key_of = {}
    nxt = 1

    for t in tiles:
        sub = mask[t.slices]
        if not sub.any():
            continue
        lab, n = ndimage.label(sub, structure=_Q8)
        for k in range(1, n + 1):
            ys, xs = np.nonzero(lab == k)
            gy, gx = ys + t.y0, xs + t.x0
            kid = key_of.setdefault((t.ix, t.iy, k), nxt)
            if kid == nxt:
                nxt += 1
            uf.find(kid)
            prev = owner[gy, gx]
            for other in np.unique(prev[prev > 0]):
                uf.union(int(other), kid)       # same pixels -> same component
            owner[gy, gx] = kid

    if nxt == 1:
        return np.zeros((H, W), np.int64), []

    roots = {}
    out = np.zeros((H, W), np.int64)
    nz = owner > 0
    if nz.any():
        vals = owner[nz]
        mapped = np.array([roots.setdefault(uf.find(int(v)), len(roots) + 1)
                           for v in vals], np.int64)
        out[nz] = mapped

    recs = []
    for lbl in range(1, len(roots) + 1):
        m = out == lbl
        n = int(m.sum())
        if n < min_px:
            out[m] = 0
            continue
        ys, xs = np.nonzero(m)
        recs.append({"label": lbl, "n_px": n,
                     "cy": float(ys.mean()), "cx": float(xs.mean()),
                     "bbox_px": [int(xs.min()), int(ys.min()),
                                 int(xs.max()) + 1, int(ys.max()) + 1]})
    # relabel densely after the size filter
    keep = sorted(r["label"] for r in recs)
    remap = {old: i + 1 for i, old in enumerate(keep)}
    final = np.zeros_like(out)
    for old, new in remap.items():
        final[out == old] = new
    for r in recs:
        r["label"] = remap[r["label"]]
    return final, recs


def rank_disturbances(labels, recs, px_area_m2, *, building_geom=None,
                      point=None, bbox=None, width=None, height=None,
                      shape_fn=None, top_k=None) -> list[dict]:
    """Rank merged components by how much they look like localized disturbance.

    Score is deliberately simple and legible: area above the minimum mapping
    unit, tightened by shape, and pulled up by proximity to a planned footprint
    or the project point. It is a ranking aid for a review queue, NOT a
    calibrated probability -- no labelled stage data exists to fit one.
    """
    from change import distance_m, pixel_to_lonlat, shape_metrics
    from shapely.geometry import Point as _P

    out = []
    for r in recs:
        m = labels == r["label"]
        area = r["n_px"] * px_area_m2
        rec = dict(r)
        rec["area_m2"] = area
        if shape_fn is not None:
            rec.update(shape_fn(m))
        else:
            rec.update(shape_metrics(m, px_area_m2 ** 0.5))
        d_b = d_p = None
        if bbox is not None:
            lon, lat = pixel_to_lonlat(bbox, width, height, r["cx"], r["cy"])
            cen = _P(lon, lat)
            rec["lon"], rec["lat"] = round(lon, 7), round(lat, 7)
            if building_geom is not None:
                d_b = distance_m(cen, building_geom, lat)
            if point is not None:
                d_p = distance_m(cen, point, lat)
        rec["dist_to_building_m"] = None if d_b is None else round(d_b, 1)
        rec["dist_to_point_m"] = None if d_p is None else round(d_p, 1)

        score = min(area / 8000.0, 4.0)
        rect = rec.get("rectangularity")
        if rect is not None:
            score *= (0.6 + 0.8 * rect)
        near = d_b if d_b is not None else d_p
        if near is not None:
            score *= 1.0 / (1.0 + near / 500.0)
        rec["disturbance_score"] = round(float(score), 4)
        out.append(rec)

    out.sort(key=lambda r: -r["disturbance_score"])
    return out[:top_k] if top_k else out
