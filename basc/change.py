"""
Change detection against a per-pixel SEASONAL baseline, and zone features.

The distinction that matters: "changed" here does NOT mean "soil fraction is
above 0.5". It means "this pixel's soil fraction is far above what THIS PIXEL
usually shows at THIS TIME OF YEAR". An absolute threshold cannot tell a
construction pad from a ploughed field in February, and February is 15.8% of
all verified starts in this dataset -- the single largest month, at 48% clear.

Baseline construction, per pixel and per fraction:

    take the same calendar fortnight in earlier years (+/- tol periods),
    take the MEDIAN and a MAD-scaled sigma over those observations,
    score the target as z = (value - median) / sigma.

Median/MAD rather than mean/sd because a single missed cloud in the baseline
years would otherwise drag the mean and inflate sigma, quietly suppressing real
change. A sigma floor stops near-constant pixels (water, tarmac) producing
enormous z from trivial wobble.

Connected components use 8-connectivity (Queen), matching the paper's spatial
filter.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage

from geo_utils import M_PER_DEG_LAT

PERIODS_PER_YEAR = 24

# Detection policy. Unvalidated starting values -- see MODEL_STRUCTURE.txt.
DEFAULTS = dict(
    k_sigma=2.5,          # departure from the seasonal norm
    min_delta=0.10,       # and a minimum absolute fraction move, so a very
                          # stable pixel cannot qualify on noise alone
    sigma_floor=0.04,
    baseline_years=(1, 2, 3),
    baseline_tol=1,       # +/- periods around the calendar slot
    min_baseline_obs=3,
    min_component_px=6,   # the paper's spatial filter, kept at 6 for S2 too
)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

def seasonal_baseline(by_ord: dict[int, np.ndarray], target: int, cfg,
                      robust: bool = True) -> tuple:
    """(centre, scale, n_obs) per pixel from the same season in earlier years.

    robust=True  -> median and MAD-scaled sigma  (the default, and what the
                    detector uses: one missed cloud in a baseline year would
                    otherwise drag the mean and inflate sd, suppressing real
                    change exactly where the history is worst)
    robust=False -> mean and standard deviation  (the RAW score, computed
                    alongside so evaluation can report both and the choice can
                    be audited rather than asserted)
    """
    slots = []
    for y in cfg["baseline_years"]:
        c = target - PERIODS_PER_YEAR * y
        for d in range(-cfg["baseline_tol"], cfg["baseline_tol"] + 1):
            if (c + d) in by_ord:
                slots.append(by_ord[c + d])
    if len(slots) < cfg["min_baseline_obs"]:
        return None, None, 0
    A = np.stack(slots)                                   # (T,H,W)
    # A pixel never clear in ANY baseline year is all-NaN; nanmedian warns and
    # returns NaN, which is the right answer -- change_mask requires
    # n_obs >= min_baseline_obs, so those pixels are excluded rather than
    # guessed at. Suppress the warning, not the behaviour.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if robust:
            centre = np.nanmedian(A, axis=0)
            mad = np.nanmedian(np.abs(A - centre), axis=0)
            scale = np.maximum(1.4826 * mad, cfg["sigma_floor"])
        else:
            centre = np.nanmean(A, axis=0)
            scale = np.maximum(np.nanstd(A, axis=0), cfg["sigma_floor"])
    n = np.sum(~np.isnan(A), axis=0)
    return centre, scale, n


def anomaly_scores(by_ord, target, cfg):
    """Both scores for one fraction: {'robust': z, 'raw': z, ...}.

    Reported together during evaluation so the effect of the robust estimator
    is visible rather than assumed. The detector consumes 'robust'.
    """
    cur = by_ord.get(target)
    if cur is None:
        return None
    out = {}
    for name, rb in (("robust", True), ("raw", False)):
        c, s, n = seasonal_baseline(by_ord, target, cfg, robust=rb)
        if c is None:
            out[name] = None
            continue
        with np.errstate(all="ignore"):
            out[name] = {"z": (cur - c) / s, "centre": c, "scale": s, "n_obs": n}
    return out


def change_mask(cur, med, sigma, n_obs, cfg, direction="up"):
    """Pixels departing from their own seasonal norm, in `direction`."""
    if med is None:
        return np.zeros_like(cur, bool)
    d = cur - med
    z = d / sigma
    ok = (~np.isnan(cur)) & (~np.isnan(med)) & (n_obs >= cfg["min_baseline_obs"])
    if direction == "up":
        return ok & (z >= cfg["k_sigma"]) & (d >= cfg["min_delta"])
    return ok & (z <= -cfg["k_sigma"]) & (d <= -cfg["min_delta"])


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------

_Q8 = np.ones((3, 3), int)          # Queen's rule


def components(mask: np.ndarray, min_px: int):
    lab, n = ndimage.label(mask, structure=_Q8)
    if n == 0:
        return lab, np.array([], int), np.array([], int)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1)).astype(int)
    keep = np.where(sizes >= min_px)[0] + 1
    out = np.where(np.isin(lab, keep), lab, 0)
    return out, keep, sizes[keep - 1]


def _perimeter_px(m: np.ndarray) -> int:
    """Boundary edge count (4-neighbour), in pixel-side units."""
    p = 0
    p += np.sum(m[:, 0]) + np.sum(m[:, -1]) + np.sum(m[0, :]) + np.sum(m[-1, :])
    p += np.sum(m[:, :-1] & ~m[:, 1:]) + np.sum(m[:, 1:] & ~m[:, :-1])
    p += np.sum(m[:-1, :] & ~m[1:, :]) + np.sum(m[1:, :] & ~m[:-1, :])
    return int(p)


def shape_metrics(m: np.ndarray, px_m: float) -> dict:
    """Compactness and rectangularity of one component.

    compactness    4*pi*A / P^2, 1.0 for a disc. Construction pads are blocky,
                   scattered agricultural noise is ragged and scores low.
    rectangularity A / area(minimum rotated rectangle), 1.0 for a rectangle.
                   This is what separates a building pad from a river bend.
    """
    a_px = int(m.sum())
    if a_px == 0:
        return {"compactness": None, "rectangularity": None}
    per = _perimeter_px(m)
    comp = (4 * math.pi * a_px) / (per * per) if per else None
    rect = None
    try:
        from shapely.geometry import MultiPoint
        ys, xs = np.nonzero(m)
        if len(xs) >= 3:
            mrr = MultiPoint(list(zip(xs.tolist(), ys.tolist()))).minimum_rotated_rectangle
            # +1px on each side: hull of centres understates a pixel blob
            ar = mrr.area
            rect = float(min(1.0, a_px / ar)) if ar > 0 else None
    except Exception:
        rect = None
    return {"compactness": round(comp, 4) if comp is not None else None,
            "rectangularity": round(rect, 4) if rect is not None else None}


# ---------------------------------------------------------------------------
# geometry helpers in pixel space
# ---------------------------------------------------------------------------

def pixel_to_lonlat(bbox, width, height, x, y):
    x0, y0, x1, y1 = bbox
    return (x0 + (x + 0.5) * (x1 - x0) / width,
            y1 - (y + 0.5) * (y1 - y0) / height)


def distance_m(geom_a, geom_b, lat: float) -> float | None:
    """Distance between two lon/lat geometries, in metres.

    Scales longitude by cos(lat) first, so the shapely distance is isotropic
    on the ground rather than in degrees.
    """
    if geom_a is None or geom_b is None:
        return None
    from shapely.affinity import scale
    k = max(math.cos(math.radians(lat)), 1e-6)
    a = scale(geom_a, xfact=k, yfact=1.0, origin=(0, 0))
    b = scale(geom_b, xfact=k, yfact=1.0, origin=(0, 0))
    return float(a.distance(b) * M_PER_DEG_LAT)


# ---------------------------------------------------------------------------
# zone features
# ---------------------------------------------------------------------------

@dataclass
class ZoneFeatures:
    zone: str
    zone_area_m2: float
    n_valid_px: int
    veg_loss_area_m2: float = 0.0
    new_soil_area_m2: float = 0.0
    new_high_albedo_area_m2: float = 0.0
    total_changed_area_m2: float = 0.0
    pct_zone_affected: float = 0.0
    n_components: int = 0
    largest_component_m2: float = 0.0
    largest_component_compactness: float | None = None
    largest_component_rectangularity: float | None = None
    dist_to_building_m: float | None = None
    dist_to_point_m: float | None = None
    persistence: float | None = None

    def as_dict(self):
        return asdict(self)


def zone_features(zone_mask, masks, px_area, bbox, width, height,
                  building_geom=None, point=None, cfg=None,
                  persistence_masks=None) -> ZoneFeatures:
    """Features for one analysis zone at one observation.

    `masks` holds the three boolean change layers already computed against the
    seasonal baseline: veg_loss, new_soil, new_high.
    `persistence_masks` is a list of the SAME combined-change mask at later
    clear observations; persistence is the share of them in which at least half
    the largest component's pixels are still changed.
    """
    cfg = cfg or DEFAULTS
    z = zone_mask
    f = ZoneFeatures(zone="", zone_area_m2=float(z.sum() * px_area),
                     n_valid_px=int(z.sum()))
    if z.sum() == 0:
        return f

    veg = masks["veg_loss"] & z
    soil = masks["new_soil"] & z
    high = masks["new_high"] & z
    combined = (soil | high) & z          # construction-positive evidence
    f.veg_loss_area_m2 = float(veg.sum() * px_area)
    f.new_soil_area_m2 = float(soil.sum() * px_area)
    f.new_high_albedo_area_m2 = float(high.sum() * px_area)
    f.total_changed_area_m2 = float(combined.sum() * px_area)
    f.pct_zone_affected = round(100.0 * combined.sum() / max(z.sum(), 1), 3)

    lab, keep, sizes = components(combined, cfg["min_component_px"])
    f.n_components = int(len(keep))
    if len(keep) == 0:
        return f

    big = keep[int(np.argmax(sizes))]
    m = lab == big
    f.largest_component_m2 = float(m.sum() * px_area)
    sm = shape_metrics(m, math.sqrt(px_area))
    f.largest_component_compactness = sm["compactness"]
    f.largest_component_rectangularity = sm["rectangularity"]

    ys, xs = np.nonzero(m)
    lon, lat = pixel_to_lonlat(bbox, width, height, xs.mean(), ys.mean())
    from shapely.geometry import Point as _P
    cen = _P(lon, lat)
    if building_geom is not None:
        f.dist_to_building_m = round(distance_m(cen, building_geom, lat), 1)
    if point is not None:
        f.dist_to_point_m = round(distance_m(cen, point, lat), 1)

    if persistence_masks:
        need = max(1, int(0.5 * m.sum()))
        hits = sum(1 for later in persistence_masks
                   if int((later & m).sum()) >= need)
        f.persistence = round(hits / len(persistence_masks), 3)
    return f
