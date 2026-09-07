"""
Core of the Automatic Change Detector, ported to run locally against Regrid
parcel polygons instead of a point-centred square.

Faithful to supabase/functions/acd-run-detection/index.ts in the things that
decide an outcome -- the evalscript, the index maths, the half-month period
model, the two-branch threshold rule -- so a result here is comparable to a
production result. Deliberate differences:

  1. AOI is the actual parcel polygon (Statistics API `bounds.geometry`), not a
     square around the centroid. This is the point of the exercise: the mean is
     taken over the land the operator actually owns.

  2. The production edge function's AOI is 375 m on a side (its ACD_BUFFER_M is
     documented as "750m x 750m" but is passed to a function whose third
     argument is a *side*, so the box is half the intended width). Nothing here
     inherits that; polygons make the question moot.

  3. Baselines come from a local results file, not the acd_results table, so the
     lab never reads or writes production state.

Env: CDSE_CLIENT_ID / CDSE_CLIENT_SECRET (Copernicus Data Space OAuth client).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# ── Detection constants (values match the production edge function) ───────────
MAX_CLOUD_COVERAGE = 40
MIN_VALID_FRAC = 0.40
NDBI_DELTA_THRESHOLD = 0.04
NDBI_DELTA_SOFT_THRESHOLD = 0.02
SWIR_DELTA_THRESHOLD = 0.03
CHIP_PX = 512
RES_M = 10  # Sentinel-2 native for B04/B08/B11-resampled

# Statistics API refuses very large AOIs at 10 m. A parcel above this is almost
# always a bad Regrid match (the sweep log warns 87% of campuses hold a single
# parcel and matching was against unconfirmed coordinates) -- see sites.py,
# which flags them rather than silently measuring a county.
MAX_AOI_KM2 = 25.0
MIN_AOI_M2 = 5_000.0

PT = ZoneInfo("America/Los_Angeles")


# ── Half-month periods (port of periodStr / periodBounds / prevPeriodStr) ─────

def period_str(d: Optional[datetime] = None) -> str:
    """'YYYY-MM-H1'|'YYYY-MM-H2', bucketed on America/Los_Angeles wall clock.

    The timezone matters and is not incidental: the production function buckets
    in PT so the bucket agrees with the PT run label an analyst reads. A UTC
    bucket disagrees near midnight UTC.
    """
    d = d or datetime.now(timezone.utc)
    local = d.astimezone(PT)
    half = "H1" if local.day <= 15 else "H2"
    return f"{local.year:04d}-{local.month:02d}-{half}"


def period_bounds(period: str) -> tuple[str, str]:
    """[start, end) as YYYY-MM-DD, end exclusive."""
    y_s, m_s, half = period.split("-")
    y, m = int(y_s), int(m_s)
    if half == "H1":
        return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-16"
    nm, ny = (1, y + 1) if m == 12 else (m + 1, y)
    return f"{y:04d}-{m:02d}-16", f"{ny:04d}-{nm:02d}-01"


def prev_period_str(period: str) -> str:
    y_s, m_s, half = period.split("-")
    y, m = int(y_s), int(m_s)
    if half == "H2":
        return f"{y:04d}-{m:02d}-H1"
    pm, py = (12, y - 1) if m == 1 else (m - 1, y)
    return f"{py:04d}-{pm:02d}-H2"


def period_days(start: str, end: str) -> int:
    a = datetime.fromisoformat(start)
    b = datetime.fromisoformat(end)
    return round((b - a) / timedelta(days=1))


def shift_periods(period: str, back: int) -> str:
    """N half-months earlier. Used to build a training history."""
    p = period
    for _ in range(back):
        p = prev_period_str(p)
    return p


# ── CDSE auth + retrying POST (same token-cache/retry shape as production) ────

class Cdse:
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.client_id = client_id or os.environ.get("CDSE_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("CDSE_CLIENT_SECRET", "")
        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set. These are the "
                "Copernicus Data Space OAuth client credentials; in production "
                "they are Supabase secrets, not in the repo .env."
            )
        self._token: Optional[str] = None
        self._expires_at = 0.0
        self.stats_calls = 0
        self.process_calls = 0
        # One Cdse is shared across worker threads by resweep.py. Without this,
        # every thread in flight when the token expires would refresh it at
        # once -- a thundering herd on the auth endpoint, and the losers would
        # overwrite the winner's token with their own.
        self._token_lock = threading.Lock()

    def token(self) -> str:
        if self._token and time.time() < self._expires_at - 30:
            return self._token
        with self._token_lock:
            # Re-check inside the lock: another thread may have refreshed it
            # while this one was waiting, and a second refresh is wasted.
            if self._token and time.time() < self._expires_at - 30:
                return self._token
            return self._refresh_token()

    def _refresh_token(self) -> str:
        r = requests.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        if not r.ok:
            raise RuntimeError(f"CDSE auth failed: {r.status_code} {r.text[:200]}")
        d = r.json()
        self._token = d["access_token"]
        self._expires_at = time.time() + d.get("expires_in", 600)
        return self._token

    def post(self, url: str, payload: dict, expect_json: bool, max_retries: int = 3) -> Any:
        for attempt in range(max_retries + 1):
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            if r.status_code == 401:
                self._token = None
                if attempt < max_retries:
                    continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < max_retries:
                    ra = int(r.headers.get("Retry-After") or 0)
                    time.sleep(min(max(ra, 2 ** attempt), 60))
                    continue
            if not r.ok:
                raise RuntimeError(f"CDSE {url.rsplit('/', 1)[-1]} {r.status_code}: {r.text[:300]}")
            if url == CDSE_STATS_URL:
                self.stats_calls += 1
            else:
                self.process_calls += 1
            return r.json() if expect_json else r.content
        raise RuntimeError("CDSE: max retries exceeded")


# ── Index statistics over a polygon ───────────────────────────────────────────
# Evalscript is byte-for-byte the production one. SCL 4/5/6/7 = vegetation,
# not_vegetated, water, unclassified. swir is RAW B11 reflectance, not a
# normalised index -- so SWIR_DELTA_THRESHOLD is in reflectance units.

STATS_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04","B08","B11","SCL","dataMask"] }],
    output: [
      { id: "ndbi", bands: 1, sampleType: "FLOAT32" },
      { id: "swir", bands: 1, sampleType: "FLOAT32" },
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  var valid = (s.SCL === 4 || s.SCL === 5 || s.SCL === 6 || s.SCL === 7) ? 1 : 0;
  var ndbi = (s.B11 - s.B08) / (s.B11 + s.B08 + 1e-6);
  var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
  return { ndbi: [ndbi], swir: [s.B11], ndvi: [ndvi], dataMask: [s.dataMask * valid] };
}"""


PCTL_K = [10, 25, 50, 75, 90]


@dataclass
class IndexStats:
    ndbi: float
    swir: float
    ndvi: float
    valid_frac: float
    sample_count: int
    # Per-index distribution over the parcel: {'ndbi': {'std':…, 'p10':…, …}, …}
    dist: dict = None  # type: ignore[assignment]

    def flat(self, prefix: str = "") -> dict:
        """Flatten to scalar columns for a feature table."""
        out = {
            f"{prefix}ndbi": self.ndbi, f"{prefix}swir": self.swir,
            f"{prefix}ndvi": self.ndvi, f"{prefix}valid_frac": self.valid_frac,
            f"{prefix}sample_count": self.sample_count,
        }
        for idx, d in (self.dist or {}).items():
            for k, v in d.items():
                out[f"{prefix}{idx}_{k}"] = v
        return out


def _usable(v: Any) -> bool:
    # The API returns the literal string "NaN" (not null, not a number) for an
    # interval with essentially no usable data. Confirmed in production.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _centroid_lat(geometry: dict) -> float:
    """Mean latitude of a GeoJSON Polygon/MultiPolygon ring vertex set.

    Only used to size a pixel in degrees, so the ring mean is close enough --
    over a parcel a few km across the cos(lat) factor barely moves.
    """
    lats: list[float] = []

    def walk(node: Any) -> None:
        if (isinstance(node, (list, tuple)) and len(node) >= 2
                and all(isinstance(v, (int, float)) for v in node[:2])):
            lats.append(float(node[1]))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk((geometry or {}).get("coordinates", []))
    return sum(lats) / len(lats) if lats else 0.0


def fetch_index_stats(
    cdse: Cdse,
    geometry: dict,
    start: str,
    end: str,
    res_m: int = RES_M,
) -> Optional[IndexStats]:
    """Spatial mean of NDBI/SWIR/NDVI over `geometry` for [start, end).

    `geometry` is GeoJSON (Polygon or MultiPolygon) in EPSG:4326. Passing a
    geometry rather than a bbox is the whole point: the Statistics API masks to
    the polygon, so pixels outside the parcel never enter the mean.
    """
    days = period_days(start, end)
    # resx/resy are in the units of the bounds CRS, and bounds are CRS84 --
    # i.e. DEGREES, not metres. Passing RES_M=10 straight through asked for 10
    # degrees per pixel, which Sentinel Hub clamps to a single pixel covering
    # the whole AOI: sample_count comes back as 1, valid_frac as a meaningless
    # 1.0, and every percentile collapses onto the mean. Convert at the AOI's
    # own latitude so a metre-denominated constant keeps meaning metres.
    lat = _centroid_lat(geometry)
    resx = res_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.01))
    resy = res_m / 110_540.0
    payload = {
        "input": {
            "bounds": {
                "geometry": geometry,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {"maxCloudCoverage": MAX_CLOUD_COVERAGE, "mosaickingOrder": "leastCC"},
            }],
        },
        "aggregation": {
            "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T00:00:00Z"},
            "aggregationInterval": {"of": f"P{days}D"},
            "resx": resx,
            "resy": resy,
            "evalscript": STATS_EVALSCRIPT,
        },
        # Percentiles cost nothing extra -- same response, same request. They are
        # the reason a polygon AOI is worth having: over 0.3 km2 of parcel a mean
        # buries a 40x40 m pad of fresh concrete, while p90 moves. Method B feeds
        # on these; Method A still decides on the mean alone so it stays
        # comparable to production.
        "calculations": {
            k: {"statistics": {"default": {"percentiles": {"k": PCTL_K}}}}
            for k in ("ndbi", "swir", "ndvi")
        },
    }
    data = cdse.post(CDSE_STATS_URL, payload, expect_json=True)
    intervals = data.get("data") or []
    if not intervals:
        return None
    out = intervals[0].get("outputs") or {}

    def stats_of(name: str) -> dict:
        return ((out.get(name) or {}).get("bands") or {}).get("B0", {}).get("stats", {}) or {}

    nd, sw, nv = stats_of("ndbi"), stats_of("swir"), stats_of("ndvi")
    sample = nd.get("sampleCount") or 0
    if not sample:
        return None
    total = sample + (nd.get("noDataCount") or 0)
    valid_frac = sample / total if total else 0.0
    if valid_frac < MIN_VALID_FRAC:
        return None
    if not all(_usable(s.get("mean")) for s in (nd, sw, nv)):
        return None

    dist: dict = {}
    for name, st in (("ndbi", nd), ("swir", sw), ("ndvi", nv)):
        d: dict = {}
        for key, src in (("std", "stDev"), ("min", "min"), ("max", "max")):
            if _usable(st.get(src)):
                d[key] = st[src]
        pct = st.get("percentiles") or {}
        for k in PCTL_K:
            # API keys percentiles as "10.0", "90.0"
            v = pct.get(f"{float(k)}") or pct.get(str(k))
            if _usable(v):
                d[f"p{k}"] = v
        dist[name] = d

    return IndexStats(
        ndbi=nd["mean"], swir=sw["mean"], ndvi=nv["mean"],
        valid_frac=round(valid_frac, 4), sample_count=int(sample), dist=dist,
    )


def decide_changed(ndbi_delta: float, swir_delta: float) -> bool:
    """The production rule, unchanged. One-sided: only NDBI *increases* count."""
    return (
        ndbi_delta >= NDBI_DELTA_THRESHOLD
        or (ndbi_delta >= NDBI_DELTA_SOFT_THRESHOLD and swir_delta >= SWIR_DELTA_THRESHOLD)
    )


# ── True-colour chip for the CV model ─────────────────────────────────────────

TRUECOLOR_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: ["B02","B03","B04","dataMask"], output: {bands: 4}};
}
function evaluatePixel(s) {
  var g = 2.5;
  function c(v){ return Math.max(0, Math.min(1, v * g)); }
  return [c(s.B04), c(s.B03), c(s.B02), s.dataMask];
}"""

# Multi-band stack for the CV model. True colour alone throws away the two bands
# the index method relies on (B08 NIR, B11 SWIR); a model that cannot see them
# is strictly weaker than the thresholds it is meant to beat. 6 bands, one PNG
# each would need 6 requests -- instead one TIFF with all bands in one call.
STACK_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B02","B03","B04","B08","B11","SCL","dataMask"]}],
    output: {bands: 6, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(s) {
  var valid = (s.SCL === 4 || s.SCL === 5 || s.SCL === 6 || s.SCL === 7) ? 1 : 0;
  return [s.B02, s.B03, s.B04, s.B08, s.B11, s.dataMask * valid];
}"""


def fetch_chip(
    cdse: Cdse,
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    px: int = CHIP_PX,
    kind: str = "truecolor",
    geometry: Optional[dict] = None,
) -> bytes:
    """PNG (truecolor) or float32 GeoTIFF (stack) for [start, end).

    Chips are bbox-shaped because an image has to be rectangular, but passing
    `geometry` additionally masks outside-parcel pixels to dataMask=0 so the
    feature extractor can ignore them -- keeping Method B on the same footing
    as Method A.
    """
    is_stack = kind == "stack"
    bounds: dict = {"properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}}
    bounds["bbox"] = list(bbox)
    if geometry is not None:
        bounds["geometry"] = geometry
    payload = {
        "input": {
            "bounds": bounds,
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "maxCloudCoverage": MAX_CLOUD_COVERAGE,
                    "mosaickingOrder": "leastCC",
                    "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T00:00:00Z"},
                },
            }],
        },
        "output": {
            "width": px,
            "height": px,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff" if is_stack else "image/png"},
            }],
        },
        "evalscript": STACK_EVALSCRIPT if is_stack else TRUECOLOR_EVALSCRIPT,
    }
    return cdse.post(CDSE_PROCESS_URL, payload, expect_json=False)


def stats_to_dict(s: Optional[IndexStats]) -> dict:
    return asdict(s) if s else {}
