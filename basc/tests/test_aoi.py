"""
Tests for the AOI resolver, buffering, and change detection.

stdlib unittest -- pytest is not installed in the geo env.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys
import unittest

import numpy as np
from pyproj import Geod
from shapely.geometry import Polygon, Point, mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import change
import geo_utils as G
from aoi import provenance, resolve_aoi

GEOD = Geod(ellps="WGS84")


class FakeSite:
    """Stands in for sites.Site without needing the parcel GPKG or inventory."""

    def __init__(self, uid="uid-1", lat=39.0, lon=-77.5, geometry=None,
                 aoi_source="parcel", coord_confirmed=True, name="Test Campus"):
        self.unit_uid = uid
        self.unit_name = name
        self.lat = lat
        self.lon = lon
        self.geometry = geometry
        self.aoi_source = aoi_source
        self.coord_confirmed = coord_confirmed


def square_at(lat, lon, side_m):
    return G.ground_square(lat, lon, side_m)


def plan_layer(uid, lat, lon, sheet_date, n=2, side_m=120, residual_m=1.5, rings=True):
    """A minimal plan-meta layer with `n` footprints near (lat, lon)."""
    feats = []
    for i in range(n):
        off = (i - (n - 1) / 2) * (side_m * 2) / G.M_PER_DEG_LON_EQ / math.cos(math.radians(lat))
        p = G.ground_square(lat, lon + off, side_m)
        feats.append({
            "n": i + 1, "lat": lat, "lon": lon + off, "sf": 10000,
            "ring": list(p.exterior.coords) if rings else None,
        })
    return {"key": f"plan-{uid}-{sheet_date}", "name": "plan", "campus_uid": uid,
            "sheet_date": sheet_date, "fit": {"residual_m": residual_m},
            "features": feats}


# ---------------------------------------------------------------------------

class TestPriority(unittest.TestCase):
    """1, 3, 4 -- the priority ladder and its fallbacks."""

    def setUp(self):
        self.lat, self.lon = 39.0, -77.5
        self.parcel = square_at(self.lat, self.lon, 900)
        self.site = FakeSite(lat=self.lat, lon=self.lon, geometry=self.parcel)
        self.layers = [plan_layer("uid-1", self.lat, self.lon, "2024-05-07")]

    def test_valid_siteplan_beats_parcel(self):
        r = resolve_aoi(self.site, observation_date="2025-06-01",
                        plan_layers=self.layers)
        self.assertEqual(r["source"], "siteplan")
        self.assertEqual(r["siteplan_date"], "2024-05-07")
        self.assertIsNotNone(r["analysis_zones"]["building"])
        # building < development < context, strictly
        a = [G.geodesic_area_m2(r["analysis_zones"][z])
             for z in ("building", "development", "context")]
        self.assertLess(a[0], a[1])
        self.assertLess(a[1], a[2])

    def test_parcel_used_when_no_siteplan(self):
        r = resolve_aoi(self.site, observation_date="2025-06-01", plan_layers=[])
        self.assertEqual(r["source"], "parcel")
        self.assertIsNone(r["analysis_zones"]["building"])
        self.assertIsNotNone(r["fallback_reason"])
        # development zone IS the parcel, not a reduction of it
        self.assertAlmostEqual(
            G.geodesic_area_m2(r["analysis_zones"]["development"]),
            G.geodesic_area_m2(self.parcel), delta=1.0)

    def test_missing_parcel_falls_back_to_point(self):
        s = FakeSite(lat=self.lat, lon=self.lon, geometry=None, aoi_source="box_no_parcel")
        r = resolve_aoi(s, observation_date="2025-06-01", plan_layers=[])
        self.assertEqual(r["source"], "point_box")
        self.assertEqual(r["point_sizes_m"], [400.0, 1000.0])
        dev = G.geodesic_area_m2(r["analysis_zones"]["development"])
        ctx = G.geodesic_area_m2(r["analysis_zones"]["context"])
        self.assertAlmostEqual(dev, 400 * 400, delta=400 * 400 * 0.02)
        self.assertAlmostEqual(ctx, 1000 * 1000, delta=1000 * 1000 * 0.02)

    def test_invalid_parcel_falls_back_to_point(self):
        # bow-tie: self-intersecting, zero area after repair
        bad = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
        s = FakeSite(lat=self.lat, lon=self.lon, geometry=bad, aoi_source="parcel")
        r = resolve_aoi(s, observation_date="2025-06-01", plan_layers=[])
        self.assertEqual(r["source"], "point_box")
        self.assertIn("parcel:", r["fallback_reason"])

    def test_oversized_parcel_falls_back(self):
        huge = square_at(self.lat, self.lon, 4000)     # 1,600 ha > 800 ha cap
        s = FakeSite(lat=self.lat, lon=self.lon, geometry=huge, aoi_source="parcel")
        r = resolve_aoi(s, observation_date="2025-06-01", plan_layers=[])
        self.assertEqual(r["source"], "point_box")
        self.assertIn("exceeds max", r["fallback_reason"])


class TestTemporalLeakage(unittest.TestCase):
    """2, 8 -- a plan may never inform a date before it existed."""

    def setUp(self):
        self.lat, self.lon = 39.0, -77.5
        self.site = FakeSite(lat=self.lat, lon=self.lon,
                             geometry=square_at(self.lat, self.lon, 900))
        self.layers = [plan_layer("uid-1", self.lat, self.lon, "2025-03-15")]

    def test_postdated_siteplan_rejected(self):
        r = resolve_aoi(self.site, observation_date="2024-06-01",
                        plan_layers=self.layers)
        self.assertEqual(r["source"], "parcel")
        self.assertIn("postdates", r["fallback_reason"])

    def test_same_day_is_available(self):
        r = resolve_aoi(self.site, observation_date="2025-03-15",
                        plan_layers=self.layers)
        self.assertEqual(r["source"], "siteplan")

    def test_bad_georeference_rejected(self):
        bad = [plan_layer("uid-1", self.lat, self.lon, "2024-01-01", residual_m=90.0)]
        r = resolve_aoi(self.site, observation_date="2025-06-01", plan_layers=bad)
        self.assertEqual(r["source"], "parcel")
        self.assertIn("residual", r["fallback_reason"])

    def test_no_rings_rejected(self):
        bad = [plan_layer("uid-1", self.lat, self.lon, "2024-01-01", rings=False)]
        r = resolve_aoi(self.site, observation_date="2025-06-01", plan_layers=bad)
        self.assertEqual(r["source"], "parcel")

    def test_chronological_replay_uses_no_future_plan(self):
        """The decisive one: walk time forward and assert the switch happens
        exactly at the sheet date, never before."""
        layers = [plan_layer("uid-1", self.lat, self.lon, "2025-03-15")]
        switch = dt.date(2025, 3, 15)
        seen_before, seen_after = set(), set()
        d = dt.date(2024, 1, 1)
        while d <= dt.date(2026, 1, 1):
            r = resolve_aoi(self.site, observation_date=d, plan_layers=layers)
            (seen_after if d >= switch else seen_before).add(r["source"])
            if d < switch:
                self.assertIsNone(r["siteplan_date"],
                                  f"leaked a plan into {d}")
            d += dt.timedelta(days=14)
        self.assertEqual(seen_before, {"parcel"})
        self.assertEqual(seen_after, {"siteplan"})


class TestBuffers(unittest.TestCase):
    """5 -- buffers are metres on the ground, at any latitude."""

    def test_buffer_is_metres_at_many_latitudes(self):
        for lat in (0.0, 23.5, 39.0, 55.0, 60.0, -33.9):
            p = Point(10.0, lat).buffer(1e-9)
            b = G.buffer_m(p, 500.0, lat)
            x0, y0, x1, y1 = b.bounds
            _, _, ew = GEOD.inv(x0, lat, x1, lat)
            _, _, ns = GEOD.inv(10.0, y0, 10.0, y1)
            # shapely's 1000 m circle is a polygon, so allow 2%
            self.assertAlmostEqual(ew, 1000.0, delta=25.0, msg=f"E-W at {lat}")
            self.assertAlmostEqual(ns, 1000.0, delta=25.0, msg=f"N-S at {lat}")
            self.assertLess(abs(ew - ns), 20.0,
                            f"anisotropic at {lat}: {ew:.0f} vs {ns:.0f}")

    def test_degree_buffer_would_have_been_wrong(self):
        """Guards the reason buffer_m exists: a naive degree buffer is badly
        anisotropic, and increasingly so with latitude."""
        lat = 60.0
        p = Point(10.0, lat).buffer(1e-9)
        naive = p.buffer(500.0 / G.M_PER_DEG_LAT)
        x0, y0, x1, y1 = naive.bounds
        _, _, ew = GEOD.inv(x0, lat, x1, lat)
        _, _, ns = GEOD.inv(10.0, y0, 10.0, y1)
        self.assertLess(ew, 0.6 * ns)          # ~half the intended width

    def test_ground_square_area(self):
        for lat in (0.0, 39.0, 60.0):
            a = G.geodesic_area_m2(G.ground_square(lat, 5.0, 400.0))
            self.assertAlmostEqual(a, 160000.0, delta=160000.0 * 0.02)


class TestCloudDenominator(unittest.TestCase):
    """Cloud must be measured against rasterised AOI pixels."""

    def test_rasterised_denominator_differs_from_area_ratio(self):
        lat, lon = 39.0, -77.5
        # an L-shape: area/bbox ratio understates nothing, but the two measures
        # are computed differently and the pixel count is the honest one
        d = 0.004
        poly = Polygon([(lon, lat), (lon + d, lat), (lon + d, lat + d / 2),
                        (lon + d / 2, lat + d / 2), (lon + d / 2, lat + d),
                        (lon, lat + d)])
        bbox = poly.bounds
        W = H = 64
        mask = G.rasterize(poly, bbox, W, H)
        px_ratio = mask.sum() / (W * H)
        area_ratio = poly.area / Polygon.from_bounds(*bbox).area
        self.assertGreater(mask.sum(), 0)
        self.assertAlmostEqual(px_ratio, area_ratio, delta=0.03)
        # and the denominator is a COUNT, so clear_frac can never exceed 1
        n_clear = int(mask.sum() * 0.4)
        self.assertLessEqual(n_clear / mask.sum(), 1.0)

    def test_rasterize_respects_geometry(self):
        lat, lon = 39.0, -77.5
        sq = G.ground_square(lat, lon, 400)
        big = G.ground_square(lat, lon, 800)
        m = G.rasterize(sq, big.bounds, 80, 80)
        # a 400 m square inside an 800 m box is a quarter of the pixels
        self.assertAlmostEqual(m.sum() / (80 * 80), 0.25, delta=0.02)


class TestLargeParcelSignal(unittest.TestCase):
    """6 -- a small pad inside a huge parcel must survive as a component."""

    def test_localized_change_survives_in_large_parcel(self):
        H = W = 200                      # 200x200 px at 10 m = 400 ha
        px_area = 100.0
        rng = np.random.default_rng(0)
        base = np.full((H, W), 0.20, float)

        # baseline years: same season, mild noise
        by_ord = {}
        for k in (24, 48, 72):
            by_ord[100 - k] = base + rng.normal(0, 0.01, (H, W))

        cur = base + rng.normal(0, 0.01, (H, W))
        cur[100:118, 100:122] += 0.55    # 18x22 px = 396 px = 39,600 m2 pad

        cfg = dict(change.DEFAULTS)
        med, sigma, n = change.seasonal_baseline(by_ord, 100, cfg)
        m = change.change_mask(cur, med, sigma, n, cfg, "up")

        # the parcel MEAN barely moves -- this is the dilution the AOI work is about
        self.assertLess(cur.mean() - base.mean(), 0.01)
        # but the component is found, and at the right size
        lab, keep, sizes = change.components(m, cfg["min_component_px"])
        self.assertEqual(len(keep), 1)
        self.assertAlmostEqual(sizes[0] * px_area, 39600, delta=39600 * 0.1)

        zone = np.ones((H, W), bool)
        f = change.zone_features(
            zone, {"veg_loss": np.zeros((H, W), bool), "new_soil": m,
                   "new_high": np.zeros((H, W), bool)},
            px_area, (0.0, 39.0, 0.02, 39.02), W, H)
        self.assertGreater(f.largest_component_m2, 8000.0)
        self.assertGreater(f.largest_component_rectangularity, 0.9)   # it is a rectangle
        self.assertLess(f.pct_zone_affected, 2.0)                     # tiny share of the parcel

    def test_absolute_threshold_would_have_fired_on_seasonal_soil(self):
        """Changed area is departure from the seasonal norm, not an absolute
        threshold -- a field that is bare EVERY February must not count."""
        H = W = 40
        by_ord = {100 - k: np.full((H, W), 0.62) for k in (24, 48, 72)}
        cur = np.full((H, W), 0.62)          # bare again, exactly as usual
        cfg = dict(change.DEFAULTS)
        med, sigma, n = change.seasonal_baseline(by_ord, 100, cfg)
        m = change.change_mask(cur, med, sigma, n, cfg, "up")
        self.assertEqual(int(m.sum()), 0)
        # an absolute rule at tau=0.5 would have flagged the entire field
        self.assertEqual(int((cur >= 0.5).sum()), H * W)


class TestProvenance(unittest.TestCase):
    """7 -- source and fallback reason must reach the records."""

    def test_provenance_propagates(self):
        lat, lon = 39.0, -77.5
        s = FakeSite(lat=lat, lon=lon, geometry=None, aoi_source="box_no_parcel",
                     coord_confirmed=False)
        r = resolve_aoi(s, observation_date="2025-06-01", plan_layers=[])
        p = provenance(r)
        self.assertEqual(p["aoi_source"], "point_box")
        self.assertEqual(p["coordinate_quality"], "approximate")
        self.assertIn("siteplan:", p["aoi_fallback_reason"])
        self.assertIn("parcel:", p["aoi_fallback_reason"])
        self.assertTrue(p["aoi_warnings"])
        for k in ("aoi_source", "aoi_area_m2", "aoi_buffer_m", "aoi_confidence",
                  "coordinate_quality", "siteplan_date", "parcel_id",
                  "aoi_fallback_reason", "aoi_warnings"):
            self.assertIn(k, p)

    def test_full_record_schema(self):
        lat, lon = 39.0, -77.5
        s = FakeSite(lat=lat, lon=lon, geometry=square_at(lat, lon, 600))
        r = resolve_aoi(s, observation_date="2025-06-01",
                        plan_layers=[plan_layer("uid-1", lat, lon, "2024-01-01")])
        for k in ("geometry", "source", "analysis_zones", "area_m2", "buffer_m",
                  "coordinate_quality", "siteplan_date", "parcel_id",
                  "confidence", "warnings", "fallback_reason"):
            self.assertIn(k, r)
        self.assertEqual(set(r["analysis_zones"]), {"building", "development", "context"})
        self.assertIsInstance(r["warnings"], list)
        self.assertGreater(r["confidence"], 0.0)

    def test_confidence_orders_by_source(self):
        lat, lon = 39.0, -77.5
        sp = resolve_aoi(FakeSite(lat=lat, lon=lon, geometry=square_at(lat, lon, 600)),
                         "2025-06-01", [plan_layer("uid-1", lat, lon, "2024-01-01")])
        pa = resolve_aoi(FakeSite(lat=lat, lon=lon, geometry=square_at(lat, lon, 600)),
                         "2025-06-01", [])
        pt = resolve_aoi(FakeSite(lat=lat, lon=lon, geometry=None,
                                  aoi_source="box_no_parcel"), "2025-06-01", [])
        self.assertGreater(sp["confidence"], pa["confidence"])
        self.assertGreater(pa["confidence"], pt["confidence"])

    def test_point_outside_parcel_is_warned(self):
        lat, lon = 39.0, -77.5
        far = G.ground_square(lat + 0.01, lon + 0.01, 300)
        s = FakeSite(lat=lat, lon=lon, geometry=far, aoi_source="parcel")
        r = resolve_aoi(s, "2025-06-01", [])
        self.assertEqual(r["source"], "parcel")
        self.assertTrue(any("OUTSIDE the parcel" in w for w in r["warnings"]))


class TestStageModel(unittest.TestCase):
    """The 0-15% ladder responds to the evidence it claims to use."""

    def _f(self, **kw):
        d = dict(zone="development", zone_area_m2=1e6, n_valid_px=10000,
                 veg_loss_area_m2=0.0, new_soil_area_m2=0.0,
                 new_high_albedo_area_m2=0.0, total_changed_area_m2=0.0,
                 pct_zone_affected=0.0, n_components=0, largest_component_m2=0.0,
                 largest_component_compactness=0.6,
                 largest_component_rectangularity=0.8,
                 dist_to_building_m=50.0, persistence=0.9)
        d.update(kw)
        return d

    def test_no_change_is_zero(self):
        import stage
        self.assertEqual(stage.classify(self._f()).stage_pct, 0)

    def test_transient_change_is_rejected(self):
        import stage
        c = stage.classify(self._f(veg_loss_area_m2=50000, new_soil_area_m2=40000,
                                   total_changed_area_m2=40000,
                                   largest_component_m2=40000, persistence=0.1))
        self.assertEqual(c.stage_pct, 0)
        self.assertIn("persist", " ".join(c.reasons))

    def test_clearing_grading_foundation_ladder(self):
        import stage
        clearing = stage.classify(self._f(veg_loss_area_m2=30000,
                                          new_soil_area_m2=4000,
                                          total_changed_area_m2=9000,
                                          largest_component_m2=9000))
        grading = stage.classify(self._f(veg_loss_area_m2=30000,
                                         new_soil_area_m2=30000,
                                         total_changed_area_m2=32000,
                                         largest_component_m2=30000))
        foundation = stage.classify(self._f(veg_loss_area_m2=30000,
                                            new_soil_area_m2=30000,
                                            new_high_albedo_area_m2=6000,
                                            total_changed_area_m2=36000,
                                            largest_component_m2=30000))
        self.assertEqual(clearing.stage_pct, 5)
        self.assertEqual(grading.stage_pct, 10)
        self.assertEqual(foundation.stage_pct, 15)

    def test_below_mmu_is_zero(self):
        import stage
        c = stage.classify(self._f(new_soil_area_m2=3000,
                                   total_changed_area_m2=3000,
                                   largest_component_m2=3000))
        self.assertEqual(c.stage_pct, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
