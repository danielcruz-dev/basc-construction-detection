"""
Tests for the revised detector: causal detection, tiled large parcels,
multi-evidence foundations, abstention, and the strict/relaxed plan policy.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys
import unittest

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import change
import detect
import geo_utils as G
import stage
import tiling
from aoi import resolve_aoi
from siteplan_meta import from_layer, load_docs
from truth import StartInterval, bracket_from_detections


class FakeSite:
    def __init__(self, uid="uid-1", lat=39.0, lon=-77.5, geometry=None,
                 aoi_source="parcel", coord_confirmed=True, name="Test"):
        self.unit_uid, self.unit_name = uid, name
        self.lat, self.lon = lat, lon
        self.geometry, self.aoi_source = geometry, aoi_source
        self.coord_confirmed = coord_confirmed


def plan_layer(uid, lat, lon, **kw):
    p = G.ground_square(lat, lon, 150)
    layer = {"key": kw.get("key", "k1"), "campus_uid": uid, "name": "plan",
             "fit": {"residual_m": kw.get("residual_m", 1.5)},
             "features": [{"ring": list(p.exterior.coords)}]}
    layer.update({k: v for k, v in kw.items() if k not in ("residual_m", "key")})
    return layer


def obs(period, clear=True, changed=False):
    return {"period": period, "clear": clear, "changed": changed}


# ---------------------------------------------------------------------------
# 1. online predictions never access future observations
# ---------------------------------------------------------------------------

class TestCausality(unittest.TestCase):

    def test_online_call_never_uses_the_future(self):
        """The call at index i must be identical whether or not the series
        continues past i. Anything else is a look-ahead."""
        full = [obs(f"p{i}", changed=(i >= 5)) for i in range(12)]
        for i in range(len(full)):
            prefix = detect.run_causal(full[:i + 1])
            whole = detect.run_causal(full)
            self.assertEqual(prefix[i].as_dict(), whole[i].as_dict(),
                             f"call at {i} changed when future data was added")

    def test_causal_persistence_uses_only_past(self):
        seq = [obs(f"p{i}", changed=(i >= 4)) for i in range(10)]
        for i in range(len(seq)):
            a = detect.causal_persistence(seq[:i + 1], i)
            b = detect.causal_persistence(seq, i)
            self.assertEqual(a, b)

    def test_cloud_holds_the_run_rather_than_breaking_it(self):
        seq = [obs("p0"), obs("p1"), obs("p2"),
               obs("p3", changed=True), obs("p4", clear=False),
               obs("p5", changed=True)]
        d = detect.run_causal(seq)
        self.assertEqual(d[5].state, detect.CONFIRMED)
        self.assertEqual(d[4].state, detect.INSUFFICIENT)
        self.assertIn("no clear observation", d[4].notes[0])


# ---------------------------------------------------------------------------
# 2. provisional detections work at the end of a series
# ---------------------------------------------------------------------------

class TestProvisional(unittest.TestCase):

    def test_last_period_is_never_none_for_lack_of_future(self):
        seq = [obs(f"p{i}") for i in range(4)] + [obs("p4", changed=True)]
        d = detect.run_causal(seq)
        last = detect.latest_call(d)
        self.assertIsNotNone(last)
        self.assertEqual(last.state, detect.PROVISIONAL)
        self.assertEqual(last.provisional_period, "p4")
        self.assertIsNone(last.confirmed_period)
        self.assertIn("confirmable later", " ".join(last.notes))

    def test_provisional_upgrades_to_confirmed_when_imagery_arrives(self):
        base = [obs(f"p{i}") for i in range(4)] + [obs("p4", changed=True)]
        self.assertEqual(detect.run_causal(base)[-1].state, detect.PROVISIONAL)
        later = base + [obs("p5", changed=True)]
        d = detect.run_causal(later)
        self.assertEqual(d[-1].state, detect.CONFIRMED)
        # onset is dated to the FIRST sighting, not the confirming one
        self.assertEqual(d[-1].confirmed_period, "p4")

    def test_transient_does_not_confirm_and_is_withdrawn(self):
        seq = ([obs(f"p{i}") for i in range(4)] + [obs("p4", changed=True)]
               + [obs(f"p{i}") for i in range(5, 9)])
        d = detect.run_causal(seq)
        self.assertEqual(d[4].state, detect.PROVISIONAL)
        self.assertEqual(d[-1].state, detect.NO_CHANGE)
        self.assertIsNone(d[-1].provisional_period)

    def test_insufficient_evidence_is_not_no_change(self):
        d = detect.run_causal([obs("p0", clear=False), obs("p1", clear=False)])
        self.assertEqual(d[-1].state, detect.INSUFFICIENT)
        self.assertNotEqual(d[-1].state, detect.NO_CHANGE)


# ---------------------------------------------------------------------------
# 3. large parcels retain localized changes, through tiling
# ---------------------------------------------------------------------------

class TestTiling(unittest.TestCase):

    def test_component_on_a_tile_seam_is_merged_not_split(self):
        """The failure tiling could introduce: a pad straddling a seam split
        into two sub-threshold fragments and dropped."""
        m = np.zeros((128, 128), bool)
        m[50:70, 56:76] = True                 # 400 px straddling the x=64 seam
        lab, recs = tiling.tiled_components(m, tile_px=64, overlap_px=16, min_px=6)
        self.assertEqual(len(recs), 1, "seam split the component")
        self.assertEqual(recs[0]["n_px"], 400)

    def test_two_separate_disturbances_stay_separate(self):
        m = np.zeros((128, 128), bool)
        m[10:20, 10:20] = True
        m[100:110, 100:110] = True
        lab, recs = tiling.tiled_components(m, tile_px=64, overlap_px=16, min_px=6)
        self.assertEqual(len(recs), 2)

    def test_localized_pad_survives_in_a_very_large_parcel(self):
        H = W = 400                            # 400x400 px @10 m = 1,600 ha
        px_area = 100.0
        m = np.zeros((H, W), bool)
        m[200:222, 200:218] = True             # 396 px = 39,600 m2
        lab, recs = tiling.tiled_components(m, tile_px=64, overlap_px=16, min_px=6)
        self.assertEqual(len(recs), 1)
        area = recs[0]["n_px"] * px_area
        self.assertAlmostEqual(area, 39600, delta=100)
        # the same event as a share of the parcel mean is vanishing
        self.assertLess(area / (H * W * px_area), 0.0025)

    def test_ranking_prefers_compact_near_disturbances(self):
        H = W = 200
        m = np.zeros((H, W), bool)
        m[100:120, 100:120] = True                     # compact block, 400 px
        for i in range(40):                            # ragged scatter, similar area
            m[20 + (i % 7), 20 + i * 4 % 150] = True
            m[21 + (i % 7), 20 + i * 4 % 150] = True
        lab, recs = tiling.tiled_components(m, 64, 16, min_px=6)
        ranked = tiling.rank_disturbances(lab, recs, 100.0)
        self.assertGreaterEqual(len(ranked), 1)
        top = ranked[0]
        self.assertGreater(top["rectangularity"], 0.9)

    def test_resolver_flags_large_parcel_for_tiling(self):
        big = G.ground_square(39.0, -77.5, 2000)       # 400 ha
        s = FakeSite(lat=39.0, lon=-77.5, geometry=big, aoi_source="parcel")
        r = resolve_aoi(s, "2025-06-01", [])
        self.assertEqual(r["source"], "parcel")
        self.assertTrue(r["needs_tiling"])


# ---------------------------------------------------------------------------
# 4 & 5. foundations need multiple evidence; 10 vs 15 can abstain
# ---------------------------------------------------------------------------

def feats(**kw):
    d = dict(zone="development", zone_area_m2=1e6, n_valid_px=10000,
             veg_loss_area_m2=0.0, new_soil_area_m2=0.0,
             new_high_albedo_area_m2=0.0, total_changed_area_m2=0.0,
             pct_zone_affected=0.0, n_components=1, largest_component_m2=0.0,
             largest_component_compactness=0.6,
             largest_component_rectangularity=0.8,
             dist_to_building_m=50.0, persistence=0.9,
             prior_veg_loss_m2=30000.0, visual_confirmed=False)
    d.update(kw)
    return d


class TestFoundationEvidence(unittest.TestCase):

    def test_15_cannot_come_from_high_albedo_alone(self):
        """Bright pixels with nothing else must never be called foundation."""
        c = stage.classify(feats(
            new_high_albedo_area_m2=40000, new_soil_area_m2=0.0,
            total_changed_area_m2=40000, largest_component_m2=40000,
            persistence=None, largest_component_rectangularity=None,
            dist_to_building_m=None, prior_veg_loss_m2=None))
        self.assertNotEqual(c.stage_pct, 15)

    def test_bright_without_worked_ground_is_not_foundation(self):
        c = stage.classify(feats(
            new_high_albedo_area_m2=30000, new_soil_area_m2=1000,
            total_changed_area_m2=31000, largest_component_m2=30000))
        self.assertNotEqual(c.stage_pct, 15)

    def test_15_requires_the_configured_number_of_signals(self):
        full = feats(new_high_albedo_area_m2=6000, new_soil_area_m2=30000,
                     total_changed_area_m2=36000, largest_component_m2=30000)
        self.assertEqual(stage.classify(full).stage_pct, 15)
        strict = stage.classify(full, cfg={"foundation_min_evidence": 7})
        self.assertNotEqual(strict.stage_pct, 15)

    def test_partial_foundation_evidence_abstains_to_10_15(self):
        c = stage.classify(feats(
            new_high_albedo_area_m2=6000, new_soil_area_m2=30000,
            total_changed_area_m2=36000, largest_component_m2=30000,
            persistence=0.55, largest_component_rectangularity=0.35,
            dist_to_building_m=None, prior_veg_loss_m2=None))
        self.assertIsNone(c.stage_pct)
        self.assertEqual(c.stage_group, "10-15")
        self.assertIn("abstain", " ".join(c.reasons).lower())

    def test_abstention_lowers_confidence_below_a_committed_call(self):
        committed = stage.classify(feats(
            new_high_albedo_area_m2=6000, new_soil_area_m2=30000,
            total_changed_area_m2=36000, largest_component_m2=30000))
        abstained = stage.classify(feats(
            new_high_albedo_area_m2=6000, new_soil_area_m2=30000,
            total_changed_area_m2=36000, largest_component_m2=30000,
            persistence=0.55, largest_component_rectangularity=0.35,
            dist_to_building_m=None, prior_veg_loss_m2=None))
        self.assertLess(abstained.confidence, committed.confidence)

    def test_evidence_is_itemised_in_the_record(self):
        c = stage.classify(feats(new_high_albedo_area_m2=6000,
                                 new_soil_area_m2=30000,
                                 total_changed_area_m2=36000,
                                 largest_component_m2=30000))
        ev = c.evidence["foundation_evidence"]
        for k in ("bright_area", "connected_area", "persistence", "geometry",
                  "footprint_proximity", "prior_veg_loss", "visual"):
            self.assertIn(k, ev)


# ---------------------------------------------------------------------------
# 6. strict vs relaxed site-plan policy
# ---------------------------------------------------------------------------

class TestPlanPolicy(unittest.TestCase):

    def setUp(self):
        self.lat, self.lon = 39.0, -77.5
        self.site = FakeSite(lat=self.lat, lon=self.lon,
                             geometry=G.ground_square(self.lat, self.lon, 900))

    def test_strict_uses_public_date_not_document_date(self):
        layer = plan_layer("uid-1", self.lat, self.lon,
                           document_date="2024-02-01", public_date="2025-09-01")
        r = resolve_aoi(self.site, "2025-01-01", [layer])
        self.assertEqual(r["source"], "parcel")           # not yet public
        self.assertIn("strict", r["fallback_reason"])

    def test_relaxed_is_a_separately_labelled_sensitivity(self):
        layer = plan_layer("uid-1", self.lat, self.lon,
                           document_date="2024-02-01", public_date="2025-09-01")
        r = resolve_aoi(self.site, "2025-01-01", [layer], config={"policy": "relaxed"})
        self.assertEqual(r["source"], "siteplan")
        self.assertEqual(r["siteplan_policy"], "relaxed")

    def test_as_built_never_usable_under_either_policy(self):
        layer = plan_layer("uid-1", self.lat, self.lon,
                           document_date="2020-01-01", plan_type="as_built")
        for pol in ("strict", "relaxed"):
            r = resolve_aoi(self.site, "2025-01-01", [layer], config={"policy": pol})
            self.assertEqual(r["source"], "parcel", f"as_built leaked under {pol}")

    def test_no_future_plan_in_strict_replay(self):
        """The requirement, stated directly: walk time forward under strict
        policy and assert no plan is used before its usable_from."""
        layer = plan_layer("uid-1", self.lat, self.lon,
                           document_date="2024-02-01", public_date="2025-06-10")
        gate = dt.date(2025, 6, 10)
        d = dt.date(2024, 1, 1)
        while d <= dt.date(2026, 1, 1):
            r = resolve_aoi(self.site, d, [layer])
            if d < gate:
                self.assertEqual(r["source"], "parcel", f"leaked at {d}")
                self.assertIsNone(r["siteplan_date"])
            else:
                self.assertEqual(r["source"], "siteplan")
            d += dt.timedelta(days=10)

    def test_usable_from_is_the_latest_known_date(self):
        doc = from_layer(plan_layer("u", 39.0, -77.5, document_date="2024-01-01",
                                    revision_date="2024-06-01",
                                    public_date="2024-03-01"))
        self.assertEqual(doc.usable_from, dt.date(2024, 6, 1))
        self.assertEqual(doc.document_from, dt.date(2024, 1, 1))

    def test_missing_public_date_is_flagged_as_optimistic(self):
        doc = from_layer(plan_layer("u", 39.0, -77.5, sheet_date="2024-01-01"))
        self.assertIn("optimistic", doc.date_provenance)


# ---------------------------------------------------------------------------
# ground truth as an interval
# ---------------------------------------------------------------------------

class TestStartInterval(unittest.TestCase):

    def _pb(self, p):
        i = int(p[1:])
        return (f"2025-01-{i+1:02d}", f"2025-01-{i+2:02d}")

    def test_interval_brackets_the_first_confirmed_change(self):
        seq = [obs(f"p{i}") for i in range(5)] + \
              [obs(f"p{i}", changed=True) for i in range(5, 8)]
        d = detect.run_causal(seq)
        si = bracket_from_detections("uid", d, self._pb,
                                     reported_date="2025-01-06",
                                     reported_period="p5")
        self.assertTrue(si.is_bounded)
        self.assertEqual(si.first_changed_period, "p5")
        self.assertEqual(si.last_unchanged_period, "p4")

    def test_reported_date_is_preserved_separately(self):
        seq = [obs(f"p{i}") for i in range(5)] + [obs("p5", changed=True),
                                                  obs("p6", changed=True)]
        si = bracket_from_detections("uid", detect.run_causal(seq), self._pb,
                                     reported_date="2024-11-01",
                                     reported_source="event_date",
                                     reported_confidence=0.2)
        self.assertEqual(si.reported_date, "2024-11-01")
        self.assertEqual(si.reported_source, "event_date")
        self.assertEqual(si.reported_confidence, 0.2)
        # the interval is NOT overwritten by the reported date
        self.assertEqual(si.first_changed_period, "p5")

    def test_open_interval_when_nothing_detected(self):
        si = bracket_from_detections("uid", detect.run_causal(
            [obs(f"p{i}") for i in range(6)]), self._pb)
        self.assertFalse(si.is_bounded)
        self.assertIn("open at the right", " ".join(si.notes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
