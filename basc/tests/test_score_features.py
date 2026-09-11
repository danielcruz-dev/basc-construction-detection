"""Tests for the ring gate in score_features.

stdlib unittest -- pytest is not installed in the geo env.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import score_features as S
from replay import ordn, unordn

O = ordn("2024-06-H1")


def rows(pcts, area=S.MMU_M2):
    """period-ordinal -> record, with a changed area at or above the MMU for
    every nonzero pct so that only the ring ratio decides the gate."""
    return {O + i: {"pct_zone_affected": p,
                    "total_changed_area_m2": area if p > 0 else 0.0}
            for i, p in enumerate(pcts)}


class RingGate(unittest.TestCase):
    def test_quiet_ring_passes_site_change(self):
        zone = rows([0, 0, 0, 5, 5, 5, 5])
        ring = rows([0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(S.gated_first_confirmed(zone, ring, 2.0), O + 3)

    def test_regional_change_is_gated_out(self):
        zone = rows([0, 0, 0, 5, 5, 5, 5])
        ring = rows([0, 0, 0, 5, 5, 5, 5])       # ring as active as the site
        self.assertIsNone(S.gated_first_confirmed(zone, ring, 1.5))
        # ratio 1 lets an equal ring through: the gate is "not weaker than"
        self.assertEqual(S.gated_first_confirmed(zone, ring, 1.0), O + 3)

    def test_concentrated_change_passes_a_noisy_ring(self):
        zone = rows([0, 0, 0, 9, 9, 9, 9])
        ring = rows([0, 0, 0, 2, 2, 2, 2])
        self.assertEqual(S.gated_first_confirmed(zone, ring, 3.0), O + 3)
        self.assertIsNone(S.gated_first_confirmed(zone, ring, 5.0))

    def test_area_bar_still_applies(self):
        zone = rows([0, 0, 0, 5, 5, 5, 5], area=S.MMU_M2 / 2)   # below MMU
        ring = rows([0, 0, 0, 0, 0, 0, 0])
        self.assertIsNone(S.gated_first_confirmed(zone, ring, 2.0))

    def test_periods_missing_from_ring_are_skipped(self):
        zone = rows([0, 0, 0, 5, 5, 5, 5])
        ring = {k: v for k, v in rows([0] * 7).items() if k != O + 3}
        # the first changed period is unobservable in the ring, so onset is
        # dated to the next one, not invented
        self.assertEqual(S.gated_first_confirmed(zone, ring, 2.0), O + 4)

    def test_ring_diff_rows(self):
        zone = rows([1, 4]); ring = rows([1, 1])
        d = S.ring_diff_rows(zone, ring)
        self.assertEqual([d[k]["ring_diff_pct"] for k in sorted(d)], [0, 3])


if __name__ == "__main__":
    unittest.main()
