"""
Tests for the positives-vs-negatives discrimination statistics.

stdlib unittest -- pytest is not installed in the geo env.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import basc_discriminate as D


class TestAuc(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertEqual(D.auc([3, 4, 5], [0, 1, 2]), 1.0)

    def test_perfect_inversion(self):
        self.assertEqual(D.auc([0, 1, 2], [3, 4, 5]), 0.0)

    def test_identical_distributions_are_half(self):
        self.assertEqual(D.auc([1, 2, 3], [1, 2, 3]), 0.5)

    def test_ties_count_as_half(self):
        self.assertEqual(D.auc([1], [1]), 0.5)

    def test_none_values_are_dropped_not_counted(self):
        # a dropped None must not silently become a 0 and fabricate separation
        self.assertEqual(D.auc([5, None], [1]), 1.0)

    def test_empty_side_returns_none(self):
        self.assertIsNone(D.auc([], [1, 2]))
        self.assertIsNone(D.auc([1, 2], [None, None]))


class TestExpectedDirections(unittest.TestCase):
    def test_every_metric_has_a_fixed_direction(self):
        for m in D.METRICS:
            self.assertIn(m, D.EXPECTED)
            self.assertIn(D.EXPECTED[m], (+1, -1))

    def test_orientation_makes_a_falling_metric_separate(self):
        """ndvi is expected to FALL at a start; orienting by EXPECTED must turn
        that into a high score, otherwise the AUC reads inverted."""
        sgn = D.EXPECTED["ndvi"]
        self.assertEqual(sgn, -1)
        pos_raw, neg_raw = [-0.4, -0.3], [0.0, 0.1]      # positives fall
        self.assertEqual(D.auc([sgn * v for v in pos_raw],
                               [sgn * v for v in neg_raw]), 1.0)


class TestDelta(unittest.TestCase):
    def _rows(self, onset_ord, onset_val, base_val):
        rows = {}
        for k in range(onset_ord - 60, onset_ord + 10):
            rows[k] = {"soil": base_val}
        for k in range(onset_ord - 1, onset_ord + 3):
            rows[k] = {"soil": onset_val}
        return rows

    def test_delta_is_onset_minus_earlier_seasons(self):
        rows = self._rows(100, 1.0, 0.0)
        self.assertAlmostEqual(D.delta(rows, 100, "soil"), 1.0)

    def test_no_signal_gives_zero(self):
        rows = self._rows(100, 0.25, 0.25)
        self.assertAlmostEqual(D.delta(rows, 100, "soil"), 0.0)

    def test_control_uses_the_same_calendar_fortnights(self):
        rows = self._rows(100, 1.0, 0.0)
        # the controls must be exactly one and two years back, not any other lag
        self.assertIn(100 - D.PERIODS_PER_YEAR, rows)
        self.assertIn(100 - 2 * D.PERIODS_PER_YEAR, rows)

    def test_missing_onset_returns_none(self):
        self.assertIsNone(D.delta({}, 100, "soil"))

    def test_missing_all_controls_returns_none(self):
        rows = {k: {"soil": 1.0} for k in range(99, 103)}   # onset only
        self.assertIsNone(D.delta(rows, 100, "soil"))

    def test_one_available_control_is_enough(self):
        rows = {k: {"soil": 1.0} for k in range(99, 103)}
        for k in range(99 - 24, 103 - 24):
            rows[k] = {"soil": 0.0}
        self.assertAlmostEqual(D.delta(rows, 100, "soil"), 1.0)


class TestLift(unittest.TestCase):
    def test_perfect_ranking_gives_max_lift(self):
        scored = [(9, True), (8, True), (1, False), (0, False)]
        prec, lift, k = D.lift_at_k(scored, 2)
        self.assertEqual(prec, 1.0)
        self.assertEqual(lift, 2.0)          # base rate 0.5
        self.assertEqual(k, 2)

    def test_random_ranking_gives_unit_lift(self):
        scored = [(1, True), (1, False), (1, True), (1, False)]
        prec, lift, _ = D.lift_at_k(scored, 2)
        self.assertEqual(lift, 1.0)

    def test_inverted_ranking_gives_zero_precision(self):
        scored = [(9, False), (8, False), (1, True), (0, True)]
        prec, lift, _ = D.lift_at_k(scored, 2)
        self.assertEqual(prec, 0.0)
        self.assertEqual(lift, 0.0)

    def test_k_larger_than_sample_is_clamped(self):
        scored = [(9, True), (1, False)]
        _, _, k = D.lift_at_k(scored, 99)
        self.assertEqual(k, 2)

    def test_none_scores_are_excluded_from_the_queue(self):
        scored = [(None, True), (9, True), (1, False)]
        prec, _, k = D.lift_at_k(scored, 2)
        self.assertEqual(k, 2)
        self.assertEqual(prec, 0.5)


class TestPairedBootstrap(unittest.TestCase):
    def test_clear_effect_excludes_zero(self):
        m, lo, hi, n = D.boot_paired([1.0, 1.1, 0.9, 1.2, 1.05])
        self.assertGreater(lo, 0)
        self.assertEqual(n, 5)

    def test_symmetric_noise_includes_zero(self):
        m, lo, hi, _ = D.boot_paired([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        self.assertLessEqual(lo, 0)
        self.assertGreaterEqual(hi, 0)

    def test_too_few_pairs_is_nan_not_a_confident_answer(self):
        m, lo, hi, n = D.boot_paired([1.0, 1.0])
        self.assertTrue(m != m)          # NaN
        self.assertEqual(n, 2)

    def test_nones_are_dropped(self):
        _, _, _, n = D.boot_paired([1.0, None, 1.0, None, 1.0])
        self.assertEqual(n, 3)


class TestClassGuards(unittest.TestCase):
    """The arithmetic completes silently on mixed-up arms; the guards must not."""

    def _write(self, d, uid, name, pseudo, start="2025-06-H1"):
        os.makedirs(d, exist_ok=True)
        site = {"uid": uid, "name": name, "state": "VA", "area_ha": 50.0,
                "n_buildings": 1, "start_period": start}
        if pseudo is not None:
            site["start_is_pseudo"] = pseudo
        rows = [{"period": "2025-06-H1", "n_valid": 100, "clear_frac": 1.0,
                 "soil": 0.1, "high_albedo": 0.1, "vegetation": 0.1,
                 "low_albedo": 0.1, "ndvi": 0.1, "ndbi": 0.1, "swir": 0.1}]
        with open(f"{d}/{uid}.json", "w") as fh:
            json.dump({"site": site, "series": rows}, fh)

    def _run(self, pos, neg):
        argv = sys.argv
        sys.argv = ["basc_discriminate.py", "--pos", pos, "--neg", neg]
        try:
            D.main()
        finally:
            sys.argv = argv

    def test_pseudo_start_in_the_positive_arm_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            self._write(f"{t}/p", "u1", "Fake Positive", pseudo=True)
            self._write(f"{t}/n", "u2", "A Negative", pseudo=True)
            with self.assertRaises(SystemExit) as e:
                self._run(f"{t}/p", f"{t}/n")
            self.assertIn("PSEUDO", str(e.exception))

    def test_unflagged_campus_in_the_negative_arm_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            self._write(f"{t}/p", "u1", "A Positive", pseudo=None)
            self._write(f"{t}/n", "u2", "Unflagged", pseudo=None)
            with self.assertRaises(SystemExit) as e:
                self._run(f"{t}/p", f"{t}/n")
            self.assertIn("start_is_pseudo", str(e.exception))

    def test_same_campus_in_both_arms_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            self._write(f"{t}/p", "same", "Both Arms", pseudo=None)
            self._write(f"{t}/n", "same", "Both Arms", pseudo=True)
            with self.assertRaises(SystemExit) as e:
                self._run(f"{t}/p", f"{t}/n")
            self.assertIn("BOTH", str(e.exception))


if __name__ == "__main__":
    unittest.main()


class TestMergeStarts(unittest.TestCase):
    """A pseudo-start must never overwrite a verified one."""

    def setUp(self):
        import basc_fetch
        from replay import ordn
        self.F, self.ordn = basc_fetch, ordn

    def test_pseudo_start_is_added_for_an_unlabelled_campus(self):
        truth = {"a": self.ordn("2025-06-H1")}
        start_of, pseudo = self.F.merge_starts(truth, {"b": "2025-01-H1"})
        self.assertEqual(start_of["b"], self.ordn("2025-01-H1"))
        self.assertEqual(list(pseudo), ["b"])

    def test_verified_start_wins_over_a_pseudo_start(self):
        truth = {"a": self.ordn("2025-06-H1")}
        start_of, pseudo = self.F.merge_starts(truth, {"a": "2020-01-H1"})
        self.assertEqual(start_of["a"], self.ordn("2025-06-H1"))
        self.assertNotIn("a", pseudo)

    def test_truth_is_not_mutated(self):
        truth = {"a": self.ordn("2025-06-H1")}
        self.F.merge_starts(truth, {"b": "2025-01-H1"})
        self.assertEqual(list(truth), ["a"])

    def test_no_starts_map_leaves_truth_alone(self):
        truth = {"a": self.ordn("2025-06-H1")}
        start_of, pseudo = self.F.merge_starts(truth, {})
        self.assertEqual(start_of, truth)
        self.assertEqual(pseudo, {})
