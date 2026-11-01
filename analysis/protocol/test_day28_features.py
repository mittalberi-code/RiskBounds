import unittest

from protocol.day28_features import aggregate_day28, signed_registration_lead_time


class Day28FeatureTest(unittest.TestCase):
    def test_exact_bin_edges_and_signed_registration_convention(self):
        events = [(0, 1), (6, 2), (7, 4), (13, 8), (14, 16),
                  (20, 32), (21, 64), (27, 128), (28, 256)]
        got = aggregate_day28(events)
        expected = [3, 12, 48, 192, 256]
        for i, (lo, hi) in enumerate(((0, 6), (7, 13), (14, 20), (21, 27), (28, 28))):
            self.assertAlmostEqual(got[f"log1p_clicks_days_{lo}_{hi}"], __import__("math").log1p(expected[i]))
        self.assertAlmostEqual(got["log1p_total_clicks"], __import__("math").log1p(sum(expected)))
        self.assertEqual(got["active_days"], 9)
        self.assertEqual(got["recency_days_at_day28"], 0)
        self.assertEqual(signed_registration_lead_time(-12), 12)
        self.assertEqual(signed_registration_lead_time(3), -3)

    def test_outside_window_activity_cannot_change_day28_features(self):
        base = aggregate_day28([(0, 2), (28, 5)])
        with_late = aggregate_day28([(0, 2), (28, 5), (-1, 999), (29, 999), (57, 999)])
        self.assertEqual(base, with_late)

    def test_zero_activity_unit_has_explicit_zero_features(self):
        got = aggregate_day28([])
        self.assertEqual(got["log1p_total_clicks"], 0)
        self.assertEqual(got["active_days"], 0)
        self.assertEqual(got["recency_days_at_day28"], 0)
        self.assertEqual(got["no_activity"], 1)
        self.assertTrue(all(got[f"log1p_clicks_days_{lo}_{hi}"] == 0
                            for lo, hi in ((0, 6), (7, 13), (14, 20), (21, 27), (28, 28))))

    def test_negative_click_count_fails_closed(self):
        with self.assertRaises(ValueError):
            aggregate_day28([(1, -1)])


if __name__ == "__main__":
    unittest.main()
