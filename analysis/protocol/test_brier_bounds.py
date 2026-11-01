from fractions import Fraction
from itertools import product
import random
import unittest
import numpy as np

from protocol.brier_bounds import paired_brier_bounds


def d(r, a, y):
    return (r - y) ** 2 - (a - y) ** 2


def exhaustive(known_r, known_a, labels, unknown_r, unknown_a):
    known_sum = sum((d(r, a, y) for r, a, y in zip(known_r, known_a, labels)), Fraction())
    outcomes = []
    for state in product((-1, 0, 1), repeat=len(unknown_r)):
        numerator = known_sum
        included = 0
        for r, a, s in zip(unknown_r, unknown_a, state):
            if s == -1:
                continue
            included += 1
            numerator += d(r, a, s)
        outcomes.append(numerator / (len(labels) + included))
    return min(outcomes), max(outcomes)


class PairedBrierBoundsTest(unittest.TestCase):
    def test_sorted_prefix_bounds_match_exhaustive_ternary_assignments(self):
        rng = random.Random(20261031)
        found_interior = False
        allocations = 0
        for k in range(8):
            for _ in range(40):
                n = rng.randint(1, 8)
                kr = [Fraction(rng.randrange(101), 100) for _ in range(n)]
                ka = [Fraction(rng.randrange(101), 100) for _ in range(n)]
                labels = [rng.randrange(2) for _ in range(n)]
                ur = [Fraction(rng.randrange(101), 100) for _ in range(k)]
                ua = [Fraction(rng.randrange(101), 100) for _ in range(k)]
                result = paired_brier_bounds(kr, ka, labels, ur, ua)
                expected = exhaustive(kr, ka, labels, ur, ua)
                self.assertEqual((result.lower, result.upper), expected)
                self.assertGreaterEqual(result.lower_included_unknown, 0)
                self.assertLessEqual(result.lower_included_unknown, k)
                self.assertGreaterEqual(result.upper_included_unknown, 0)
                self.assertLessEqual(result.upper_included_unknown, k)
                found_interior |= 0 < result.lower_included_unknown < k or 0 < result.upper_included_unknown < k
                allocations += 3**k
        self.assertTrue(found_interior)
        self.assertGreater(allocations, 100000)

    def test_m_zero_and_positive_sign_convention(self):
        result = paired_brier_bounds([Fraction(4, 5)], [Fraction(1, 5)], [1], [], [])
        self.assertEqual(result.lower, Fraction(-3, 5))
        self.assertEqual(result.upper, Fraction(-3, 5))
        self.assertEqual(result.lower_included_unknown, 0)
        self.assertEqual(result.upper_included_unknown, 0)

    def test_equal_predictors_give_zero_for_every_assignment(self):
        result = paired_brier_bounds([Fraction(2, 5), Fraction(9, 10)], [Fraction(2, 5), Fraction(9, 10)], [0, 1],
                                     [Fraction(1, 10)] * 4, [Fraction(1, 10)] * 4)
        self.assertEqual((result.lower, result.upper), (Fraction(0), Fraction(0)))

    def test_model_swap_reverses_endpoints(self):
        r = [Fraction(1, 5), Fraction(9, 10)]
        a = [Fraction(7, 10), Fraction(3, 10)]
        y = [0, 1]
        ur = [Fraction(1, 4), Fraction(3, 4)]
        ua = [Fraction(4, 5), Fraction(1, 5)]
        forward = paired_brier_bounds(r, a, y, ur, ua)
        reverse = paired_brier_bounds(a, r, y, ua, ur)
        self.assertEqual((reverse.lower, reverse.upper), (-forward.upper, -forward.lower))

    def test_invalid_or_missing_uncertain_probability_fails_closed(self):
        for bad in (None, float("nan"), float("inf"), -0.01, 1.01, True, False,
                    "0.5", "1", np.bool_(True), np.float64("nan"), np.float32("inf"),
                    np.float64(-0.1), np.float32(1.1)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    paired_brier_bounds([0.5], [0.4], [1], [bad], [0.3])

    def test_supported_numpy_numeric_scalars_are_accepted(self):
        result = paired_brier_bounds([np.float32(.5)], [np.float64(.4)], [np.int64(1)],
                                     [np.float64(.3)], [np.float32(.2)])
        self.assertEqual(result.known_records, 1)
        self.assertEqual(result.uncertain_records, 1)

    def test_missing_uncertain_predictions_do_not_fall_back(self):
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            paired_brier_bounds([0.5], [0.4], [1], [0.5], [0.4], expected_uncertain_count=58)

    def test_lengths_and_empty_complete_cohort_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_brier_bounds([], [], [], [], [])
        with self.assertRaises(ValueError):
            paired_brier_bounds([0.2], [], [1], [], [])
        with self.assertRaises(ValueError):
            paired_brier_bounds([0.2], [0.1], [1], [0.2], [])


if __name__ == "__main__":
    unittest.main()
