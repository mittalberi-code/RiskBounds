import unittest
from unittest.mock import patch

import protocol.target_bounds as target_bounds


def inputs():
    known = [(f"s{i:05d}", "AAA", "2014J") for i in range(6771)]
    unknown = [(f"u{i:03d}", "BBB", "2014J") for i in range(58)]
    return known, unknown


class KeyAlignedPrimaryBoundsTest(unittest.TestCase):
    def test_key_alignment_is_invariant_to_input_order(self):
        known, unknown = inputs()
        kr = [(k, .2) for k in known]
        ka = [(k, .1) for k in known]
        labels = [(k, i % 2) for i, k in enumerate(known)]
        ur = [(k, .3) for k in unknown]
        ua = [(k, .25) for k in unknown]
        with patch.object(target_bounds, "KNOWN_TARGET_KEYSET_SHA256", target_bounds._fingerprint(set(known))), \
             patch.object(target_bounds, "UNCERTAIN_TARGET_KEYSET_SHA256", target_bounds._fingerprint(set(unknown))):
            first = target_bounds.primary_target_bounds(known_unit_keys=known, uncertain_unit_keys=unknown,
                                          known_registration_scores=kr, known_activity_scores=ka,
                                          known_labels=labels, uncertain_registration_scores=ur,
                                          uncertain_activity_scores=ua)
            second = target_bounds.primary_target_bounds(known_unit_keys=known, uncertain_unit_keys=unknown,
                                           known_registration_scores=list(reversed(kr)),
                                           known_activity_scores=list(reversed(ka)), known_labels=list(reversed(labels)),
                                           uncertain_registration_scores=list(reversed(ur)),
                                           uncertain_activity_scores=list(reversed(ua)))
        self.assertEqual(first, second)

    def test_missing_duplicate_and_mismatched_keys_rejected(self):
        known, unknown = inputs()
        kr = [(k, .2) for k in known]
        ka = [(k, .1) for k in known]
        labels = [(k, i % 2) for i, k in enumerate(known)]
        ur = [(k, .3) for k in unknown]
        ua = [(k, .25) for k in unknown]
        base = dict(known_unit_keys=known, uncertain_unit_keys=unknown,
                    known_registration_scores=kr, known_activity_scores=ka, known_labels=labels,
                    uncertain_registration_scores=ur, uncertain_activity_scores=ua)
        for field, value in (("known_labels", labels[:-1]),
                             ("known_labels", labels + [labels[0]]),
                             ("uncertain_activity_scores", ua[:-1]),
                             ("uncertain_registration_scores", ur[:-1] + [(known[0], .3)]),
                             ("known_unit_keys", known[:-1]),
                             ("uncertain_unit_keys", unknown + [unknown[0]])):
            with self.subTest(field=field, length=len(value)):
                kwargs = dict(base)
                kwargs[field] = value
                with patch.object(target_bounds, "KNOWN_TARGET_KEYSET_SHA256", target_bounds._fingerprint(set(known))), \
                     patch.object(target_bounds, "UNCERTAIN_TARGET_KEYSET_SHA256", target_bounds._fingerprint(set(unknown))):
                    with self.assertRaises(ValueError):
                        target_bounds.primary_target_bounds(**kwargs)


if __name__ == "__main__":
    unittest.main()
