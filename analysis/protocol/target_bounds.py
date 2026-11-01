"""Key-aligned primary target bound wrapper; does not fit or score models."""
from __future__ import annotations

import hashlib

from protocol.brier_bounds import paired_brier_bounds

KNOWN_TARGET_COUNT = 6771
UNCERTAIN_TARGET_COUNT = 58
KNOWN_TARGET_KEYSET_SHA256 = "56b57649c21f265990e0d4331a2096b526371f822482b8b59847af6d14025ce4"
UNCERTAIN_TARGET_KEYSET_SHA256 = "de7f860d8288ad4fcdccc7e0902ce9d0dc7ee06425932c62e755fc4900e3a178"


def _key(raw):
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        raise ValueError("unit key must be (student, module, presentation)")
    result = tuple(str(part) for part in raw)
    if any(not part for part in result):
        raise ValueError("unit key fields must be non-empty")
    return result


def _keyed(rows, name):
    result = {}
    for raw_key, value in rows:
        key = _key(raw_key)
        if key in result:
            raise ValueError(f"duplicate {name} unit key")
        result[key] = value
    return result


def _fingerprint(keys):
    payload = "".join("\t".join(key) + "\n" for key in sorted(keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def primary_target_bounds(*, known_unit_keys, uncertain_unit_keys,
                          known_registration_scores, known_activity_scores,
                          known_labels, uncertain_registration_scores,
                          uncertain_activity_scores):
    """Require complete canonical-key coverage, align by key, then compute bounds.

    The exact audited target size is contractual here. Generic/small synthetic
    calculations should use ``paired_brier_bounds`` directly.
    """
    known = [_key(k) for k in known_unit_keys]
    uncertain = [_key(k) for k in uncertain_unit_keys]
    known_set, uncertain_set = set(known), set(uncertain)
    if len(known) != KNOWN_TARGET_COUNT or len(known_set) != len(known):
        raise ValueError("known target keys must be the exact 6,771 unique audited units")
    if len(uncertain) != UNCERTAIN_TARGET_COUNT or len(uncertain_set) != len(uncertain):
        raise ValueError("uncertain target keys must be the exact 58 unique audited units")
    if known_set & uncertain_set:
        raise ValueError("known and uncertain target key sets overlap")
    if _fingerprint(known_set) != KNOWN_TARGET_KEYSET_SHA256:
        raise ValueError("known target keys do not match the frozen audited key set")
    if _fingerprint(uncertain_set) != UNCERTAIN_TARGET_KEYSET_SHA256:
        raise ValueError("uncertain target keys do not match the frozen audited key set")

    kr = _keyed(known_registration_scores, "known registration score")
    ka = _keyed(known_activity_scores, "known activity score")
    ky = _keyed(known_labels, "known label")
    ur = _keyed(uncertain_registration_scores, "uncertain registration score")
    ua = _keyed(uncertain_activity_scores, "uncertain activity score")
    for name, values, expected in (("known registration scores", kr, known_set),
                                   ("known activity scores", ka, known_set),
                                   ("known labels", ky, known_set),
                                   ("uncertain registration scores", ur, uncertain_set),
                                   ("uncertain activity scores", ua, uncertain_set)):
        if values.keys() != expected:
            raise ValueError(f"{name} have missing or mismatched canonical keys")
    order_known, order_uncertain = sorted(known_set), sorted(uncertain_set)
    return paired_brier_bounds(
        [kr[k] for k in order_known], [ka[k] for k in order_known], [ky[k] for k in order_known],
        [ur[k] for k in order_uncertain], [ua[k] for k in order_uncertain],
        expected_uncertain_count=UNCERTAIN_TARGET_COUNT,
    )
