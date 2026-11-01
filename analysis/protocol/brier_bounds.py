"""Pure finite-cohort paired-Brier bounds; contains no fit or prediction code.

Project implementation of the sorted-prefix formula in partial_brier_bounds.md.
Unknown records may be ineligible, or eligible with either binary label. Exact
Fractions are formed from the supplied probability values (binary float values
are represented by their exact rational value).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import numbers
from typing import Iterable


@dataclass(frozen=True)
class BoundResult:
    lower: Fraction
    upper: Fraction
    lower_included_unknown: int
    upper_included_unknown: int
    known_records: int
    uncertain_records: int
    known_paired_sum: Fraction


def _probability(value: object) -> Fraction:
    # bool is a Real in Python, but is never a valid probability input here.
    # Numeric strings are also rejected rather than silently coerced.
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError("probability must be a real numeric scalar, not bool or string")
    try:
        if isinstance(value, Fraction):
            result = value
        else:
            if not isinstance(value, numbers.Real):
                raise ValueError("probability must be a real numeric scalar")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("probability must be finite")
            result = Fraction.from_float(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probability is missing or invalid") from exc
    if result < 0 or result > 1:
        raise ValueError("probability must be in [0, 1]")
    return result


def _label(value: object) -> int:
    if value not in (0, 1, False, True):
        raise ValueError("known labels must be binary 0/1")
    return int(value)


def _loss_improvement(p_registration: Fraction, p_activity: Fraction, y: int) -> Fraction:
    return (p_registration - y) ** 2 - (p_activity - y) ** 2


def paired_brier_bounds(
    known_registration_probabilities: Iterable[object],
    known_activity_probabilities: Iterable[object],
    known_labels: Iterable[object],
    uncertain_registration_probabilities: Iterable[object],
    uncertain_activity_probabilities: Iterable[object],
    *,
    expected_uncertain_count: int | None = None,
) -> BoundResult:
    """Bound record-average Brier improvement for a frozen pair of predictors.

    Positive improvement means the activity predictor's Brier loss is lower.
    Each uncertain row independently is ineligible, or eligible with y=0/1.
    Inputs for every uncertain row must have valid scores; no fallback is used.
    """
    kr = [_probability(x) for x in known_registration_probabilities]
    ka = [_probability(x) for x in known_activity_probabilities]
    ky = [_label(x) for x in known_labels]
    ur = [_probability(x) for x in uncertain_registration_probabilities]
    ua = [_probability(x) for x in uncertain_activity_probabilities]
    if not kr:
        raise ValueError("at least one known eligible record is required")
    if len(kr) != len(ka) or len(kr) != len(ky):
        raise ValueError("known probability/label lengths differ")
    if len(ur) != len(ua):
        raise ValueError("uncertain predictor probability lengths differ")
    if expected_uncertain_count is not None and len(ur) != expected_uncertain_count:
        raise ValueError("uncertain-record probability coverage is incomplete")

    known_sum = sum((_loss_improvement(r, a, y) for r, a, y in zip(kr, ka, ky)), Fraction())
    lower_contributions = []
    upper_contributions = []
    for r, a in zip(ur, ua):
        d0 = _loss_improvement(r, a, 0)
        d1 = _loss_improvement(r, a, 1)
        lower_contributions.append(min(d0, d1))
        upper_contributions.append(max(d0, d1))

    lower_prefix = sorted(lower_contributions)
    upper_prefix = sorted(upper_contributions, reverse=True)
    n = len(kr)
    best_lower = known_sum / n
    best_upper = best_lower
    lower_m = upper_m = 0
    low_sum = high_sum = Fraction()
    for m, (low, high) in enumerate(zip(lower_prefix, upper_prefix), 1):
        low_sum += low
        high_sum += high
        lower_candidate = (known_sum + low_sum) / (n + m)
        upper_candidate = (known_sum + high_sum) / (n + m)
        if lower_candidate < best_lower:
            best_lower, lower_m = lower_candidate, m
        if upper_candidate > best_upper:
            best_upper, upper_m = upper_candidate, m
    return BoundResult(best_lower, best_upper, lower_m, upper_m, n, len(ur), known_sum)
