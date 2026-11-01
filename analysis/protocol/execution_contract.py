"""Fail-closed validation of a future fit's recorded outcome; no fitting code."""
from __future__ import annotations

import math
import numbers


def validate_fit_record(*, elapsed_seconds, overall_elapsed_seconds, timed_out, convergence_warning,
                        coefficients_finite, probabilities_valid, log_path):
    """Return a pass record or raise with the preserved log reference.

    The later runner must enforce the 300-second per-fit/1,200-second total
    deadlines externally. This function prevents a failed run from being marked
    usable; it deliberately offers no retry or parameter mutation path.
    """
    problems = []
    for name, value in (("elapsed_seconds", elapsed_seconds),
                        ("overall_elapsed_seconds", overall_elapsed_seconds)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or value < 0:
            problems.append(f"invalid_{name}")
    if timed_out:
        problems.append("timeout")
    if elapsed_seconds > 300:
        problems.append("per_fit_timeout_exceeded")
    if overall_elapsed_seconds > 1200:
        problems.append("overall_timeout_exceeded")
    if convergence_warning:
        problems.append("convergence_warning")
    if not coefficients_finite:
        problems.append("non_finite_coefficients")
    if not probabilities_valid:
        problems.append("invalid_probability")
    if problems:
        raise RuntimeError(f"fit outcome rejected ({','.join(problems)}); preserved log: {log_path}")
    return {"status": "candidate_fit_record_valid", "elapsed_seconds": elapsed_seconds,
            "log_path": str(log_path), "retry": False}
