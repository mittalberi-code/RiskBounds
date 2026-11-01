# Draft: finite-cohort paired Brier bounds

**Status: DRAFT FOR REVIEW.** A separate mathematical review conditionally supports the finite sorted-prefix result under the stated assumptions. This defines a candidate primary analysis only. It does not lift the audit NO-GO, authorize fitting, calibrating, predicting, or lock the protocol.

## Estimand and states

Let R be the fixed registration-only predictor and A the fixed registration-plus-VLE predictor. For every record, define positive paired Brier improvement as registration loss minus activity loss:

`d_i(y) = (pR_i - y)^2 - (pA_i - y)^2`.

Let O contain the 6,771 accepted known-label target records and `D = sum_{i in O} d_i(y_i)`. Let U contain the 58 remaining target records with known registration by day 28, missing unregistration date, and final status `Withdrawn`. The six target records with missing registration dates have known unregistration by day 28 and are ineligible; they are not members of U.

Each record in U is allowed one of three states: ineligible because withdrawal occurred by day 28; eligible with `y=1` if withdrawal occurred on days 29–56; or eligible with `y=0` if withdrawal occurred after day 56. These are feasible-state assignments, not imputed dates or observed outcomes. The draft assumes the 58 records independently take any of these states (Cartesian product), with no cross-record event-total or timing constraints. This assumption must be reviewed; without it the state set and bounds may change.

The finite-cohort estimand is the **unweighted record-average paired Brier improvement among eligible target records**, not student-average or module-weighted improvement. If m uncertain records are eligible under one assignment, its denominator is `n+m`, with `n=6,771`.

## Exact finite optimization for frozen probabilities

For uncertain record j, compute the two eligible contributions `d_j(0)` and `d_j(1)`, then `l_j=min(d_j(0),d_j(1))` and `u_j=max(d_j(0),d_j(1))`. Sort l increasingly and u decreasingly. Let `L_m` and `U_m` be their prefix sums, including `L_0=U_0=0`. The exact endpoints are:

`lower = min_{m=0,...,58} (D + L_m)/(n+m)`

`upper = max_{m=0,...,58} (D + U_m)/(n+m)`.

The bounds are sharp for the finite Cartesian-product state space: for fixed m, the denominator is constant; selecting the m smallest minimum contributions and assigning each its minimizing label gives the attainable minimum, and the corresponding largest maxima give the attainable maximum. Taking extrema over all m gives the full bounds. Include m=0. Do not assume m=58 is extremal. The implementation uses exact rational arithmetic on the supplied fixed probability values and runs in `O(K log K)` for `K=58`.

These are **identification bounds only** for the fixed predictors. They do not cover source selection, model fitting, calibration, future-population uncertainty, administrative recording delay, or violations of the independent-state assumption. No nominal confidence region or bootstrap interval is claimed for the nonsmooth endpoints.

## Conditions before any future execution

- Both base models and their preprocessing/calibration must be frozen. The same predictor and transformations used to calibrate must score target rows; there is no post-calibration refit.
- Valid probabilities must be available for all 58 uncertain records from both models. Any unavailable, non-finite, or out-of-range score stops the model-specific bound analysis. There is no unknown-score fallback in the primary method.
- The formulas apply only to paired Brier contributions. Do not sort contributions to claim bounds for AP, AUROC, calibration slope, or other nonsmooth metrics.
- Complete-case AP, AUROC, Brier, and fixed-bin reliability diagnostics remain secondary and explicitly conditional on observed labels. They are not a substitute for the target-population bounds.
- No student-cluster bootstrap confidence region for the partial-identification endpoints is specified. Any later uncertainty analysis needs independent coverage review; bootstrap intervals on endpoints are not automatically valid.

The pure implementation in [`brier_bounds.py`](brier_bounds.py) accepts synthetic or later frozen probability arrays but performs no training, calibration, prediction, or data loading. Tests compare it to exhaustive ternary enumeration over 320 seeded synthetic cases (`K=0..7`, 131,200 total ternary assignments) using exact rational arithmetic; further tests cover sign direction, model-swap symmetry, ties/equal predictors, empty cases, and invalid scores. No fitted predictions or OULAD scores were used. The method is an elementary finite optimization result, not a new algorithm claim.
