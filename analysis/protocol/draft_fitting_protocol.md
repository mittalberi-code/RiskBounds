# Locked fitting protocol for one reviewed execution

**LOCKED FOR ONE REGISTERED RUN — no tuning or retry.** This protocol retains the audited target, split, model definitions, metrics, quarantines and explicit assumptions below. The independent reviewer approved this exact source revision for one bounded run, and the parent has provided the separate authorization recorded in [`fit_authorization.json`](../audit/fit_authorization.json). This lock does not authorize any other run or claim publication readiness.

## Objective and locked population candidate

Candidate objective: use day-28 information to predict recorded withdrawal in days 29–56 among the existing 2014J target risk set. Preserve the original 2013J source / 2014J target split, actual shared modules (AAA, BBB, DDD, EEE, FFF, GGG), and new-to-training exclusion. Do not change the outcome or population to avoid missing labels.

The candidate source fitting set is the 7,676 accepted label-complete 2013J records. Its three first-audit quarantines stay out of fitting and remain reported. The accepted target set has 6,771 known eligible/outcome records plus 58 uncertain risk-set/label records. The six 2014J shared-module rows with missing registration dates have unregistration by day 28 and are known ineligible; they are not included among the 58. The 58 are never silently removed from the primary target estimand.

## One prespecified source fit/calibration split

Use installed `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, first generated fold only (index 0), grouping by `id_student`. This creates the source fit partition and held-out source calibration partition. No other fold may be inspected or selected. Split-feasibility counts are calculated without fitting and stored in `split_diagnostics.json`.

Stop if fit or calibration lacks either class, calibration has fewer than 20 positives, students overlap between partitions, any shared module needed for the one-hot feature is unseen in the fit partition, or installed versions differ from the draft pins. Do not change the split seed, fold, module set, threshold, or population to pass these checks.

Current non-fitting feasibility check (installed pins matched): source fit has 6,141 records / 6,138 students (174 positive, 5,967 negative); calibration has 1,535 records / 1,534 students (44 positive, 1,491 negative). All six module categories appear in source fit. These are split diagnostics only, not model results. Full per-module counts and API/runtime pins are in [`split_diagnostics.json`](split_diagnostics.json).

Only the source fit partition may determine feature means/scales, module category vocabulary, model coefficients, and the source-prevalence constant. The calibration partition is reserved for sigmoid calibration. No vocabulary, scaling value, calibration parameter, threshold, or other transformation may use 2014J.

## Candidate predictors and fixed feature transforms

1. **Source-prevalence constant:** probability equals positive prevalence in source fit. It is a descriptive reference; it is not in the primary calibrated pair.
2. **Registration logistic regression:** signed registration lead time `-date_registration` (positive means early registration) and one-hot module.
3. **Registration plus VLE logistic regression:** the same registration inputs plus day-28 activity features below.

Activity uses observed VLE days 0–28 inclusive. Sum `sum_click` in bins [0,6], [7,13], [14,20], [21,27], and [28], then apply `log1p` to each. Also include `log1p(total_clicks_0_28)`, active-day count, recency `28 - last_active_day` (0 when no activity), and a separate no-activity indicator. Left-aggregate onto the eligible record population; preserve zero-activity students. Negative-day events and day-29-or-later events do not contribute.

The executable non-fitting implementation and aggregate coverage check are in [`day28_features.py`](day28_features.py) and [`feature_preflight.json`](feature_preflight.json). They enforce unique canonical `(student,module,presentation)` keys, preserve all 6,771 accepted and 58 uncertain target units, reject missing/duplicate key coverage, and write no row-level matrix or keys. SHA-256 fingerprints of the exact two canonical target key sets are frozen in [`target_bounds.py`](target_bounds.py) and checked before primary bounds are computed. The preflight asserts observed raw coverage of 206 accepted and 29 uncertain zero-activity units, with 759 ignored later-day rows across 7 uncertain units. This is a data-shape check, not a fitted transform; no vocabulary, scaler, or model parameters are learned.

Exclude grades, demographics, final status, withdrawal dates, identifiers, future activity, and optional assessment features. No feature selection, class reweighting, oversampling, threshold search, or hyperparameter search.

Use `StandardScaler` for numeric features, fit on source fit only; do not scale one-hot columns. Fit module category vocabulary on source fit only and error on unseen categories. Candidate logistic parameters for both base predictors: L2 penalty, `C=1.0`, `solver="lbfgs"`, `class_weight=None`, intercept enabled, `max_iter=1000`, `tol=1e-4`, `random_state=42`. These fixed values are draft choices, not tuned values.

## Calibration and inference

Fit each base predictor once on source fit. Use its `decision_function` score for held-out source calibration. Fit a separate one-feature unpenalized logistic sigmoid with intercept for each predictor on the same source calibration records. Freeze each predictor and its sigmoid; score target with these exact objects and do not refit the base model after calibration. Calibration must have both classes and at least 20 positive records, or stop. There is no target recalibration. Raw base probabilities are predefined secondary diagnostics; the source-prevalence constant is not a target prior correction.

Primary comparison is unweighted record-average **paired Brier improvement**, sign-coded `Brier(registration calibrated) - Brier(activity calibrated)` so positive values favor activity. Report finite-cohort identification bounds over the 58 uncertain target records using [`partial_brier_bounds.md`](partial_brier_bounds.md) and the pure function in [`brier_bounds.py`](brier_bounds.py). This requires valid fixed probabilities from both models for every uncertain record. Any unavailable or invalid probability stops the model-specific analysis; do not impute a score or use a generic fallback.

The candidate bound uses three feasible states per uncertain record and an explicit independent Cartesian-product assumption. It is conditional on fixed fitted/calibrated probabilities and accepted complete-case labels. It does not account for source selection, fitting/calibration uncertainty, future-population uncertainty, administrative recording delay, or unmodeled constraints among records. No nominal bootstrap confidence region is claimed for nonsmooth bound endpoints.

Secondary metrics are complete-case 2014J AP (with prevalence baseline), AUROC, Brier, raw-probability Brier, and a fixed 10-bin reliability table on [0,.1), …, [.9,1] (last bin includes 1). They describe only observed-label complete cases. Do not claim sorted partial-identification bounds for AP, AUROC, or calibration slope. Per-module results are descriptive; no module-specific superiority claims. No threshold metric is prespecified.

## Runtime/API evidence and remaining review questions

The execution cap is **at most two CPU threads**, **20 minutes overall**, and **5 minutes per base-estimator fit**; CPU only, no GPU and no paid resources. Keep every listed estimator parameter unchanged, including `max_iter=1000`. A convergence warning, timeout, non-finite coefficient, or invalid probability fails closed with logs preserved; do not retry under altered settings. These limits apply to this one parent-authorized run.

The gated implementation is [`locked_runner.py`](locked_runner.py). Ordinary invocation is a dry run. The only supported execution entry requires a parent authorization reference supplied by the authenticated parent task, separate from the local manifest, plus a locked configuration, reviewer approval and matching protocol/data/split/code SHA-256 values. The manifest is an audit receipt; it cannot authorize execution by itself. It applies source-fit-only `StandardScaler`/module one-hot preprocessing, fits each base model once, calibrates on held-out source scores from that same frozen model, validates target score coverage by canonical unit key, and calls the fixed 6,771/58-key partial-Brier wrapper. Per-fit work runs in a killable child process; all artifacts and row-level probabilities go only under Git-ignored `data/derived`. Each run ID/output path is claimed once with an exclusive atomic claim before fitting; claims remain after failure and prevent replay. Consumers must verify `run_manifest.json` with `verify_complete_run`: partial metric/prediction files are unusable unless both the overall completion marker and stage ledger say `complete` and all artifact hashes match. Failure records and logs remain available. This guard supports a private single-operator workflow; it is not tamper-proof, and source-owner code modification is outside its threat model. Mock-only tests cover gates, replays, deadlines, convergence warnings, non-finite outputs, no refit, source-only fitting, and partial-output failure. No real estimator, calibrator, or target prediction has been executed.

The selected runtime reports Python 3.12.14, scikit-learn 1.8.0, NumPy 2.3.5, SciPy 1.17.0, joblib 1.5.3, and threadpoolctl 3.6.0. The public constructors for `StratifiedGroupKFold`, `LogisticRegression(penalty="l2")`, `LogisticRegression(penalty=None)`, `OneHotEncoder(sparse_output=...)`, and `StandardScaler` were checked without fitting. Direct/transitive pins are in [`requirements-modeling.txt`](requirements-modeling.txt). Recheck before any future lock.

The independent reviewer accepted the finite-cohort bound subject to its stated Cartesian-product assumption, confirmed the feature and split diagnostics, and approved the fixed reliability diagnostic and execution limits. Source selection, quarantined cases, administrative recording delay and fitting uncertainty remain explicit limitations. The runner must stop if any uncertain target score is invalid or unavailable. This protocol and code are hashed before the single run. No real estimator, calibrator, or prediction has been run as of this lock.
